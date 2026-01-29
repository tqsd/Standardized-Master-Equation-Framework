from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from smef.core.sim.eval import eval_coeff
from smef.core.sim.protocols import SolverAdapterProto
from smef.core.sim.types import MEProblemDense, MESolveResult


@dataclass
class QuTiPAdapter(SolverAdapterProto):
    """
    QuTiP adapter.

    Conventions:
    - For any term kind, effective operator is: coeff(t) * op
    - This applies to H terms and collapse (C) terms equally.
    """

    interp: str = "linear"  # "nearest" or "linear"

    # Storage preferences (QuTiP 5 data layer)
    # e.g. "csr", "dense", or None (no conversion)
    op_dtype: Optional[str] = "csr"
    # often leave rho dense; set to "csr" if desired
    rho_dtype: Optional[str] = None

    # Constant detection tolerance (helps avoid false time-dependence)
    const_atol: float = 0.0
    const_rtol: float = 0.0

    def solve(
        self,
        problem: MEProblemDense,
        *,
        options: Optional[Mapping[str, Any]] = None,
    ) -> MESolveResult:
        import qutip as qt  # type: ignore

        tlist = np.asarray(problem.tlist, dtype=float)

        if problem.rho0 is None:
            raise ValueError("MEProblemDense.rho0 is required for QuTiPAdapter")

        dims = list(problem.dims)
        rho0 = self._toq(
            qt,
            np.asarray(problem.rho0, dtype=complex),
            dims=dims,
            dtype=self.rho_dtype,
        )

        H = self._build_hamiltonian(problem, qt)
        c_ops = self._build_collapse_ops(problem, qt)
        e_ops, e_keys = self._build_e_ops(problem, qt)

        mesolve_options: Dict[str, Any] = dict(
            options.get("qutip_options", {}) if options else {}
        )
        if options and "progress_bar" in options:
            mesolve_options["progress_bar"] = options["progress_bar"]

        mesolve_options.setdefault("store_states", False)
        mesolve_options.setdefault("store_final_state", True)

        res = qt.mesolve(
            H, rho0, tlist, c_ops, e_ops=e_ops, args={}, options=mesolve_options
        )

        expect: Dict[str, np.ndarray] = {}
        if res.expect is not None:
            for k, arr in zip(e_keys, res.expect):
                expect[k] = np.asarray(arr)

        final_qobj = getattr(res, "final_state", None)
        states_out = None
        if final_qobj is not None:
            # final_qobj is a qutip.Qobj; .full() returns a dense np.ndarray
            states_out = np.asarray(final_qobj.full(), dtype=complex)

        return MESolveResult(
            tlist=tlist,
            states=states_out,
            expect=expect,
            meta={
                "backend": "qutip",
                "op_dtype": self.op_dtype,
                "rho_dtype": self.rho_dtype,
            },
        )

    def _toq(
        self, qt: Any, mat: np.ndarray, *, dims: list[int], dtype: Optional[str]
    ) -> Any:
        q = qt.Qobj(mat, dims=[dims, dims])
        if dtype:
            q = q.to(dtype)
        return q

    def _build_hamiltonian(self, problem: MEProblemDense, qt: Any) -> Any:
        tlist = np.asarray(problem.tlist, dtype=float)
        dims = list(problem.dims)
        D = problem.D

        H0 = np.zeros((D, D), dtype=complex)
        H_td = []

        for term in problem.h_terms:
            op = np.asarray(term.op, dtype=complex)
            coeff = eval_coeff(term, tlist, time_unit_s=problem.time_unit_s)

            if _is_constant(coeff, atol=self.const_atol, rtol=self.const_rtol):
                H0 += complex(coeff[0]) * op
            else:
                f = self._make_time_func(tlist, coeff)
                H_td.append([self._toq(qt, op, dims=dims, dtype=self.op_dtype), f])

        if H_td:
            return [self._toq(qt, H0, dims=dims, dtype=self.op_dtype)] + H_td
        return self._toq(qt, H0, dims=dims, dtype=self.op_dtype)

    def _build_collapse_ops(self, problem: MEProblemDense, qt: Any) -> Any:
        tlist = np.asarray(problem.tlist, dtype=float)
        dims = list(problem.dims)

        c_ops = []
        for term in problem.c_terms:
            op = np.asarray(term.op, dtype=complex)
            coeff = eval_coeff(term, tlist, time_unit_s=problem.time_unit_s)

            if _is_constant(coeff, atol=self.const_atol, rtol=self.const_rtol):
                c_ops.append(
                    self._toq(
                        qt, complex(coeff[0]) * op, dims=dims, dtype=self.op_dtype
                    )
                )
            else:
                f = self._make_time_func(tlist, coeff)
                c_ops.append([self._toq(qt, op, dims=dims, dtype=self.op_dtype), f])

        return c_ops

    def _build_e_ops(
        self, problem: MEProblemDense, qt: Any
    ) -> Tuple[Any, Tuple[str, ...]]:
        dims = list(problem.dims)

        e_ops = []
        keys = []
        for i, term in enumerate(problem.e_terms):
            label = term.label if term.label else f"E[{i}]"
            e_ops.append(
                self._toq(
                    qt,
                    np.asarray(term.op, dtype=complex),
                    dims=dims,
                    dtype=self.op_dtype,
                )
            )
            keys.append(label)

        return e_ops, tuple(keys)

    def _make_time_func(self, tlist: np.ndarray, coeff: np.ndarray):
        if self.interp == "nearest":

            def f(t: float, args: Any) -> complex:
                idx = _nearest_index(tlist, float(t))
                return complex(coeff[idx])

            return f

        t0 = float(tlist[0])
        t1 = float(tlist[-1])
        re = np.asarray(np.real(coeff), dtype=float)
        im = np.asarray(np.imag(coeff), dtype=float)

        def f(t: float, args: Any) -> complex:
            tt = float(t)
            if tt <= t0:
                return complex(re[0], im[0])
            if tt >= t1:
                return complex(re[-1], im[-1])
            r = float(np.interp(tt, tlist, re))
            j = float(np.interp(tt, tlist, im))
            return complex(r, j)

        return f


def _is_constant(coeff: np.ndarray, *, atol: float = 0.0, rtol: float = 0.0) -> bool:
    if coeff.size == 0:
        return True
    c0 = coeff[0]
    return bool(np.allclose(coeff, c0, atol=atol, rtol=rtol))


def _nearest_index(tlist: np.ndarray, t: float) -> int:
    i = int(np.searchsorted(tlist, t, side="left"))
    if i <= 0:
        return 0
    if i >= len(tlist):
        return len(tlist) - 1
    left = float(tlist[i - 1])
    right = float(tlist[i])
    if (t - left) <= (right - t):
        return i - 1
    return i
