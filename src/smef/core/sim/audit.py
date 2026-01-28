from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from smef.core.sim.types import CompiledTermDense, MEProblemDense
from smef.core.ir.coeffs import eval_coeff_any


@dataclass(frozen=True)
class AuditOptions:
    max_terms: int = 200
    top_entries: int = 6
    check_shapes: bool = True
    check_hermitian_H: bool = True
    hermitian_atol: float = 1e-10
    coeff_stats: bool = True
    coeff_area: bool = True


def _prod_int(xs: Sequence[int]) -> int:
    out = 1
    for x in xs:
        out *= int(x)
    return out


def _top_abs_entries(
    mat: np.ndarray, k: int
) -> Sequence[Tuple[float, Tuple[int, int], complex]]:
    m = np.asarray(mat)
    a = np.abs(m).ravel()
    if a.size == 0:
        return []
    k = min(int(k), int(a.size))
    # partial selection then sort those
    idx = np.argpartition(a, -k)[-k:]
    idx = idx[np.argsort(a[idx])[::-1]]
    out = []
    n = m.shape[1]
    for flat in idx:
        i = int(flat // n)
        j = int(flat % n)
        out.append((float(abs(m[i, j])), (i, j), complex(m[i, j])))
    return out


def _fro_norm(mat: np.ndarray) -> float:
    m = np.asarray(mat, dtype=complex)
    return float(np.linalg.norm(m.ravel()))


def audit_problem_dense(
    problem: MEProblemDense,
    *,
    options: Optional[AuditOptions] = None,
) -> Dict[str, Any]:
    """
    Print-like audit (returns a structured dict) for a dense compiled ME problem.

    This does not assume any particular physics model. It's meant to catch:
    - shape/dim inconsistencies
    - missing/odd coefficients
    - huge/small operator norms
    - non-Hermitian Hamiltonian terms (optional)
    - coefficient ranges/areas, peaks
    """

    opt = options or AuditOptions()
    tlist = np.asarray(problem.tlist, dtype=float)
    dims = tuple(int(x) for x in problem.dims)
    D = _prod_int(dims)

    report: Dict[str, Any] = {}
    report["dims"] = dims
    report["D"] = D
    report["tlist_N"] = int(len(tlist))
    if len(tlist) >= 2:
        report["tlist_range"] = (float(tlist[0]), float(tlist[-1]))
        report["dt_min"] = float(np.min(np.diff(tlist)))
        report["dt_max"] = float(np.max(np.diff(tlist)))
    else:
        report["tlist_range"] = None

    if problem.rho0 is not None:
        rho0 = np.asarray(problem.rho0, dtype=complex)
        report["rho0_shape"] = tuple(rho0.shape)
        if opt.check_shapes and rho0.shape != (D, D):
            raise ValueError(f"rho0 has shape {rho0.shape}, expected {(D, D)}")

    def _audit_terms(kind: str, terms: Tuple[CompiledTermDense, ...]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"count": int(len(terms)), "terms": []}
        n_take = min(opt.max_terms, len(terms))
        for idx in range(n_take):
            term = terms[idx]
            op = np.asarray(term.op, dtype=complex)
            item: Dict[str, Any] = {
                "index": idx,
                "label": term.label,
                "op_shape": tuple(op.shape),
                "op_fro_norm": _fro_norm(op),
            }

            if opt.check_shapes and op.shape != (D, D):
                raise ValueError(
                    f"{kind} term {idx} op has shape {
                        op.shape}, expected {(D, D)}"
                )

            if opt.check_hermitian_H and kind == "H":
                # Check hermiticity of the operator matrix itself (not including coeff).
                herm_err = np.max(np.abs(op - op.conj().T))
                item["op_hermitian_max_abs_err"] = float(herm_err)
                item["op_is_hermitian"] = bool(herm_err <= opt.hermitian_atol)

            if opt.coeff_stats:
                coeff = eval_coeff_any(
                    term.coeff, tlist, time_unit_s=problem.time_unit_s
                )
                item["coeff_min_abs"] = float(np.min(np.abs(coeff)))
                item["coeff_max_abs"] = float(np.max(np.abs(coeff)))
                peak_i = int(np.argmax(np.abs(coeff))) if len(coeff) else 0
                item["coeff_peak_index"] = peak_i
                item["coeff_peak_t"] = float(
                    tlist[peak_i]) if len(tlist) else None
                item["coeff_peak_val"] = (
                    complex(coeff[peak_i]) if len(coeff) else 0.0 + 0.0j
                )

                if opt.coeff_area and len(tlist) >= 2:
                    # trapezoidal integral in solver units
                    item["coeff_area"] = complex(np.trapezoid(coeff, tlist))

            item["op_top_entries"] = _top_abs_entries(op, opt.top_entries)
            out["terms"].append(item)

        if len(terms) > n_take:
            out["truncated"] = int(len(terms) - n_take)
        return out

    report["H"] = _audit_terms("H", problem.h_terms)
    report["C"] = _audit_terms("C", problem.c_terms)
    report["E"] = _audit_terms("E", problem.e_terms)
    return report
