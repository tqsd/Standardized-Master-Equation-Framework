from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from smef.core.ir.ops import OpExpr, SymbolOp
from smef.core.ir.terms import Term, TermKind
from smef.core.ir.coeffs import CallableCoeffUnits, ConstCoeff
from smef.core.ir.protocols import OpMaterializeContextProto
from smef.core.model.protocols import (
    CompileBundle,
    CompilableModelProto,
    MaterializeBundle,
    ModeRegistryProto,
    TermCatalogProto,
)
from smef.core.units import Q, UnitSystem
from smef.engine import SimulationEngine


# ----------------------------
# Minimal helpers / catalogs
# ----------------------------


@dataclass(frozen=True)
class FrozenCatalog(TermCatalogProto):
    _terms: tuple[Term, ...]

    @property
    def all_terms(self) -> Sequence[Term]:
        return self._terms


@dataclass(frozen=True)
class TLSModes(ModeRegistryProto):
    def dims(self) -> Sequence[int]:
        return (2,)

    def index_of(self, key: Any) -> int:
        if key == "tls":
            return 0
        raise KeyError(key)

    @property
    def channels(self) -> Optional[Sequence[Any]]:
        return ("tls",)


@dataclass
class TLSMaterializer(OpMaterializeContextProto):
    def resolve_symbol(self, symbol: str, dims: Sequence[int]) -> np.ndarray:
        if tuple(dims) != (2,):
            raise ValueError(f"TLSMaterializer expects dims=(2,), got {dims}")

        # basis: |g> = [1,0], |e> = [0,1]
        proj_g = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        proj_e = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)

        # sigma_plus = |e><g|, sigma_minus = |g><e|
        sigma_plus = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
        sigma_minus = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
        sigma_x = sigma_plus + sigma_minus

        if symbol == "proj_g":
            return proj_g
        if symbol == "proj_e":
            return proj_e
        if symbol == "sigma_plus":
            return sigma_plus
        if symbol == "sigma_minus":
            return sigma_minus
        if symbol == "sigma_x":
            return sigma_x

        raise KeyError(symbol)

    def resolve_embedded(self, embedded: Any, dims: Sequence[int]) -> np.ndarray:
        raise NotImplementedError("Not needed for this 1-mode example")


# ----------------------------
# Unitful driven TLS model
# ----------------------------


@dataclass(frozen=True)
class UnitfulDrivenTLS(CompilableModelProto):
    """
    Two-level system with:
    - detuning term: (delta/2) * |e><e|
    - gaussian drive: (Omega(t)/2) * sigma_x
    - decay collapse: sqrt(gamma) * sigma_minus   (Lindblad)
    All parameters are unitful and lowered via UnitSystem at compile time.
    """

    # unitful inputs (pint quantities)
    delta: Any  # rad/s
    omega_rabi: Any  # rad/s (peak)
    gamma: Any  # 1/s

    t0: Any  # s
    sigma: Any  # s

    def compile_bundle(self, *, units: UnitSystem) -> CompileBundle:
        modes = TLSModes()

        # ---- Static Hamiltonian (solver units) ----
        # H_det = (delta/2) * |e><e|
        delta_solver = units.omega_to_solver(self.delta)  # dimensionless
        H_det = Term(
            kind=TermKind.H,
            op=OpExpr.atom(SymbolOp("proj_e")),
            coeff=ConstCoeff(0.5 * complex(delta_solver)),
            label="H_det",
            meta={"delta": str(self.delta)},
        )

        # ---- Drive Hamiltonian coefficient (solver units) ----
        # Drive coefficient should return Omega_solver(t_solver).
        # We implement it as unit-aware coeff: it receives time_unit_s and can map
        # t_solver -> t_s internally.
        t0_s = (
            float(Q(self.t0, "s").magnitude)
            if hasattr(self.t0, "to")
            else float(self.t0)
        )
        sigma_s = (
            float(Q(self.sigma, "s").magnitude)
            if hasattr(self.sigma, "to")
            else float(self.sigma)
        )

        # We'll convert the peak Omega to solver units once, so the function returns solver values directly.
        omega0_solver = units.omega_to_solver(self.omega_rabi)  # dimensionless

        def omega_solver(t_solver: np.ndarray, time_unit_s: float) -> np.ndarray:
            t_s = np.asarray(t_solver, dtype=float) * float(time_unit_s)
            x = (t_s - t0_s) / sigma_s
            y = omega0_solver * np.exp(-0.5 * x * x)
            return np.asarray(y, dtype=complex).reshape(len(t_solver))

        H_drive = Term(
            kind=TermKind.H,
            op=OpExpr.atom(SymbolOp("sigma_x")),
            coeff=CallableCoeffUnits(omega_solver),
            label="H_drive_gauss",
            meta={
                "omega_rabi": str(self.omega_rabi),
                "t0": str(self.t0),
                "sigma": str(self.sigma),
            },
        )

        # ---- Collapse (solver units) ----
        # Collapse operator for spontaneous emission: sqrt(gamma_solver) * sigma_minus
        gamma_solver = units.rate_to_solver(self.gamma)  # dimensionless
        c_pref = np.sqrt(float(gamma_solver))

        C_decay = Term(
            kind=TermKind.C,
            # Put sqrt(gamma) into the operator via OpExpr.scale
            op=OpExpr.scale(complex(c_pref), OpExpr.atom(
                SymbolOp("sigma_minus"))),
            coeff=None,  # constant 1
            label="C_decay",
            meta={"gamma": str(self.gamma)},
        )

        ham = FrozenCatalog((H_det, H_drive))
        col = FrozenCatalog((C_decay,))
        return CompileBundle(modes=modes, hamiltonian=ham, collapse=col)

    def materialize_bundle(self) -> MaterializeBundle:
        return MaterializeBundle(ops=TLSMaterializer())


def main() -> None:
    # Pick a solver time unit: 1 solver unit = 1 ps
    time_unit_s = float(Q(1.0, "ps").to("s").magnitude)

    # Unitful parameters
    delta = Q(0.0, "rad/s")
    # large so you see oscillations on ps scale
    omega_rabi = Q(2.0e11, "rad/s")
    gamma = Q(1.0e10, "1/s")  # decay rate

    # Pulse parameters
    t0 = Q(5.0, "ps")
    sigma = Q(1.0, "ps")

    model = UnitfulDrivenTLS(
        delta=delta,
        omega_rabi=omega_rabi,
        gamma=gamma,
        t0=t0,
        sigma=sigma,
    )

    # Solver grid in solver units (unitless)
    tlist = np.linspace(0.0, 10.0, 1200)

    # Initial state: excited
    rho0 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)

    engine = SimulationEngine(audit=True)

    res = engine.run(
        model,
        tlist=tlist,
        time_unit_s=time_unit_s,
        rho0=rho0,
        drives=None,
        solve_options={"progress_bar": "tqdm"},
    )

    # Observable: P_e = Tr(|e><e| rho)
    proj_e = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)

    if res.states is None:
        raise RuntimeError(
            "No states returned by adapter; enable state output in your adapter."
        )

    pe = np.empty(len(res.tlist), dtype=float)
    for i, st in enumerate(res.states):
        if hasattr(st, "full"):
            rho = np.asarray(st.full(), dtype=complex)
        else:
            rho = np.asarray(st, dtype=complex)
        pe[i] = float(np.real(np.trace(proj_e @ rho)))

    # Plot vs physical time in ps (for sanity)
    t_ps = (np.asarray(res.tlist, dtype=float) * time_unit_s) / float(
        Q(1.0, "ps").to("s").magnitude
    )

    plt.figure()
    plt.plot(t_ps, pe)
    plt.xlabel("t (ps)")
    plt.ylabel("P_e")
    plt.title("Unitful driven TLS with decay (SMEF)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
