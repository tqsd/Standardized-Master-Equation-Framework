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

        proj_g = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        proj_e = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)

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


@dataclass(frozen=True)
class UnitfulDrivenTLS(CompilableModelProto):
    """
    Two-level system with:
    - detuning term: (delta/2) * |e><e|
    - gaussian drive: Omega(t) * sigma_x   (note: if you want Omega/2, scale it)
    - decay collapse: sqrt(gamma) * sigma_minus
    - observable: P_e = |e><e|
    """

    delta: Any  # rad/s
    omega_rabi: Any  # rad/s (peak)
    gamma: Any  # 1/s
    t0: Any  # s
    sigma: Any  # s

    def compile_bundle(self, *, units: UnitSystem) -> CompileBundle:
        modes = TLSModes()

        delta_solver = units.omega_to_solver(self.delta)
        H_det = Term(
            kind=TermKind.H,
            op=OpExpr.atom(SymbolOp("proj_e")),
            coeff=ConstCoeff(0.5 * complex(delta_solver)),
            label="H_det",
        )

        t0_s = float(Q(self.t0, "s").to("s").magnitude)
        sigma_s = float(Q(self.sigma, "s").to("s").magnitude)
        omega0_solver = units.omega_to_solver(self.omega_rabi)

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
        )

        gamma_solver = units.rate_to_solver(self.gamma)
        c_pref = np.sqrt(float(gamma_solver))
        C_decay = Term(
            kind=TermKind.C,
            op=OpExpr.scale(complex(c_pref), OpExpr.atom(SymbolOp("sigma_minus"))),
            coeff=None,
            label="C_decay",
        )

        # Observables (E terms): this is what will populate res.expect["P_e"]
        P_e = Term(
            kind=TermKind.E,
            op=OpExpr.atom(SymbolOp("proj_e")),
            coeff=None,
            label="P_e",
        )

        ham = FrozenCatalog((H_det, H_drive))
        col = FrozenCatalog((C_decay,))
        obs = FrozenCatalog((P_e,))

        return CompileBundle(
            modes=modes, hamiltonian=ham, collapse=col, observables=obs
        )

    def materialize_bundle(self) -> MaterializeBundle:
        return MaterializeBundle(ops=TLSMaterializer())


def main() -> None:
    time_unit_s = float(Q(1.0, "ps").to("s").magnitude)

    model = UnitfulDrivenTLS(
        delta=Q(0.0, "rad/s"),
        omega_rabi=Q(2.0e11, "rad/s"),
        gamma=Q(1.0e10, "1/s"),
        t0=Q(5.0, "ps"),
        sigma=Q(1.0, "ps"),
    )

    tlist = np.linspace(0.0, 10.0, 1200)

    rho0 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)

    engine = SimulationEngine(audit=True)

    res = engine.run(
        model,
        tlist=tlist,
        time_unit_s=time_unit_s,
        rho0=rho0,
        drives=None,
        solve_options={
            "progress_bar": "tqdm",
            # allow user override; adapter setdefault will apply if omitted
            "qutip_options": {"store_states": False, "store_final_state": True},
        },
    )

    # ---- Final state (single matrix) ----
    if res.states is not None:
        rho_final = np.asarray(res.states, dtype=complex)
        print("Final state rho(t_end):")
        print(rho_final)
        print("Trace:", np.trace(rho_final))
        print("Eigenvalues:", np.linalg.eigvals(rho_final))
        # optional: compare to final observable value if present
        if "P_e" in res.expect:
            print("P_e(t_end) from expect:", float(np.real(res.expect["P_e"][-1])))
    else:
        print("No final state returned (states is None).")

    # ---- Time trace comes only from observables (res.expect) ----
    if "P_e" not in res.expect:
        raise RuntimeError("Observable P_e not found. Did you provide E-terms?")

    pe = np.asarray(res.expect["P_e"], dtype=float)

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
