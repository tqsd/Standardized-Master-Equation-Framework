from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from smef.core.ir.ops import OpExpr, SymbolOp
from smef.core.ir.terms import Term, TermKind
from smef.core.ir.coeffs import CallableCoeff, ConstCoeff
from smef.core.model.protocols import (
    CompileBundle,
    CompilableModelProto,
    MaterializeBundle,
    ModeRegistryProto,
    OpMaterializeContextProto,
    TermCatalogProto,
)
from smef.engine import SimulationEngine


@dataclass(frozen=True)
class FrozenCatalog(TermCatalogProto):
    _terms: tuple[Term, ...]

    @property
    def all_terms(self) -> Sequence[Term]:
        return self._terms


@dataclass(frozen=True)
class TLSModes(ModeRegistryProto):
    def num_modes(self) -> int:
        return 1

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
        sigma_plus = np.array([[0.0, 0.0], [1.0, 0.0]],
                              dtype=complex)  # |e><g|
        sigma_minus = np.array([[0.0, 1.0], [0.0, 0.0]],
                               dtype=complex)  # |g><e|
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
class DrivenTLSModel(CompilableModelProto):
    delta: float  # detuning in rad/s (physical)
    omega_rabi: float  # drive amplitude in rad/s (physical)
    t0_s: float
    sigma_s: float
    time_unit_s: float

    def compile_bundle(self) -> CompileBundle:
        modes = TLSModes()

        # Static detuning term: H0 = (delta/2) * |e><e|  (simple choice)
        H_det = Term(
            kind=TermKind.H,
            op=OpExpr.atom(SymbolOp("proj_e")),
            coeff=ConstCoeff(0.5 * self.delta),
            label="H_det",
        )

        # Drive term: H_drive(t) = (Omega(t)/2) * sigma_x
        # Here we use a Gaussian envelope. You can swap this to sin/cos easily.
        def omega_t(t_s: float) -> complex:
            x = (t_s - self.t0_s) / self.sigma_s
            return complex(self.omega_rabi * np.exp(-0.5 * x * x))

        # CallableCoeff should interpret its input "t" as solver time; we convert via time_unit_s in engine.
        # If your CallableCoeff already expects solver time, then define omega over solver time.
        # We'll do solver-time here and convert inside with time_unit_s below using the same formula but t_s = t*time_unit_s.
        def omega_solver(t_solver: np.ndarray) -> np.ndarray:
            t_s = np.asarray(t_solver, dtype=float) * self.time_unit_s
            x = (t_s - self.t0_s) / self.sigma_s
            return (self.omega_rabi * np.exp(-0.5 * x * x)).astype(complex)

        H_drive = Term(
            kind=TermKind.H,
            op=OpExpr.atom(SymbolOp("sigma_x")),
            coeff=CallableCoeff(omega_solver),
            label="H_drive",
        )

        ham = FrozenCatalog((H_det, H_drive))
        return CompileBundle(modes=modes, hamiltonian=ham)

    def materialize_bundle(self) -> MaterializeBundle:
        return MaterializeBundle(ops=TLSMaterializer())


def main() -> None:
    # Physical parameters
    delta = 0.0  # rad/s
    omega_rabi = 2.0  # rad/s (peak)
    time_unit_s = 1.0  # solver time unit in seconds

    # Pulse params in seconds
    t0_s = 5.0
    sigma_s = 1.0

    model = DrivenTLSModel(
        delta=delta,
        omega_rabi=omega_rabi,
        t0_s=t0_s,
        sigma_s=sigma_s,
        time_unit_s=time_unit_s,
    )

    # Solver time grid (unitless)
    tlist = np.linspace(0.0, 10.0, 1200)

    # Initial state: excited, so you can see both driven dynamics and (if you add decay) relaxation
    rho0 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)

    engine = SimulationEngine(audit=False)

    # Add observable: P_e = <proj_e>
    # Easiest: just include it as an observable catalog in the model for now.
    # If your pipeline supports passing observables separately, use that.
    # For minimalism, we hack it by compiling once, then injecting e_terms would be messy,
    # so instead just compute from states below.
    res = engine.run(
        model,
        tlist=tlist,
        time_unit_s=time_unit_s,
        rho0=rho0,
        solve_options={"progress_bar": "tqdm"},
    )

    # Compute P_e(t) from states (density matrices)
    # QuTiP returns Qobj states; support both Qobj and ndarray-like.
    proj_e = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)

    pe = []
    if res.states is None:
        raise RuntimeError(
            "No states returned; enable states in adapter if you disabled them."
        )
    for st in res.states:
        if hasattr(st, "full"):
            rho = np.asarray(st.full(), dtype=complex)
        else:
            rho = np.asarray(st, dtype=complex)
        pe.append(float(np.real(np.trace(proj_e @ rho))))
    pe = np.asarray(pe, dtype=float)

    plt.figure()
    plt.plot(res.tlist, pe)
    plt.xlabel("t (solver units)")
    plt.ylabel("P_e")
    plt.title("Driven two-level system")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
