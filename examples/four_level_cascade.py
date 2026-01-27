from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from smef.core.ir.ops import EmbeddedKron, LocalSymbolOp, OpExpr, SymbolOp
from smef.core.ir.terms import Term, TermKind
from smef.core.ir.coeffs import ConstCoeff
from smef.core.ir.protocols import OpMaterializeContextProto
from smef.core.model.protocols import (
    CompileBundle,
    CompilableModelProto,
    MaterializeBundle,
    ModeRegistryProto,
    TermCatalogProto,
)
from smef.core.units import Q, UnitSystem, magnitude
from smef.engine import SimulationEngine
from smef.core.ir.coeffs import CallableCoeffUnits
from smef.core.ir.ops import EmbeddedKron, LocalSymbolOp, OpExpr
from smef.core.ir.terms import Term, TermKind


@dataclass(frozen=True)
class FrozenCatalog(TermCatalogProto):
    _terms: tuple[Term, ...]

    @property
    def all_terms(self) -> Sequence[Term]:
        return self._terms


@dataclass(frozen=True)
class CascadeModes(ModeRegistryProto):
    def dims(self) -> Sequence[int]:
        # (qd, m1, m2)
        return (4, 2, 2)

    def index_of(self, key: Any) -> int:
        if key == "qd":
            return 0
        if key == "m1":
            return 1
        if key == "m2":
            return 2
        raise KeyError(key)

    @property
    def channels(self) -> Optional[Sequence[Any]]:
        return ("qd", "m1", "m2")


@dataclass
class CascadeMaterializer(OpMaterializeContextProto):

    def resolve_symbol(self, symbol: str, dims: Sequence[int]) -> np.ndarray:
        dims_t = tuple(int(x) for x in dims)

        if dims_t == (4,):
            qd = self._qd_local_ops()
            if symbol in qd:
                return np.asarray(qd[symbol], dtype=complex)
            raise KeyError(
                f"{symbol} (dims={dims_t}) qd_keys={
                    sorted(qd.keys())}"
            )

        if dims_t == (2,):
            m = self._mode_local_ops()
            if symbol in m:
                return np.asarray(m[symbol], dtype=complex)
            raise KeyError(
                f"{symbol} (dims={dims_t}) mode_keys={
                    sorted(m.keys())}"
            )

        raise KeyError(f"{symbol} (dims={dims_t}) unsupported")

    def resolve_embedded(self, embedded: Any, dims: Sequence[int]) -> np.ndarray:
        # With your current materialize.py this likely won't be called for EmbeddedKron.
        # Keep it for future compatibility.
        raise NotImplementedError(
            "resolve_embedded not used by current materializer")

    @staticmethod
    def _qd_local_ops() -> dict[str, np.ndarray]:
        # Basis order: [G, X1, X2, XX]
        G = 0
        X1 = 1
        X2 = 2
        XX = 3

        def ket(i: int) -> np.ndarray:
            v = np.zeros((4,), dtype=complex)
            v[i] = 1.0 + 0.0j
            return v

        def op(i: int, j: int) -> np.ndarray:
            # |i><j|
            return np.outer(ket(i), np.conjugate(ket(j)))

        proj_G = op(G, G)
        proj_X1 = op(X1, X1)
        proj_X2 = op(X2, X2)
        proj_XX = op(XX, XX)

        # Down transitions for the cascade
        # XX -> X1, XX -> X2
        t_X1_XX = op(X1, XX)
        t_X2_XX = op(X2, XX)
        # X1 -> G, X2 -> G
        t_G_X1 = op(G, X1)
        t_G_X2 = op(G, X2)
        t_XX_G = op(XX, G)  # |XX><G|
        t_G_XX = op(G, XX)  # |G><XX|
        sigma_x_G_XX = t_XX_G + t_G_XX

        return {
            "proj_G": proj_G,
            "proj_X1": proj_X1,
            "proj_X2": proj_X2,
            "proj_XX": proj_XX,
            "t_X1_XX": t_X1_XX,
            "t_X2_XX": t_X2_XX,
            "t_G_X1": t_G_X1,
            "t_G_X2": t_G_X2,
            "t_XX_G": t_XX_G,
            "t_G_XX": t_G_XX,
            "sx_G_XX": sigma_x_G_XX,
        }

    @staticmethod
    def _mode_local_ops() -> dict[str, np.ndarray]:
        # Truncated 0/1 photon space
        # a = |0><1|, adag = |1><0|, n = diag(0,1)
        a = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
        adag = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
        n = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
        return {"a": a, "adag": adag, "n": n}


@dataclass(frozen=True)
class FourLevelCascadeModel(CompilableModelProto):
    # Unitful rates (pint quantities)
    gamma_xx: Any  # 1/s total XX -> X* (split between X1 and X2)
    gamma_x: Any  # 1/s total X* -> G (split between X1 and X2)

    # drive (unitful)
    omega_drive_rad_s: Any  # rad/s (peak)
    t0_s: Any  # s
    sigma_s: Any  # s

    # Optional: add simple energy splittings (unitful, as energies or rad/s).
    # For this verification example we keep H=0 to focus on the cascade.
    # If you want, you can add detunings via units.omega_to_solver(...) here.
    def omega_pulse_solver(
        self, t_solver: np.ndarray, *, time_unit_s: float
    ) -> np.ndarray:
        t_s = np.asarray(t_solver, dtype=float) * float(time_unit_s)
        t0 = magnitude(self.t0_s, "s")
        sig = magnitude(self.sigma_s, "s")
        omega0 = magnitude(self.omega_drive_rad_s, "rad/s")
        x = (t_s - t0) / sig
        omega_rad_s = omega0 * np.exp(-0.5 * x * x)
        return (omega_rad_s * float(time_unit_s)).astype(complex)

    def compile_bundle(self, *, units: UnitSystem) -> CompileBundle:
        modes = CascadeModes()

        # No coherent Hamiltonian terms for now (unit sanity focuses on collapse)

        H_drive = Term(
            kind=TermKind.H,
            op=OpExpr.scale(
                0.5 + 0.0j,
                OpExpr.atom(
                    EmbeddedKron(indices=(0,), locals=(
                        LocalSymbolOp("sx_G_XX"),))
                ),
            ),
            coeff=CallableCoeffUnits(
                lambda t, tu: self.omega_pulse_solver(t, time_unit_s=tu)
            ),
            label="H_drive_G_XX",
        )

        ham = FrozenCatalog((H_drive,))
        # Split rates equally across the two branches
        gamma_xx_solver = float(units.rate_to_solver(self.gamma_xx))
        gamma_x_solver = float(units.rate_to_solver(self.gamma_x))

        # Each branch gets half the rate
        c1_pref = np.sqrt(0.5 * gamma_xx_solver)
        c2_pref = np.sqrt(0.5 * gamma_xx_solver)
        c3_pref = np.sqrt(0.5 * gamma_x_solver)
        c4_pref = np.sqrt(0.5 * gamma_x_solver)

        # Jump operators:
        # L1 = sqrt(gamma_xx/2) * (|X1><XX| kron adag_m1)
        # L2 = sqrt(gamma_xx/2) * (|X2><XX| kron adag_m1)
        # L3 = sqrt(gamma_x/2)  * (|G><X1|  kron adag_m2)
        # L4 = sqrt(gamma_x/2)  * (|G><X2|  kron adag_m2)

        def embed_two(idx_a: int, sym_a: str, idx_b: int, sym_b: str) -> OpExpr:
            emb = EmbeddedKron(
                indices=(idx_a, idx_b),
                locals=(LocalSymbolOp(sym_a), LocalSymbolOp(sym_b)),
            )
            return OpExpr.atom(emb)

        qd = modes.index_of("qd")
        m1 = modes.index_of("m1")
        m2 = modes.index_of("m2")

        L1 = Term(
            kind=TermKind.C,
            op=OpExpr.scale(complex(c1_pref), embed_two(
                qd, "t_X1_XX", m1, "adag")),
            coeff=None,
            label="L_XX_to_X1_emit_m1",
        )
        L2 = Term(
            kind=TermKind.C,
            op=OpExpr.scale(complex(c2_pref), embed_two(
                qd, "t_X2_XX", m1, "adag")),
            coeff=None,
            label="L_XX_to_X2_emit_m1",
        )
        L3 = Term(
            kind=TermKind.C,
            op=OpExpr.scale(complex(c3_pref), embed_two(
                qd, "t_G_X1", m2, "adag")),
            coeff=None,
            label="L_X1_to_G_emit_m2",
        )
        L4 = Term(
            kind=TermKind.C,
            op=OpExpr.scale(complex(c4_pref), embed_two(
                qd, "t_G_X2", m2, "adag")),
            coeff=None,
            label="L_X2_to_G_emit_m2",
        )

        col = FrozenCatalog((L1, L2, L3, L4))
        return CompileBundle(modes=modes, hamiltonian=ham, collapse=col)

    def materialize_bundle(self) -> MaterializeBundle:
        return MaterializeBundle(ops=CascadeMaterializer())


def _ket(dim: int, i: int) -> np.ndarray:
    v = np.zeros((dim,), dtype=complex)
    v[int(i)] = 1.0 + 0.0j
    return v


def main() -> None:
    # Choose solver time unit: 1 solver unit = 1 ps
    time_unit_s = float(Q(1.0, "ps").to("s").magnitude)

    # Physical rates
    gamma_xx = Q(5.0e9, "1/s")  # XX lifetime ~20 ps
    gamma_x = Q(5.0e9, "1/s")  # X lifetime ~20 ps

    model = FourLevelCascadeModel(
        gamma_xx=gamma_xx,
        gamma_x=gamma_x,
        sigma_s=Q(5, "ps"),  # Q(10.0, "ps"),
        omega_drive_rad_s=Q(5e10, "rad/s"),
        t0_s=Q(50, "ps"),
    )

    # Sim time: 0..200 ps in solver units
    tlist = np.linspace(0.0, 5000.0, 5001)

    # Initial state: |XX> in the dot, both modes vacuum |0> |0>
    # dims: (4,2,2), basis order: (qd, m1, m2)
    ket_xx = _ket(4, 3)  # XX index = 3
    ket_g = _ket(4, 0)
    ket_0 = _ket(2, 0)
    psi0 = np.kron(np.kron(ket_g, ket_0), ket_0)
    rho0 = np.outer(psi0, np.conjugate(psi0))

    engine = SimulationEngine(audit=True)

    problem = engine.compile(
        model, tlist=tlist, time_unit_s=time_unit_s, rho0=rho0)

    H0 = problem.h_terms[0]

    idx_G00 = ((0 * 2) + 0) * 2 + 0
    idx_XX00 = ((3 * 2) + 0) * 2 + 0

    print("H op[G00,XX00] =", H0.op[idx_G00, idx_XX00])
    print("H op[XX00,G00] =", H0.op[idx_XX00, idx_G00])

    from smef.core.ir.coeffs import eval_coeff_any

    c = eval_coeff_any(H0.coeff, problem.tlist,
                       time_unit_s=problem.time_unit_s)
    print("max|coeff| =", float(np.max(np.abs(c))))
    print("area =", complex(np.trapezoid(c, problem.tlist)))

    solve_options = {
        "method": "bdf",  # stiff-safe; "adams" can be fine but bdf is more robust here
        "atol": 1e-10,
        "rtol": 1e-8,
        "nsteps": 200000,
        "max_step": 0.02,  # solver units; for sigma=3ps and time_unit=1ps this is conservative
        "progress_bar": "tqdm",
    }
    res = engine.run(
        model,
        tlist=tlist,
        time_unit_s=time_unit_s,
        rho0=rho0,
        solve_options={"qutip_options": solve_options},
    )
    probe_ps = [0.0, 40.0, 50.0, 60.0, 100.0, 200.0]
    idxs = [int(np.argmin(np.abs(tlist - t))) for t in probe_ps]

    # Build full-space projectors / number operators by explicit kron
    proj_G = np.diag([1.0, 0.0, 0.0, 0.0]).astype(complex)
    proj_X1 = np.diag([0.0, 1.0, 0.0, 0.0]).astype(complex)
    proj_X2 = np.diag([0.0, 0.0, 1.0, 0.0]).astype(complex)
    proj_XX = np.diag([0.0, 0.0, 0.0, 1.0]).astype(complex)

    n = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    # Full operators on (qd, m1, m2)
    def full_qd(op4: np.ndarray) -> np.ndarray:
        return np.kron(np.kron(op4, I2), I2)

    def full_m1(op2: np.ndarray) -> np.ndarray:
        return np.kron(np.kron(np.eye(4, dtype=complex), op2), I2)

    def full_m2(op2: np.ndarray) -> np.ndarray:
        return np.kron(np.kron(np.eye(4, dtype=complex), I2), op2)

    P_G = full_qd(proj_G)
    P_X1 = full_qd(proj_X1)
    P_X2 = full_qd(proj_X2)
    P_XX = full_qd(proj_XX)

    N1 = full_m1(n)
    N2 = full_m2(n)

    # Extract expectations
    pop_G = np.empty(len(res.tlist), dtype=float)
    pop_X1 = np.empty(len(res.tlist), dtype=float)
    pop_X2 = np.empty(len(res.tlist), dtype=float)
    pop_XX = np.empty(len(res.tlist), dtype=float)
    n1 = np.empty(len(res.tlist), dtype=float)
    n2 = np.empty(len(res.tlist), dtype=float)

    for i, st in enumerate(res.states):
        if hasattr(st, "full"):
            rho = np.asarray(st.full(), dtype=complex)
        else:
            rho = np.asarray(st, dtype=complex)

        pop_G[i] = float(np.real(np.trace(P_G @ rho)))
        pop_X1[i] = float(np.real(np.trace(P_X1 @ rho)))
        pop_X2[i] = float(np.real(np.trace(P_X2 @ rho)))
        pop_XX[i] = float(np.real(np.trace(P_XX @ rho)))
        n1[i] = float(np.real(np.trace(N1 @ rho)))
        n2[i] = float(np.real(np.trace(N2 @ rho)))

    # Plot vs physical time in ps
    t_ps = (np.asarray(res.tlist, dtype=float) * time_unit_s) / float(
        Q(1.0, "ps").to("s").magnitude
    )

    t_s = res.tlist * time_unit_s  # seconds if time_unit_s is in seconds
    t_ps = t_s * 1e12

    # Evaluate pulse on the solver grid for plotting

    Omega = model.omega_pulse_solver(res.tlist, time_unit_s=time_unit_s)
    Omega_abs = np.abs(Omega)

    fig, axes = plt.subplots(3, 1, sharex=True)

    axes[0].plot(t_ps, Omega_abs)
    axes[0].set_ylabel("|Omega(t)| (solver)")
    axes[0].set_title("Driven 4-level cascade (SMEF)")

    axes[1].plot(t_ps, pop_XX, label="P_XX")
    axes[1].plot(t_ps, pop_X1, label="P_X1")
    axes[1].plot(t_ps, pop_X2, label="P_X2")
    axes[1].plot(t_ps, pop_G, label="P_G")

    axes[1].set_ylabel("population")
    axes[1].legend()

    axes[2].plot(t_ps, n1, label="n_m1 (first photon)")
    axes[2].plot(t_ps, n2, label="n_m2 (second photon)")
    axes[2].set_xlabel("t (ps)")
    axes[2].set_ylabel("expected photon number")
    axes[2].legend()

    for ax in axes:
        ax.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
