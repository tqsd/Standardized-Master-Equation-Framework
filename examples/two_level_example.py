from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from smef.core import ConstCoeff
from smef.core.ir.ops import OpExpr, LocalSymbolOp, SymbolOp
from smef.core.ir.terms import Term, TermKind
from smef.core.ir.protocols import OpMaterializeContextProto
from smef.core.model.protocols import (
    CompileBundle,
    MaterializeBundle,
    ModeRegistryProto,
    TermCatalogProto,
)
from smef.engine import SimulationEngine


@dataclass(frozen=True)
class SimpleModeRegistry(ModeRegistryProto):
    _dims: Tuple[int, ...] = (2,)
    _channels: Tuple[str, ...] = ("tls",)

    def dims(self) -> Sequence[int]:
        return self._dims

    def index_of(self, key: Any) -> int:
        if key == "tls" or key == 0:
            return 0
        raise KeyError(f"Unknown mode key: {key}")

    @property
    def channels(self) -> Optional[Sequence[Any]]:
        return self._channels


@dataclass(frozen=True)
class SimpleCatalog(TermCatalogProto):
    _terms: Tuple[Term, ...]

    @property
    def all_terms(self) -> Sequence[Term]:
        return self._terms


@dataclass
class TLSMaterializer(OpMaterializeContextProto):
    """
    Resolves a few standard TLS operator symbols.

    Convention used by SMEF materializer:
    - full-space symbols: resolve_symbol(symbol, dims_full)
    - local symbols: resolve_symbol(symbol, (dim,))
    """

    def resolve_symbol(self, symbol: str, dims: Sequence[int]) -> np.ndarray:
        dims_t = tuple(int(x) for x in dims)

        # local op request: dims == (2,)
        if dims_t == (2,):
            return self._resolve_local_2(symbol)

        # full space request for this toy model is also just (2,)
        if dims_t == (2,):
            return self._resolve_local_2(symbol)

        # If later you add more modes, expand this.
        raise ValueError(f"Unsupported dims for TLSMaterializer: {dims_t}")

    def resolve_embedded(self, embedded: Any, dims: Sequence[int]) -> np.ndarray:
        raise NotImplementedError(
            "TLSMaterializer does not implement resolve_embedded")

    def _resolve_local_2(self, symbol: str) -> np.ndarray:
        if symbol == "id":
            return np.eye(2, dtype=complex)

        if symbol == "pauli_z":
            return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

        if symbol == "sigma_plus":
            return np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)

        if symbol == "sigma_minus":
            return np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)

        if symbol == "proj_g":
            return np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)

        if symbol == "proj_e":
            return np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)

        raise KeyError(f"Unknown operator symbol: {symbol}")


@dataclass
class TwoLevelModel:
    """
    Minimal compilable model implementing SMEF bundle interface.

    H = 0.5 * w * sigma_z
    c = sqrt(gamma) * sigma_minus
    e = proj_e
    """

    w: float  # angular frequency in solver units
    gamma: float  # decay rate in solver units

    def compile_bundle(self) -> CompileBundle:
        modes = SimpleModeRegistry()

        # Hamiltonian term
        H_op = OpExpr.atom(SymbolOp("pauli_z"))
        H_term = Term(
            kind=TermKind.H,
            op=H_op,
            coeff=ConstCoeff(0.5 * float(self.w)),
            label="H_tls",
            meta={"w": float(self.w)},
        )

        # Collapse term: coeff multiplies operator directly
        c_op = OpExpr.atom(SymbolOp("sigma_minus"))
        c_term = Term(
            kind=TermKind.C,
            op=c_op,
            coeff=ConstCoeff(np.sqrt(float(self.gamma))),
            label="c_decay",
            meta={"gamma": float(self.gamma)},
        )

        # Observable: excited state population
        e_op = OpExpr.atom(SymbolOp("proj_e"))
        e_term = Term(
            kind=TermKind.E,
            op=e_op,
            coeff=None,
            label="P_e",
        )

        ham = SimpleCatalog((H_term,))
        col = SimpleCatalog((c_term,))
        obs = SimpleCatalog((e_term,))

        return CompileBundle(
            modes=modes,
            hamiltonian=ham,
            collapse=col,
            observables=obs,
            meta={"name": "TwoLevelModel"},
        )

    def materialize_bundle(self) -> MaterializeBundle:
        return MaterializeBundle(ops=TLSMaterializer())


def main() -> None:
    # Solver grid (unitless solver time)
    tlist = np.linspace(0.0, 50.0, 1201, dtype=float)

    # Choose solver-unit parameters
    w = 1.0
    gamma = 0.2

    model = TwoLevelModel(w=w, gamma=gamma)

    # Initial state: excited state |e><e|
    rho0 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)

    engine = SimulationEngine(audit=True)

    # time_unit_s is only metadata for now in this toy example
    res = engine.run(
        model,
        tlist=tlist,
        time_unit_s=1.0,
        rho0=rho0,
        solve_options={"progress_bar": "tqdm"},
    )

    pe = res.expect.get("P_e")
    if pe is None:
        raise RuntimeError("Missing observable P_e")

    print("P_e(t0) =", float(np.real(pe[0])))
    print("P_e(tend) =", float(np.real(pe[-1])))

    plt.figure()
    plt.plot(tlist, np.real(pe))
    plt.xlabel("t (solver units)")
    plt.ylabel("P_e")
    plt.title("Two-level decay")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
