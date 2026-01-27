from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class SymbolOp:
    """
    Symbolic operator reference for a full-space operator.

    The materializer is responsible for resolving the symbol into a (D, D) matrix.
    """

    symbol: str


@dataclass(frozen=True)
class LocalSymbolOp:
    """
    Symbolic operator reference for a local operator on a single subsystem.
    """

    symbol: str


@dataclass(frozen=True)
class EmbeddedKron:
    """
    Place local operators on specific subsystem indices.

    indices are mode indices in compiler ordering.
    locals are LocalSymbolOp.
    """

    indices: Tuple[int, ...]
    locals: Tuple[LocalSymbolOp, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.locals):
            raise ValueError(
                "EmbeddedKron indices and locals must have same length")


class OpExprKind(str, Enum):
    ATOM = "ATOM"
    SCALE = "SCALE"
    SUM = "SUM"
    PROD = "PROD"


OpAtom = Union[SymbolOp, EmbeddedKron]


@dataclass(frozen=True)
class OpExpr:
    """
    Backend-agnostic operator expression tree.

    - ATOM: atom must be set
    - SCALE: scalar and args[0] must be set
    - SUM: args must be non-empty
    - PROD: args must be non-empty
    """

    kind: OpExprKind
    atom: Optional[OpAtom] = None
    scalar: Optional[complex] = None
    args: Tuple["OpExpr", ...] = ()

    @staticmethod
    def atom(x: OpAtom) -> "OpExpr":
        return OpExpr(kind=OpExprKind.ATOM, atom=x)

    @staticmethod
    def scale(s: complex, x: "OpExpr") -> "OpExpr":
        return OpExpr(kind=OpExprKind.SCALE, scalar=s, args=(x,))

    @staticmethod
    def summation(xs: Sequence["OpExpr"]) -> "OpExpr":
        xs_t = tuple(xs)
        if not xs_t:
            raise ValueError("SUM requires at least one argument")
        return OpExpr(kind=OpExprKind.SUM, args=xs_t)

    @staticmethod
    def product(xs: Sequence["OpExpr"]) -> "OpExpr":
        xs_t = tuple(xs)
        if not xs_t:
            raise ValueError("PROD requires at least one argument")
        return OpExpr(kind=OpExprKind.PROD, args=xs_t)
