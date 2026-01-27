from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from smef.core.ir.terms import Term


@dataclass(frozen=True)
class MEProblemIR:
    dims: Tuple[int, ...]
    tlist: np.ndarray
    time_unit_s: float

    h_terms: Tuple[Term, ...] = ()
    c_terms: Tuple[Term, ...] = ()
    e_terms: Tuple[Term, ...] = ()

    rho0: Optional[np.ndarray] = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def D(self) -> int:
        d = 1
        for x in self.dims:
            d *= int(x)
        return d


@dataclass(frozen=True)
class CompiledTermDense:
    """
    Dense compiled term.

    op: (D, D) complex
    coeff: optional coeff object (eval on tlist) OR None means constant 1
    """

    op: np.ndarray
    coeff: Optional[Any] = None
    label: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MEProblemDense:
    dims: Tuple[int, ...]
    tlist: np.ndarray
    time_unit_s: float

    h_terms: Tuple[CompiledTermDense, ...] = ()
    c_terms: Tuple[CompiledTermDense, ...] = ()
    e_terms: Tuple[CompiledTermDense, ...] = ()

    rho0: Optional[np.ndarray] = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def D(self) -> int:
        d = 1
        for x in self.dims:
            d *= int(x)
        return d


@dataclass(frozen=True)
class MESolveResult:
    tlist: np.ndarray
    states: Optional[Any] = None
    expect: Mapping[str, np.ndarray] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)
