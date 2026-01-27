from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

import numpy as np


class TermKind(str, Enum):
    H = "H"  # Hamiltonian
    C = "C"  # Collapse
    E = "E"  # Expectation / observable


@dataclass(frozen=True)
class Term:
    """
    Unitless IR term.

    Interpretation:
    - op is a backend-agnostic operator reference (symbolic or embedded)
    - coeff is either None (static 1.0) or a callable/expr evaluated on solver time grid
    - kind indicates which bucket the pipeline uses it in

    All numeric values are already in solver units.
    """

    kind: TermKind
    op: Any
    coeff: Optional[Any] = None

    label: str = ""
    tags: Sequence[str] = field(default_factory=tuple)
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DenseOp:
    """
    Escape hatch: already-materialized dense operator (unitless).
    Prefer symbolic refs in real models, but this is useful for tests.
    """

    mat: np.ndarray
