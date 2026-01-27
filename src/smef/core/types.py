from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, TypeAlias

import numpy as np

# Generic keys used across models
ModeKey: TypeAlias = Hashable
TransitionKey: TypeAlias = Hashable
DriveId: TypeAlias = str


@dataclass(frozen=True)
class ResolvedDrive:
    """
    Unitless drive description after decoding.

    This is the output of the drive-decode stage (or provided directly by a model).
    All quantities here must already be in solver units, except for purely symbolic
    fields like transition_key.

    Typical usage:
    - transition_key points to a model-defined transition/operator family
    - amp is a unitless scalar multiplier (complex allowed)
    - envelope is a coeff-like object (callable or ConstCoeff etc.) that returns
      complex values over tlist in solver units
    """

    drive_id: DriveId
    transition_key: TransitionKey

    # Unitless amplitude multiplier (often encodes polarization overlap, dipole scaling, etc.)
    amp: complex = 1.0 + 0.0j

    # Envelope in solver units; expected to have .eval(tlist) -> complex array
    envelope: Any = None

    # Optional metadata for reporting/debugging
    label: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DriveCoefficients:
    """
    Time-series coefficients in solver units.

    The pipeline typically produces one coefficient trace per drive_id.
    Values are complex arrays of shape (len(tlist),).
    """

    tlist: np.ndarray
    coeffs: Dict[DriveId, np.ndarray]
    meta: Mapping[str, Any] = field(default_factory=dict)

    def get(self, drive_id: DriveId) -> np.ndarray:
        return self.coeffs[drive_id]


@dataclass(frozen=True)
class CompileContext:
    """
    Common compile-time context shared across stages.

    tlist is in solver units (unitless). time_unit_s defines the mapping to SI.
    unit_system is optional so SMEF core does not force a particular unit module,
    but you will likely pass smef.units.UnitSystem here in your app.
    """

    tlist: np.ndarray
    time_unit_s: float
    unit_system: Optional[Any] = None

    def tlist_float(self) -> np.ndarray:
        return np.asarray(self.tlist, dtype=float)
