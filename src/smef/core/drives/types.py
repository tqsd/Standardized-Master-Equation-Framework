from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Mapping, Optional, Sequence, Tuple

import numpy as np

from smef.core.ir.terms import Term

TransitionKey = Hashable
DriveId = Hashable


@dataclass(frozen=True)
class DriveSpec:
    """
    Opaque user drive specification.

    SMEF does not interpret this. Models may accept arbitrary user objects here,
    but having a minimal wrapper is convenient for pipeline plumbing.
    """

    payload: Any
    drive_id: Optional[DriveId] = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedDrive:
    """
    Model-decoded drive that SMEF can route to operator terms.

    - transition_key: "what it couples to" (model-defined)
    - carrier_omega: optional angular carrier frequency in rad/s (physical units)
    - envelope: optional envelope samples in physical units (not required; model can store payload instead)
    - meta: anything else needed by the strength model (pulse width, area, detuning, polarization, etc.)
    """

    drive_id: DriveId
    transition_key: TransitionKey

    carrier_omega_rad_s: Optional[float] = None

    t_s: Optional[np.ndarray] = None
    envelope: Optional[np.ndarray] = None  # complex envelope in physical units

    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DriveCoefficients:
    """
    Output of a drive strength model.

    For each resolved drive, produce a complex coefficient array on solver tlist.

    coeffs maps (drive_id, transition_key) -> complex array (N,)
    """

    tlist: np.ndarray
    coeffs: Mapping[Tuple[DriveId, TransitionKey], np.ndarray] = field(
        default_factory=dict
    )
    meta: Mapping[str, Any] = field(default_factory=dict)

    def get(self, drive_id: DriveId, transition_key: TransitionKey) -> np.ndarray:
        return self.coeffs[(drive_id, transition_key)]


@dataclass(frozen=True)
class DriveTermBundle:
    h_terms: Sequence[Term] = ()
    c_terms: Sequence[Term] = ()
    e_terms: Sequence[Term] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)
