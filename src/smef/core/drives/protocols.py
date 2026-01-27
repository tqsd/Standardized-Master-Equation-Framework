from __future__ import annotations
from typing import Any, Optional, Protocol, Sequence, runtime_checkable
import numpy as np
from smef.core.drives.types import (
    DriveCoefficients,
    DriveSpec,
    DriveTermBundle,
    ResolvedDrive,
)


@runtime_checkable
class DriveTermEmitterProto(Protocol):
    def emit_drive_terms(
        self,
        resolved: Sequence[ResolvedDrive],
        coeffs: DriveCoefficients,
        *,
        decode_ctx: Optional[DriveDecodeContextProto] = None,
    ) -> DriveTermBundle: ...


@runtime_checkable
class DriveDecodeContextProto(Protocol):
    """
    Passive adapter bundle provided by the model.

    The stage code won't assume anything about its internals; it is passed
    through to the decoder/strength model.
    """

    ...


@runtime_checkable
class DriveDecoderProto(Protocol):
    """
    Convert user drive specs to ResolvedDrive objects.
    """

    def decode(
        self,
        specs: Sequence[DriveSpec],
        *,
        ctx: Optional[DriveDecodeContextProto] = None,
    ) -> Sequence[ResolvedDrive]: ...


@runtime_checkable
class DriveStrengthModelProto(Protocol):
    """
    Convert resolved drives into solver-grid coefficients.
    """

    def compute(
        self,
        resolved: Sequence[ResolvedDrive],
        tlist: np.ndarray,
        *,
        time_unit_s: float,
        decode_ctx: Optional[DriveDecodeContextProto] = None,
    ) -> DriveCoefficients: ...
