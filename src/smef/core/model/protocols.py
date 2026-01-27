from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Hashable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
    TYPE_CHECKING,
)

from smef.core.drives.protocols import DriveDecoderProto, DriveTermEmitterProto
from smef.core.ir.protocols import OpMaterializeContextProto
import numpy as np

if TYPE_CHECKING:
    from smef.core.types import DriveCoefficients, ResolvedDrive

    # Term will live in smef.core.ir.terms later
    from smef.core.ir.terms import Term


@runtime_checkable
class ModeRegistryProto(Protocol):
    """
    Compiler-facing mode registry.

    The compiler uses this to build dims and to give consistent ordering.
    Keys are intentionally generic (Hashable) so models can use enums,
    strings, tuples, etc.
    """

    def dims(self) -> Sequence[int]:
        """
        Per-mode local dimensions in compiler ordering.
        Example: [4, 2, 2] for (system, pol, fock).
        """
        ...

    def index_of(self, key: Hashable) -> int:
        """
        Map a model-defined key to a compiler mode index.
        """
        ...

    @property
    def channels(self) -> Optional[Sequence[Hashable]]:
        """
        Optional: ordered keys for reporting/debug. If provided, should align
        with dims()/index ordering.
        """
        ...


@runtime_checkable
class TermCatalogProto(Protocol):
    """
    Drive-agnostic list of IR terms; metadata is embedded in Term itself.
    """

    @property
    def all_terms(self) -> Sequence["Term"]: ...


@runtime_checkable
class DriveDecodeContextProto(Protocol):
    """
    Passive adapter bundle used by the pipeline to interpret user drives.
    """

    # Keep empty for now; later add minimal required properties, e.g.
    # transition_space, polarization_space, phonon_space
    pass


@runtime_checkable
class DriveStrengthModelProto(Protocol):
    """
    Converts resolved drives into time-series coefficients in solver units.
    """

    def compute_drive_coeffs(
        self,
        resolved: Sequence["ResolvedDrive"],
        tlist: np.ndarray,
        *,
        time_unit_s: float,
        decode_ctx: Optional[DriveDecodeContextProto] = None,
    ) -> "DriveCoefficients": ...


@dataclass(frozen=True)
class CompileBundle:
    # required
    modes: ModeRegistryProto
    hamiltonian: TermCatalogProto

    # optional (future-proof)
    collapse: Optional[TermCatalogProto] = None
    observables: Optional[TermCatalogProto] = None

    # drives (all optional)
    # decode context: passive adapters/config
    drive_decode: Optional[DriveDecodeContextProto] = None

    # drive decoder: specs -> resolved drives
    drive_decoder: Optional[DriveDecoderProto] = None

    # strength model: resolved -> coefficients on tlist
    drive_strength: Optional[DriveStrengthModelProto] = None

    # emitter: resolved + coeffs -> extra terms to merge into catalogs
    drive_emitter: Optional[DriveTermEmitterProto] = None

    meta: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class MaterializeBundle:
    ops: OpMaterializeContextProto
    meta: Optional[Mapping[str, Any]] = None


@runtime_checkable
class CompilableModelProto(Protocol):
    def compile_bundle(self) -> CompileBundle:
        """Lazy. Should be cheap to construct."""
        ...

    def materialize_bundle(self) -> MaterializeBundle:
        """Lazy. May be heavy/cached/backend-specific."""
        ...
