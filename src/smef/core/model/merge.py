from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

from smef.core.drives.types import DriveTermBundle
from smef.core.ir.terms import Term
from smef.core.model.protocols import CompileBundle, TermCatalogProto


@dataclass(frozen=True)
class FrozenTermCatalog(TermCatalogProto):
    """
    Small immutable catalog used for merged views.

    This is intentionally minimal: it only satisfies TermCatalogProto by
    exposing .all_terms.
    """

    _terms: Tuple[Term, ...]

    @property
    def all_terms(self) -> Sequence[Term]:
        return self._terms


def merge_bundle_with_drive_terms(
    bundle: CompileBundle, drive_terms: DriveTermBundle
) -> CompileBundle:
    """
    Return a new CompileBundle where drive-emitted terms are appended to the
    existing catalogs.

    Rules:
    - Hamiltonian terms are required in the base bundle; drive h_terms are appended.
    - Collapse / observables are optional in the base; if drive provides terms,
      missing catalogs are created.
    - Bundle's drive-related fields (decoder, ctx, strength, emitter) are preserved.
    - meta dicts are shallow-merged (drive meta overrides base on key collisions).
    """

    h_cat = _merge_catalog(bundle.hamiltonian, drive_terms.h_terms)

    c_cat = _merge_optional_catalog(bundle.collapse, drive_terms.c_terms)
    e_cat = _merge_optional_catalog(bundle.observables, drive_terms.e_terms)

    meta = _merge_meta(bundle.meta, drive_terms.meta)

    kwargs: dict[str, Any] = {
        "modes": bundle.modes,
        "hamiltonian": h_cat,
        "collapse": c_cat,
        "observables": e_cat,
        "meta": meta,
    }

    for name in (
        "drive_decode",
        "drive_strength",
        "drive_decoder",
        "drive_decode_ctx",
        "drive_emitter",
    ):
        if hasattr(bundle, name):
            kwargs[name] = getattr(bundle, name)

    return CompileBundle(**kwargs)


def _merge_catalog(
    base: TermCatalogProto, extra_terms: Sequence[Term]
) -> TermCatalogProto:
    if not extra_terms:
        return base
    merged = tuple(base.all_terms) + tuple(extra_terms)
    return FrozenTermCatalog(merged)


def _merge_optional_catalog(
    base: Optional[TermCatalogProto],
    extra_terms: Sequence[Term],
) -> Optional[TermCatalogProto]:
    if not extra_terms:
        return base
    if base is None:
        return FrozenTermCatalog(tuple(extra_terms))
    return FrozenTermCatalog(tuple(base.all_terms) + tuple(extra_terms))


def _merge_meta(
    base: Optional[Mapping[str, Any]],
    extra: Mapping[str, Any],
) -> Optional[dict]:
    if (not base) and (not extra):
        return None
    out: dict[str, Any] = {}
    if base:
        out.update(dict(base))
    if extra:
        out.update(dict(extra))
    return out
