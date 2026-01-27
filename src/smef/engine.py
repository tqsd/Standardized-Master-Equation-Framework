from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from smef.core.drives.types import DriveSpec
from smef.core.model.merge import merge_bundle_with_drive_terms
from smef.core.model.protocols import CompilableModelProto
from smef.core.sim.compile import compile_to_dense
from smef.core.sim.protocols import SolverAdapterProto
from smef.core.sim.types import MEProblemDense, MESolveResult
from smef.core.sim.audit import audit_problem_dense, AuditOptions


def _default_adapter() -> SolverAdapterProto:
    # Late import to avoid core depending on QuTiP
    from smef.adapters.qutip.adapter import QuTiPAdapter  # adjust path to your adapter

    return QuTiPAdapter()


@dataclass
class SimulationEngine:
    adapter: Optional[SolverAdapterProto] = None
    audit: bool = False
    audit_options: Optional[AuditOptions] = None

    def compile(
        self,
        model: CompilableModelProto,
        *,
        tlist: np.ndarray,
        time_unit_s: float,
        rho0: Optional[np.ndarray] = None,
        drives: Optional[Sequence[DriveSpec]] = None,
    ) -> MEProblemDense:
        bundle = model.compile_bundle()
        material = model.materialize_bundle()
        if drives:
            decoder = getattr(bundle, "drive_decoder", None)
            strength = getattr(bundle, "drive_strength", None)
            decode_ctx = getattr(bundle, "drive_decode_ctx", None)

            if decoder is None or strength is None:
                raise ValueError(
                    "Drives provided but model did not provide drive_decoder and drive_strength"
                )

            resolved = decoder.decode(drives, ctx=decode_ctx)
            coeffs = strength.compute(
                resolved,
                np.asarray(tlist, dtype=float),
                time_unit_s=float(time_unit_s),
                decode_ctx=decode_ctx,
            )

            emitter = getattr(bundle, "drive_emitter", None)
            if emitter is None:
                raise ValueError(
                    "Model supports drive decoding/strength but did not provide drive_emitter"
                )

            drive_terms = emitter.emit_drive_terms(
                resolved, coeffs, decode_ctx=decode_ctx
            )

            bundle = merge_bundle_with_drive_terms(bundle, drive_terms)

        problem = compile_to_dense(
            bundle=bundle,
            material=material,
            tlist=np.asarray(tlist, dtype=float),
            time_unit_s=float(time_unit_s),
            rho0=rho0,
        )
        return problem

    def run(
        self,
        model: CompilableModelProto,
        *,
        tlist: np.ndarray,
        time_unit_s: float,
        rho0: Optional[np.ndarray] = None,
        solve_options: Optional[Mapping[str, Any]] = None,
    ) -> MESolveResult:
        problem = self.compile(
            model,
            tlist=tlist,
            time_unit_s=time_unit_s,
            rho0=rho0,
        )

        if self.audit:
            _ = audit_problem_dense(problem, options=self.audit_options)

        adapter = self.adapter or _default_adapter()
        return adapter.solve(problem, options=solve_options)
