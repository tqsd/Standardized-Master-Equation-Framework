from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from smef.core.drives.types import DriveSpec
from smef.core.model.merge import merge_bundle_with_drive_terms
from smef.core.model.protocols import CompilableModelProto
from smef.core.sim.audit import AuditOptions, audit_problem_dense
from smef.core.sim.compile import compile_to_dense
from smef.core.sim.protocols import SolverAdapterProto
from smef.core.sim.types import MEProblemDense, MESolveResult
from smef.core.units import UnitSystem


def _default_adapter() -> SolverAdapterProto:
    from smef.adapters.qutip.adapter import QuTiPAdapter

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
        units = UnitSystem(time_unit_s=time_unit_s)
        bundle = model.compile_bundle(units=units)
        material = model.materialize_bundle()

        tlist_f = np.asarray(tlist, dtype=float)
        time_unit_s_f = float(time_unit_s)

        if drives:
            decoder = getattr(bundle, "drive_decoder", None)
            strength = getattr(bundle, "drive_strength", None)
            decode_ctx = getattr(bundle, "drive_decode", None)
            emitter = getattr(bundle, "drive_emitter", None)

            if decode_ctx is not None and hasattr(decode_ctx, "with_solver_grid"):
                decode_ctx = decode_ctx.with_solver_grid(
                    tlist=tlist_f, time_unit_s=time_unit_s_f
                )

            if decoder is None or strength is None or emitter is None:
                raise ValueError(
                    "Drives provided but model must provide drive_decoder, drive_strength, and drive_emitter"
                )

            resolved = decoder.decode(drives, ctx=decode_ctx)

            coeffs = strength.compute(
                resolved,
                tlist_f,
                time_unit_s=time_unit_s_f,
                decode_ctx=decode_ctx,
            )

            drive_terms = emitter.emit_drive_terms(
                resolved,
                coeffs,
                decode_ctx=decode_ctx,
            )

            bundle = merge_bundle_with_drive_terms(bundle, drive_terms)

        return compile_to_dense(
            bundle=bundle,
            material=material,
            tlist=tlist_f,
            time_unit_s=time_unit_s_f,
            rho0=rho0,
        )

    def run(
        self,
        model: CompilableModelProto,
        *,
        tlist: np.ndarray,
        time_unit_s: float,
        rho0: Optional[np.ndarray] = None,
        drives: Optional[Sequence[DriveSpec]] = None,
        solve_options: Optional[Mapping[str, object]] = None,
    ) -> MESolveResult:
        problem = self.compile(
            model,
            tlist=tlist,
            time_unit_s=time_unit_s,
            rho0=rho0,
            drives=drives,
        )

        if self.audit:
            print(audit_problem_dense(problem, options=self.audit_options))

        adapter = self.adapter or _default_adapter()
        return adapter.solve(problem, options=solve_options)
