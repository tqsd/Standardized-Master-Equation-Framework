from __future__ import annotations

from typing import Tuple

import numpy as np

from smef.core.ir.materialize import materialize_op_expr
from smef.core.model.protocols import CompileBundle, MaterializeBundle
from smef.core.sim.types import CompiledTermDense, MEProblemDense


def compile_to_dense(
    *,
    bundle: CompileBundle,
    material: MaterializeBundle,
    tlist: np.ndarray,
    time_unit_s: float,
    rho0: np.ndarray | None = None,
) -> MEProblemDense:
    dims = tuple(int(x) for x in bundle.modes.dims())

    def _compile_terms(terms) -> Tuple[CompiledTermDense, ...]:
        out = []
        for term in terms:
            op = materialize_op_expr(term.op, dims=dims, ctx=material.ops)
            out.append(
                CompiledTermDense(
                    op=op,
                    coeff=term.coeff,
                    label=getattr(term, "label", ""),
                    meta=getattr(term, "meta", {}),
                )
            )
        return tuple(out)

    h_terms = _compile_terms(bundle.hamiltonian.all_terms)
    c_terms = _compile_terms(
        bundle.collapse.all_terms) if bundle.collapse else ()
    e_terms = _compile_terms(
        bundle.observables.all_terms) if bundle.observables else ()

    return MEProblemDense(
        dims=dims,
        tlist=np.asarray(tlist, dtype=float),
        time_unit_s=float(time_unit_s),
        h_terms=h_terms,
        c_terms=c_terms,
        e_terms=e_terms,
        rho0=rho0,
        meta=bundle.meta or {},
    )
