from __future__ import annotations

from typing import Any

import numpy as np

from smef.core.sim.types import CompiledTermDense


def eval_coeff(
    term: CompiledTermDense, tlist: np.ndarray, *, time_unit_s: float
) -> np.ndarray:
    c = term.coeff
    if c is None:
        return np.ones(len(tlist), dtype=complex)
    if hasattr(c, "eval"):
        try:
            return c.eval(tlist, time_unit_s)  # CallableCoeffUnits
        except TypeError:
            return c.eval(tlist)  # CallableCoeff
    raise TypeError("Unsupported coeff type")


def effective_op_at(
    term: CompiledTermDense,
    tlist: np.ndarray,
    index: int,
    *,
    time_unit_s: float,
) -> np.ndarray:
    """
    Effective operator at a specific solver time index.

    O_eff(t_i) = coeff(t_i) * op
    """
    coeff = eval_coeff(term, tlist, time_unit_s=time_unit_s)[int(index)]
    return complex(coeff) * np.asarray(term.op, dtype=complex)
