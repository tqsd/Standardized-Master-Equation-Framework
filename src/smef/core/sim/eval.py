from __future__ import annotations

from typing import Any, Callable

import numpy as np

from smef.core.sim.types import CompiledTermDense


def eval_coeff(
    term: CompiledTermDense, tlist: np.ndarray, *, time_unit_s: float
) -> np.ndarray:
    c = term.coeff
    if c is None:
        out = np.empty(len(tlist), dtype=complex)
        out[:] = 1.0 + 0.0j
        return out

    if hasattr(c, "eval"):
        try:
            y = c.eval(tlist, float(time_unit_s))  # CallableCoeffUnits-like
        except TypeError:
            y = c.eval(tlist)  # CallableCoeff-like
        return np.asarray(y, dtype=complex).reshape(len(tlist))

    if callable(c):
        y = c(tlist)
        return np.asarray(y, dtype=complex).reshape(len(tlist))

    raise TypeError(f"Unsupported coeff type: {type(c)!r}")


def effective_op_at(
    term: CompiledTermDense, tlist: np.ndarray, index: int, *, time_unit_s: float
) -> np.ndarray:
    coeff_i = eval_coeff(term, tlist, time_unit_s=time_unit_s)[int(index)]
    return complex(coeff_i) * np.asarray(term.op, dtype=complex)
