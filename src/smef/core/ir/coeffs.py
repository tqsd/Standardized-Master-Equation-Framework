from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class CoeffProto(Protocol):
    def eval(self, tlist: np.ndarray) -> np.ndarray:
        """
        Returns complex array of shape (len(tlist),).
        All values are in solver units.
        """
        ...


@dataclass(frozen=True)
class ConstCoeff:
    value: complex

    def eval(self, tlist: np.ndarray) -> np.ndarray:
        out = np.empty(len(tlist), dtype=complex)
        out[:] = self.value
        return out


@dataclass(frozen=True)
class CallableCoeff:
    fn: Callable[[np.ndarray], np.ndarray]

    def eval(self, tlist: np.ndarray) -> np.ndarray:
        y = self.fn(tlist)
        y = np.asarray(y, dtype=complex).reshape(len(tlist))
        return y


@dataclass(frozen=True)
class CallableCoeffUnits:
    fn: Callable[[np.ndarray, float], np.ndarray]

    def eval(self, tlist: np.ndarray, time_unit_s: float) -> np.ndarray:
        y = self.fn(tlist, float(time_unit_s))
        return np.asarray(y, dtype=complex).reshape(len(tlist))


def scale_coeff(coeff: Optional[CoeffProto], factor: complex) -> CoeffProto:
    if coeff is None:
        return ConstCoeff(factor)

    def _fn(t: np.ndarray) -> np.ndarray:
        return factor * coeff.eval(t)

    return CallableCoeff(_fn)


def _eval_coeff_maybe(
    coeff: Any, tlist: np.ndarray, *, time_unit_s: float
) -> np.ndarray:
    if coeff is None:
        out = np.empty(len(tlist), dtype=complex)
        out[:] = 1.0 + 0.0j
        return out

    if hasattr(coeff, "eval"):
        try:
            return coeff.eval(tlist, float(time_unit_s))
        except TypeError:
            return coeff.eval(tlist)

    y = coeff(tlist)
    return np.asarray(y, dtype=complex).reshape(len(tlist))


def eval_coeff_any(coeff: Any, tlist: np.ndarray, *, time_unit_s: float) -> np.ndarray:
    if coeff is None:
        out = np.empty(len(tlist), dtype=complex)
        out[:] = 1.0 + 0.0j
        return out

    if hasattr(coeff, "eval"):
        try:
            return np.asarray(
                coeff.eval(tlist, float(time_unit_s)), dtype=complex
            ).reshape(len(tlist))
        except TypeError:
            return np.asarray(coeff.eval(tlist), dtype=complex).reshape(len(tlist))

    y = coeff(tlist)
    return np.asarray(y, dtype=complex).reshape(len(tlist))
