from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class OpMaterializeContextProto(Protocol):
    def resolve_symbol(self, symbol: str, dims: Sequence[int]) -> np.ndarray: ...

    def resolve_embedded(self, embedded: Any, dims: Sequence[int]) -> np.ndarray: ...
