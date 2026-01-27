from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from smef.core.sim.types import MEProblemDense, MESolveResult


@runtime_checkable
class SolverAdapterProto(Protocol):
    def solve(
        self, problem: MEProblemDense, *, options: Optional[Mapping[str, Any]] = None
    ) -> MESolveResult: ...
