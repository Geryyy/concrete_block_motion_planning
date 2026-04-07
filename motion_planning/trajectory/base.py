from __future__ import annotations

from typing import Protocol

from motion_planning.types import TrajectoryRequest, TrajectoryResult


class TrajectoryOptimizer(Protocol):
    """Common interface for dynamics-stage trajectory optimization."""

    def optimize(self, req: TrajectoryRequest) -> TrajectoryResult:
        ...
