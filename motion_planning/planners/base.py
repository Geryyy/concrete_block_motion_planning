from __future__ import annotations

from typing import Protocol

from motion_planning.core.types import PlannerRequest, PlannerResult


class Planner(Protocol):
    """Common planner interface."""

    def plan(self, req: PlannerRequest) -> PlannerResult:
        ...
