from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from geometry_msgs.msg import PoseStamped

from ..compatibility import A2BCompatibilityRequest
from ..results import A2BCompatibilityResult, BackendPlanResult, PlannerCapabilities


class PlannerBackend(ABC):
    @property
    @abstractmethod
    def backend_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def capabilities(self) -> PlannerCapabilities:
        raise NotImplementedError

    @abstractmethod
    def plan_move_empty(
        self,
        *,
        start_pose: PoseStamped,
        goal_pose: PoseStamped,
        geometric_method: str,
        geometric_timeout_s: float,
        trajectory_method: str,
        trajectory_timeout_s: float,
        validate_dynamics: bool,
        planning_context: Dict[str, object],
    ) -> BackendPlanResult:
        raise NotImplementedError

    @abstractmethod
    def plan_a2b_compat(
        self,
        *,
        request: A2BCompatibilityRequest,
    ) -> A2BCompatibilityResult:
        raise NotImplementedError
