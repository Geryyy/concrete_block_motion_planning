from __future__ import annotations

from dataclasses import dataclass

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as NavPath
from trajectory_msgs.msg import JointTrajectory


@dataclass(frozen=True)
class PlannerCapabilities:
    supports_move_empty: bool
    supports_named_configurations: bool
    supports_world_model_obstacles: bool
    supports_pick_place: bool
    supports_geometric_stage: bool


@dataclass
class BackendPlanResult:
    success: bool
    message: str
    trajectory: JointTrajectory
    cartesian_path: NavPath
    geometric_plan_id: str = ""


@dataclass
class A2BCompatibilityResult:
    success: bool
    message: str
    trajectory: JointTrajectory
    tcp_path: list[PoseStamped]
