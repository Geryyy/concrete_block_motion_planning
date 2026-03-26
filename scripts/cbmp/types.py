from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as NavPath
from trajectory_msgs.msg import JointTrajectory

@dataclass
class StoredGeometricPlan:
    geometric_plan_id: str
    path: NavPath
    success: bool
    message: str
    method: str
    planner_path: Any = None


@dataclass
class StoredTrajectory:
    trajectory_id: str
    trajectory: JointTrajectory
    success: bool
    message: str
    method: str
    geometric_plan_id: str
    cartesian_path: NavPath = field(default_factory=NavPath)


@dataclass
class WallPlanTask:
    task_id: str
    target_block_id: str
    reference_block_id: str
    target_pose: PoseStamped
    reference_pose: PoseStamped
