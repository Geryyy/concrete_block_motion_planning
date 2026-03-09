from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from trajectory_msgs.msg import JointTrajectory

from .types import StoredGeometricPlan, StoredTrajectory, WallPlanTask


@dataclass
class RuntimeStatus:
    planning_runtime_ready: bool = False
    planning_runtime_reason: str = "runtime not initialized"
    trajectory_runtime_available: bool = False
    trajectory_runtime_reason: str = "not checked"
    geometric_runtime_available: bool = False
    geometric_runtime_reason: str = "not checked"


@dataclass
class MotionPlanningState:
    geometric_plans: Dict[str, StoredGeometricPlan] = field(default_factory=dict)
    trajectories: Dict[str, StoredTrajectory] = field(default_factory=dict)
    named_configurations: Dict[str, JointTrajectory] = field(default_factory=dict)
    wall_plans: Dict[str, List[WallPlanTask]] = field(default_factory=dict)
    wall_plan_progress: Dict[str, int] = field(default_factory=dict)

    runtime: RuntimeStatus = field(default_factory=RuntimeStatus)
    planner_scene: Any = None
    optimized_planner_params: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    trajectory_optimizers: Dict[str, Any] = field(default_factory=dict)

    analytic_cfg: Any = None
    steady_state_solver: Any = None
    reduced_joint_names: List[str] = field(default_factory=list)
    ik_seed_map: Dict[str, float] = field(default_factory=dict)
    t_world_base: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float))
    t_base_world: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float))
