from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StandaloneScenario:
    name: str
    description: str
    start_world_xyz: tuple[float, float, float]
    goal_world_xyz: tuple[float, float, float]
    start_yaw_rad: float = 0.0
    goal_yaw_rad: float = 0.0
    start_approach_direction_world: tuple[float, float, float] = (0.0, 0.0, -1.0)
    goal_approach_direction_world: tuple[float, float, float] = (0.0, 0.0, -1.0)
    planner_start_q: tuple[float, ...] | None = None
    planner_goal_q: tuple[float, ...] | None = None
    anchor_count: int = 6
    overlay_scene_name: str | None = None
    overlay_scene_translation: tuple[float, float, float] | None = None


@dataclass
class PlanEvaluation:
    final_position_error_m: float
    final_yaw_error_deg: float
    max_position_error_m: float
    mean_position_error_m: float
    max_path_deviation_m: float
    path_length_m: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StandalonePlanResult:
    stack_name: str
    success: bool
    message: str
    q_waypoints: np.ndarray
    tcp_xyz: np.ndarray
    tcp_yaw_rad: np.ndarray
    reference_xyz: np.ndarray
    reference_yaw_rad: np.ndarray
    time_s: np.ndarray | None = None
    dq_waypoints: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    evaluation: PlanEvaluation | None = None
