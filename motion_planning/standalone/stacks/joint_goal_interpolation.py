from __future__ import annotations

import numpy as np

from ..common import (
    StandalonePlanningContext,
    build_planning_context,
    evaluate_tcp_path,
)
from ..evaluate import evaluate_plan
from ..reference_paths import build_linear_reference_path
from ..types import StandalonePlanResult, StandaloneScenario


def plan_joint_goal_interpolation(scenario: StandaloneScenario) -> StandalonePlanResult:
    planning = build_planning_context(
        scenario,
        stack_name="joint_goal_interpolation",
    )
    if isinstance(planning, StandalonePlanResult):
        return planning
    assert isinstance(planning, StandalonePlanningContext)

    n_wp = max(10, int(scenario.anchor_count) * 2)
    alpha = np.linspace(0.0, 1.0, n_wp, dtype=float).reshape(-1, 1)
    q_waypoints = (1.0 - alpha) * planning.q_start.reshape(1, -1) + alpha * planning.q_goal.reshape(1, -1)

    reference = build_linear_reference_path(scenario, waypoint_count=n_wp)
    xyz, yaw = evaluate_tcp_path(
        planning.planner,
        planning.actuated_joint_names,
        q_waypoints,
    )

    result = StandalonePlanResult(
        stack_name="joint_goal_interpolation",
        success=True,
        message="Solved start/goal with static IK and interpolated actuated joints.",
        q_waypoints=q_waypoints,
        tcp_xyz=np.asarray(xyz, dtype=float),
        tcp_yaw_rad=np.asarray(yaw, dtype=float),
        reference_xyz=np.asarray(reference.xyz, dtype=float),
        reference_yaw_rad=np.asarray(reference.yaw_rad, dtype=float),
        diagnostics={
            **reference.diagnostics,
            "waypoint_count": float(n_wp),
            "joint_anchor_fallback_used": 0.0,
        },
    )
    evaluate_plan(result)
    return result
