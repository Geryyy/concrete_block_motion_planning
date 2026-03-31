from __future__ import annotations

import numpy as np

from ..evaluate import evaluate_plan
from ..common import StandalonePlanningContext, build_planning_context
from ..reference_paths import build_linear_reference_path
from ..types import StandalonePlanResult, StandaloneScenario


def plan_cartesian_anchor_joint_spline(scenario: StandaloneScenario) -> StandalonePlanResult:
    planning = build_planning_context(
        scenario,
        stack_name="cartesian_anchor_joint_spline",
    )
    if isinstance(planning, StandalonePlanResult):
        return planning
    assert isinstance(planning, StandalonePlanningContext)

    n_wp = max(10, int(scenario.anchor_count) * 2)
    reference = build_linear_reference_path(scenario, waypoint_count=n_wp)

    plan = planning.planner.plan(
        q_start=planning.q_start,
        q_goal=planning.q_goal,
        reference_xyz=reference.xyz,
        reference_yaw=reference.yaw_rad,
    )
    result = StandalonePlanResult(
        stack_name="cartesian_anchor_joint_spline",
        success=bool(plan.success),
        message=plan.message,
        q_waypoints=np.asarray(plan.q_waypoints, dtype=float),
        tcp_xyz=np.asarray(plan.xyz_waypoints, dtype=float),
        tcp_yaw_rad=np.asarray(plan.yaw_waypoints, dtype=float),
        reference_xyz=np.asarray(reference.xyz, dtype=float),
        reference_yaw_rad=np.asarray(reference.yaw_rad, dtype=float),
        diagnostics={**reference.diagnostics, **dict(plan.diagnostics)},
    )
    if result.success:
        evaluate_plan(result)
    return result
