"""VP-STO path planning standalone stack.

Uses CMA-ES-optimized B-spline trajectory (ported from timber VP-STO)
for the 5 actuated crane joints.
"""

from __future__ import annotations

import numpy as np

from motion_planning.planners.vpsto.crane_model import CraneVpstoModel
from motion_planning.planners.vpsto.solver import VpstoSolver

from ..common import (
    StandalonePlanningContext,
    build_planning_context,
    evaluate_tcp_path,
    fail_result,
)
from ..evaluate import evaluate_plan
from ..reference_paths import build_linear_reference_path
from ..types import StandalonePlanResult, StandaloneScenario


def plan_vpsto_path_planning(scenario: StandaloneScenario) -> StandalonePlanResult:
    planning = build_planning_context(
        scenario,
        stack_name="vpsto_path_planning",
    )
    if isinstance(planning, StandalonePlanResult):
        return planning
    assert isinstance(planning, StandalonePlanningContext)

    # ---- VP-STO target: [x, y, z, yaw] ----
    yd = np.array([
        scenario.goal_world_xyz[0],
        scenario.goal_world_xyz[1],
        scenario.goal_world_xyz[2],
        scenario.goal_yaw_rad,
    ], dtype=float)

    # Use validated goal joint config when available — bypasses the K5→K8
    # offset error in compute_dependent_joints() (CBS arm offset ~1 m)
    q_goal = (
        np.asarray(scenario.planner_goal_q, dtype=float)
        if scenario.planner_goal_q is not None
        else None
    )

    # ---- Run VP-STO ----
    model = CraneVpstoModel(n_eval=20)
    solver = VpstoSolver(model, n_via=5, n_eval=20, n_samples=32)

    vpsto_result = solver.solve(planning.q_start, yd, q_goal=q_goal)
    if not vpsto_result.success:
        return fail_result(
            stack_name="vpsto_path_planning",
            actuated_joint_count=len(planning.actuated_joint_names),
            message="VP-STO failed to find feasible path",
        )

    q_waypoints = vpsto_result.q_traj

    # ---- FK for TCP path ----
    tcp_xyz, tcp_yaw = evaluate_tcp_path(
        planning.planner,
        planning.actuated_joint_names,
        q_waypoints,
    )

    # Reference: straight line from actual FK start to intended goal
    reference = build_linear_reference_path(
        scenario,
        waypoint_count=len(q_waypoints),
        start_xyz=tcp_xyz[0],
        start_yaw_rad=float(tcp_yaw[0]),
    )

    result = StandalonePlanResult(
        stack_name="vpsto_path_planning",
        success=True,
        message=f"VP-STO: T={vpsto_result.T:.2f}s, {vpsto_result.iterations} CMA-ES iters, cost={vpsto_result.cost:.3f}",
        q_waypoints=q_waypoints,
        tcp_xyz=np.asarray(tcp_xyz, dtype=float),
        tcp_yaw_rad=np.asarray(tcp_yaw, dtype=float),
        reference_xyz=np.asarray(reference.xyz, dtype=float),
        reference_yaw_rad=np.asarray(reference.yaw_rad, dtype=float),
        diagnostics={
            **reference.diagnostics,
            "vpsto_T": vpsto_result.T,
            "vpsto_cost": vpsto_result.cost,
            "vpsto_iterations": float(vpsto_result.iterations),
            "waypoint_count": float(len(q_waypoints)),
            "joint_anchor_fallback_used": 0.0,
        },
    )
    evaluate_plan(result)
    return result
