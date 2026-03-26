from __future__ import annotations

import numpy as np

from motion_planning.pipeline import JointGoalStage, JointSpaceCartesianPlanner

from ..evaluate import evaluate_plan
from ..types import StandalonePlanResult, StandaloneScenario


def plan_cartesian_anchor_joint_spline(scenario: StandaloneScenario) -> StandalonePlanResult:
    stage = JointGoalStage()
    planner = JointSpaceCartesianPlanner(
        urdf_path=stage.config.urdf_path,
        target_frame=stage.config.target_frame,
        reduced_joint_names=stage.config.actuated_joints,
    )

    if scenario.planner_start_q is not None:
        q_start = np.asarray(scenario.planner_start_q, dtype=float)
        start_seed = {name: float(q_start[i]) for i, name in enumerate(stage.config.actuated_joints)}
    else:
        start_solve = stage.solve_world_pose(goal_world=scenario.start_world_xyz, target_yaw_rad=scenario.start_yaw_rad, q_seed={})
        if not start_solve.success:
            return StandalonePlanResult(
                stack_name="cartesian_anchor_joint_spline",
                success=False,
                message=f"start solve failed: {start_solve.message}",
                q_waypoints=np.zeros((0, len(stage.config.actuated_joints)), dtype=float),
                tcp_xyz=np.zeros((0, 3), dtype=float),
                tcp_yaw_rad=np.zeros(0, dtype=float),
                reference_xyz=np.zeros((0, 3), dtype=float),
                reference_yaw_rad=np.zeros(0, dtype=float),
                diagnostics={},
            )
        q_start = np.asarray([start_solve.q_actuated.get(name, 0.0) for name in stage.config.actuated_joints], dtype=float)
        start_seed = dict(start_solve.q_dynamic)

    if scenario.planner_goal_q is not None:
        q_goal = np.asarray(scenario.planner_goal_q, dtype=float)
    else:
        goal_solve = stage.solve_world_pose(goal_world=scenario.goal_world_xyz, target_yaw_rad=scenario.goal_yaw_rad, q_seed=start_seed)
        if not goal_solve.success:
            return StandalonePlanResult(
                stack_name="cartesian_anchor_joint_spline",
                success=False,
                message=f"goal solve failed: {goal_solve.message}",
                q_waypoints=np.zeros((0, len(stage.config.actuated_joints)), dtype=float),
                tcp_xyz=np.zeros((0, 3), dtype=float),
                tcp_yaw_rad=np.zeros(0, dtype=float),
                reference_xyz=np.zeros((0, 3), dtype=float),
                reference_yaw_rad=np.zeros(0, dtype=float),
                diagnostics={},
            )
        q_goal = np.asarray([goal_solve.q_actuated.get(name, 0.0) for name in stage.config.actuated_joints], dtype=float)

    if q_start is None:
        return StandalonePlanResult(
            stack_name="cartesian_anchor_joint_spline",
            success=False,
            message="failed to initialize start joint state",
            q_waypoints=np.zeros((0, len(stage.config.actuated_joints)), dtype=float),
            tcp_xyz=np.zeros((0, 3), dtype=float),
            tcp_yaw_rad=np.zeros(0, dtype=float),
            reference_xyz=np.zeros((0, 3), dtype=float),
            reference_yaw_rad=np.zeros(0, dtype=float),
            diagnostics={},
        )
    n_wp = max(10, int(scenario.anchor_count) * 2)
    reference_xyz = np.vstack([
        np.linspace(scenario.start_world_xyz[i], scenario.goal_world_xyz[i], n_wp, dtype=float)
        for i in range(3)
    ]).T
    reference_yaw = np.linspace(scenario.start_yaw_rad, scenario.goal_yaw_rad, n_wp, dtype=float)

    plan = planner.plan(
        q_start=q_start,
        q_goal=q_goal,
        reference_xyz=reference_xyz,
        reference_yaw=reference_yaw,
    )
    result = StandalonePlanResult(
        stack_name="cartesian_anchor_joint_spline",
        success=bool(plan.success),
        message=plan.message,
        q_waypoints=np.asarray(plan.q_waypoints, dtype=float),
        tcp_xyz=np.asarray(plan.xyz_waypoints, dtype=float),
        tcp_yaw_rad=np.asarray(plan.yaw_waypoints, dtype=float),
        reference_xyz=np.asarray(reference_xyz, dtype=float),
        reference_yaw_rad=np.asarray(reference_yaw, dtype=float),
        diagnostics=dict(plan.diagnostics),
    )
    if result.success:
        evaluate_plan(result)
    return result
