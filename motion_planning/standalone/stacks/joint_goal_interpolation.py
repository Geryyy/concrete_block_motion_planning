from __future__ import annotations

import numpy as np

from motion_planning.pipeline import JointGoalStage, JointSpaceCartesianPlanner

from ..evaluate import evaluate_plan
from ..types import StandalonePlanResult, StandaloneScenario


def plan_joint_goal_interpolation(scenario: StandaloneScenario) -> StandalonePlanResult:
    stage = JointGoalStage()
    planner = JointSpaceCartesianPlanner(
        urdf_path=stage.config.urdf_path,
        target_frame=stage.config.target_frame,
        reduced_joint_names=stage.config.actuated_joints,
    )

    q_start = None
    if scenario.planner_start_q is not None:
        q_start = np.asarray(scenario.planner_start_q, dtype=float)
    else:
        start_solve = stage.solve_world_pose(goal_world=scenario.start_world_xyz, target_yaw_rad=scenario.start_yaw_rad, q_seed={})
        if not start_solve.success:
            return StandalonePlanResult(
                stack_name="joint_goal_interpolation",
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
        start_seed = start_solve.q_dynamic
    
    if scenario.planner_goal_q is not None:
        q_goal = np.asarray(scenario.planner_goal_q, dtype=float)
    else:
        goal_solve = stage.solve_world_pose(
            goal_world=scenario.goal_world_xyz,
            target_yaw_rad=scenario.goal_yaw_rad,
            q_seed={} if scenario.planner_start_q is not None else start_seed,
        )
        if not goal_solve.success:
            return StandalonePlanResult(
                stack_name="joint_goal_interpolation",
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
            stack_name="joint_goal_interpolation",
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
    alpha = np.linspace(0.0, 1.0, n_wp, dtype=float).reshape(-1, 1)
    q_waypoints = (1.0 - alpha) * q_start.reshape(1, -1) + alpha * q_goal.reshape(1, -1)

    reference_xyz = np.vstack([
        np.linspace(scenario.start_world_xyz[i], scenario.goal_world_xyz[i], n_wp, dtype=float)
        for i in range(3)
    ]).T
    reference_yaw = np.linspace(scenario.start_yaw_rad, scenario.goal_yaw_rad, n_wp, dtype=float)

    xyz = []
    yaw = []
    q_seed = {name: float(q_start[i]) for i, name in enumerate(stage.config.actuated_joints)}
    for q in q_waypoints:
        xyz_i, yaw_i, q_seed = planner.fk_world_pose(q, q_seed=q_seed)
        xyz.append(xyz_i)
        yaw.append(yaw_i)

    result = StandalonePlanResult(
        stack_name="joint_goal_interpolation",
        success=True,
        message="Solved start/goal with static IK and interpolated actuated joints.",
        q_waypoints=q_waypoints,
        tcp_xyz=np.asarray(xyz, dtype=float),
        tcp_yaw_rad=np.asarray(yaw, dtype=float),
        reference_xyz=np.asarray(reference_xyz, dtype=float),
        reference_yaw_rad=np.asarray(reference_yaw, dtype=float),
        diagnostics={
            "waypoint_count": float(n_wp),
        },
    )
    evaluate_plan(result)
    return result
