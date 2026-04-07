from __future__ import annotations

from pathlib import Path

import numpy as np

from motion_planning import JointGoalStage, JointSpaceCartesianPlanner
from motion_planning.trajectory.planning_limits import load_planning_limits_yaml


def _reduced_q(result, joint_names: list[str]) -> np.ndarray:
    source = result.q_dynamic if any(name in result.q_dynamic for name in joint_names) else result.q_actuated
    return np.asarray([float(source[name]) for name in joint_names], dtype=float)


def test_joint_space_cartesian_planner_tracks_reference_between_feasible_endpoints() -> None:
    stage = JointGoalStage()
    cfg = stage.config
    repo_root = Path(__file__).resolve().parents[2]
    planning_limits = (
        repo_root / "motion_planning" / "trajectory" / "planning_limits.yaml"
    )
    joint_limits, _ = load_planning_limits_yaml(planning_limits)

    completed = stage._steady_state.complete_from_actuated(
        {
            "theta1_slewing_joint": 0.05,
            "theta2_boom_joint": -0.85,
            "theta3_arm_joint": 0.55,
            "q4_big_telescope": 0.35,
            "theta8_rotator_joint": -0.05,
        }
    )
    assert completed.success, completed.message
    q_ref = dict(completed.q_dynamic)
    q_ref["q5_small_telescope"] = q_ref["q4_big_telescope"]
    start_world, yaw, _ = stage._kin.pose_from_joint_map(
        q_ref,
        base_frame="world",
        end_frame=cfg.target_frame,
    )
    reachable_goal = (float(start_world[0]), float(start_world[1] + 0.20), float(start_world[2]))
    start = stage.solve_world_pose(
        goal_world=start_world,
        target_yaw_rad=yaw,
        q_seed=q_ref,
    )
    goal = stage.solve_world_pose(
        goal_world=reachable_goal,
        target_yaw_rad=yaw,
        q_seed=q_ref,
    )
    assert start.success, start.message
    assert goal.success, goal.message

    joint_names = list(cfg.actuated_joints)
    planner = JointSpaceCartesianPlanner(
        urdf_path=cfg.urdf_path,
        target_frame=cfg.target_frame,
        reduced_joint_names=joint_names,
        joint_position_limits=joint_limits,
        maxiter=80,
    )

    q_start = _reduced_q(start, joint_names)
    q_goal = _reduced_q(goal, joint_names)
    alpha = np.linspace(0.0, 1.0, 6, dtype=float).reshape(-1, 1)
    reference_xyz = (1.0 - alpha) * start.goal_world.reshape(1, 3) + alpha * np.asarray(reachable_goal, dtype=float).reshape(1, 3)
    reference_yaw = np.full(6, yaw, dtype=float)

    result = planner.plan(
        q_start=q_start,
        q_goal=q_goal,
        reference_xyz=reference_xyz,
        reference_yaw=reference_yaw,
    )

    assert result.success, result.message
    assert result.q_waypoints.shape == (6, len(joint_names))
    assert np.allclose(result.q_waypoints[0], q_start)
    assert np.allclose(result.q_waypoints[-1], q_goal)
    assert result.diagnostics["objective_final"] <= result.diagnostics["objective_initial"] + 1e-6
    assert result.diagnostics["anchor_count"] >= 3.0
    assert result.diagnostics["solved_anchor_count"] >= 2.0
    assert "final_position_error_m" in result.diagnostics
    assert "max_anchor_polyline_deviation_m" in result.diagnostics


def test_anchor_selection_keeps_endpoints_and_caps_interior_count() -> None:
    indices = JointSpaceCartesianPlanner._select_anchor_indices(10, 4)
    assert indices[0] == 0
    assert indices[-1] == 9
    assert len(indices) == 6
    assert indices == sorted(indices)


def test_anchor_selection_uses_all_points_when_short_path() -> None:
    indices = JointSpaceCartesianPlanner._select_anchor_indices(5, 4)
    assert indices == [0, 1, 2, 3, 4]
