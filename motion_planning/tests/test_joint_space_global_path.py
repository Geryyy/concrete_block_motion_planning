from __future__ import annotations

from pathlib import Path

import numpy as np

from motion_planning.geometry.scene import Scene
from motion_planning.pipeline import (
    JointGoalStage,
    JointSpaceCartesianPlanner,
    JointSpaceGlobalPathRequest,
)
from motion_planning.standalone.scenarios import make_default_scenarios
from motion_planning.trajectory.planning_limits import load_planning_limits_yaml


def _planner():
    stage = JointGoalStage()
    repo_root = Path(__file__).resolve().parents[2]
    planning_limits = repo_root / "motion_planning" / "trajectory" / "planning_limits.yaml"
    joint_limits, _ = load_planning_limits_yaml(planning_limits)
    return JointSpaceCartesianPlanner(
        urdf_path=stage.config.urdf_path,
        target_frame=stage.config.target_frame,
        reduced_joint_names=stage.config.actuated_joints,
        joint_position_limits=joint_limits,
    )


def _straight_min_clearance(planner, q_start, q_goal, scene, moving_block_size) -> float:
    alpha = np.linspace(0.0, 1.0, 31, dtype=float).reshape(-1, 1)
    q_path = (1.0 - alpha) * np.asarray(q_start, dtype=float).reshape(1, -1) + alpha * np.asarray(q_goal, dtype=float).reshape(1, -1)
    dists = []
    q_seed = None
    for q in q_path:
        xyz, yaw, q_seed = planner.fk_world_pose(q, q_seed=q_seed)
        dists.append(
            scene.signed_distance_block(
                size=moving_block_size,
                position=xyz,
                quat=(0.0, 0.0, float(np.sin(0.5 * yaw)), float(np.cos(0.5 * yaw))),
            )
        )
    return float(np.min(np.asarray(dists, dtype=float)))


def test_joint_space_global_path_returns_two_vias_for_reachable_case() -> None:
    scenario = make_default_scenarios()["short_reachable_move"]
    planner = _planner()
    scene = Scene()

    result = planner.plan_global_path(
        JointSpaceGlobalPathRequest(
            scene=scene,
            moving_block_size=(0.15, 0.10, 0.10),
            q_start=np.asarray(scenario.planner_start_q, dtype=float),
            q_goal=np.asarray(scenario.planner_goal_q, dtype=float),
            start_approach_direction_world=scenario.start_approach_direction_world,
            goal_approach_direction_world=scenario.goal_approach_direction_world,
        )
    )

    assert result.success, result.message
    assert result.via_points.shape == (2, 5)
    assert result.q_waypoints.shape[1] == 5
    assert result.diagnostics["via_point_count"] == 2.0
    assert "min_signed_distance_m" in result.diagnostics


def test_joint_space_global_path_improves_clearance_over_straight_interpolation() -> None:
    scenario = make_default_scenarios()["short_reachable_move"]
    planner = _planner()
    scene = Scene()
    midpoint = 0.5 * (
        np.asarray(scenario.start_world_xyz, dtype=float) + np.asarray(scenario.goal_world_xyz, dtype=float)
    )
    scene.add_block(
        size=(0.18, 0.18, 0.18),
        position=(float(midpoint[0]), float(midpoint[1]), float(midpoint[2])),
        object_id="obstacle_mid",
    )

    baseline = _straight_min_clearance(
        planner,
        scenario.planner_start_q,
        scenario.planner_goal_q,
        scene,
        (0.15, 0.10, 0.10),
    )
    result = planner.plan_global_path(
        JointSpaceGlobalPathRequest(
            scene=scene,
            moving_block_size=(0.15, 0.10, 0.10),
            q_start=np.asarray(scenario.planner_start_q, dtype=float),
            q_goal=np.asarray(scenario.planner_goal_q, dtype=float),
            start_approach_direction_world=(1.0, 0.0, 0.0),
            goal_approach_direction_world=(1.0, 0.0, 0.0),
            config={"maxiter": 80, "w_collision": 120.0, "w_penetration": 600.0},
        )
    )

    assert result.diagnostics["min_signed_distance_m"] > baseline
