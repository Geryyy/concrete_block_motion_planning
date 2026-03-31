from __future__ import annotations

import numpy as np

from motion_planning.core.types import PlannerRequest, Scenario
from motion_planning.geometry.arm_model import (
    DEFAULT_PZS100_RAIL_POSITION_M,
    CraneArmCollisionModel,
)
from motion_planning.planners.factory import create_planner
from motion_planning.scenarios import ScenarioLibrary


def test_complete_joint_map_uses_pzs100_rails() -> None:
    model = CraneArmCollisionModel()
    q_map = model.complete_joint_map(np.array([0.0, -0.7, 0.6, 0.4, 0.0], dtype=float))
    assert "q9_left_rail_joint" in q_map
    assert "q11_right_rail_joint" in q_map
    assert q_map["q9_left_rail_joint"] == DEFAULT_PZS100_RAIL_POSITION_M
    assert q_map["q11_right_rail_joint"] == DEFAULT_PZS100_RAIL_POSITION_M
    assert "theta10_outer_jaw_joint" not in q_map
    assert "theta12_inner_jaw_joint" not in q_map


def test_joint_space_stage1_planner_returns_full_contract() -> None:
    scenario_cfg = ScenarioLibrary().build_scenario("step_01_first_on_ground")
    req = PlannerRequest(
        scenario=Scenario(
            scene=scenario_cfg.scene,
            start=scenario_cfg.start,
            goal=scenario_cfg.goal,
            moving_block_size=scenario_cfg.moving_block_size,
            start_yaw_deg=scenario_cfg.start_yaw_deg,
            goal_yaw_deg=scenario_cfg.goal_yaw_deg,
            goal_normals=scenario_cfg.goal_normals,
        ),
        config={
            "joint_waypoint_count": 7,
            "n_samples_curve": 41,
            "preferred_safety_margin": 0.03,
            "approach_distance_m": 0.45,
        },
        options={"maxiter": 12, "seed": 5},
    )

    result = create_planner("Powell").plan(req)

    assert result.success, result.message
    q_path_full = np.asarray(result.diagnostics["q_path_full"], dtype=float)
    tcp_xyz = np.asarray(result.diagnostics["tcp_xyz_path"], dtype=float)
    assert q_path_full.ndim == 2
    assert q_path_full.shape[1] == 8
    assert tcp_xyz.shape[0] == q_path_full.shape[0]
    assert "combined_min_clearance_m" in result.diagnostics
    assert "approach_alignment_angle_deg" in result.diagnostics
    assert result.path.yaw_fn is not None
