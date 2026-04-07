from __future__ import annotations

import math

from motion_planning import JointGoalStage
import numpy as np


def test_generate_linear_preapproach_targets_offsets_along_line():
    start = np.array([0.0, 0.0, 0.0], dtype=float)
    target = np.array([0.0, 3.0, 0.0], dtype=float)
    out = JointGoalStage.generate_linear_preapproach_targets(
        start, target, [0.0, 1.0, 2.0]
    )
    assert len(out) == 3
    assert np.allclose(out[0][1], [0.0, 3.0, 0.0])
    assert np.allclose(out[1][1], [0.0, 2.0, 0.0])
    assert np.allclose(out[2][1], [0.0, 1.0, 0.0])


def test_joint_goal_stage_solves_world_pose_from_fk_reachable_pose():
    stage = JointGoalStage()
    q_ref = {
        "theta1_slewing_joint": 0.05,
        "theta2_boom_joint": -0.85,
        "theta3_arm_joint": 0.55,
        "q4_big_telescope": 0.35,
        "theta8_rotator_joint": -0.05,
    }
    completed = stage._steady_state.complete_from_actuated(q_ref, q_seed=q_ref)
    assert completed.success, completed.message
    q_ref = dict(completed.q_dynamic)
    q_ref["q5_small_telescope"] = q_ref["q4_big_telescope"]

    goal_world, yaw, _ = stage._kin.pose_from_joint_map(
        q_ref,
        base_frame="world",
        end_frame="K8_tool_center_point",
    )

    res = stage.solve_world_pose(goal_world=goal_world, target_yaw_rad=yaw, q_seed=q_ref)
    assert res.success
    assert res.q_dynamic["q4_big_telescope"] >= 0.0
    assert abs(res.q_dynamic["theta8_rotator_joint"]) <= math.pi
    assert res.fk_position_error_m < 2e-2
    assert abs(res.fk_yaw_error_rad) < 1e-2
