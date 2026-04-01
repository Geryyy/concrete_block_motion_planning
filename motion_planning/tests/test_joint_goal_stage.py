from __future__ import annotations

import math

import numpy as np

from motion_planning.pipeline import JointGoalStage


def _phi_tool_from_transform(T: np.ndarray) -> float:
    T_arr = np.asarray(T, dtype=float).reshape(4, 4)
    return float(math.atan2(T_arr[1, 1], T_arr[0, 1]))


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

    q_pin = np.asarray(stage._kin.pin_neutral, dtype=float).copy() if hasattr(stage._kin, "pin_neutral") else None
    if q_pin is None:
        import pinocchio as pin
        q_pin = np.asarray(pin.neutral(stage._kin.model), dtype=float)
    for jname, val in q_ref.items():
        if not stage._kin.model.existJointName(jname):
            continue
        jid = int(stage._kin.model.getJointId(jname))
        j = stage._kin.model.joints[jid]
        iq = int(j.idx_q)
        nq = int(j.nq)
        nv = int(j.nv)
        if nq == 1:
            q_pin[iq] = float(val)
        elif nq == 2 and nv == 1:
            q_pin[iq] = float(np.cos(val))
            q_pin[iq + 1] = float(np.sin(val))

    fk = stage._kin.forward_kinematics(
        q_pin, base_frame="world", end_frame="K8_tool_center_point"
    )
    T = np.asarray(fk["base_to_end"]["homogeneous"], dtype=float)
    goal_world = T[:3, 3]
    yaw = _phi_tool_from_transform(T)

    res = stage.solve_world_pose(goal_world=goal_world, target_yaw_rad=yaw, q_seed=q_ref)
    assert res.success
    assert res.q_dynamic["q4_big_telescope"] >= 0.0
    assert abs(res.q_dynamic["theta8_rotator_joint"]) <= math.pi
    assert res.fk_position_error_m < 2e-2
    assert abs(res.fk_yaw_error_rad) < 1e-2
