from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
from scipy.optimize import least_squares

from motion_planning.control import ComputedTorqueController
from motion_planning.kinematics import CraneKinematics
from motion_planning.mechanics.analytic import CraneSteadyState, ModelDescription, create_crane_config


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCENE_XML = _REPO_ROOT / "crane_urdf" / "crane.xml"

_JNT_IV: Dict[str, int] = {
    "theta1_slewing_joint": 2,
    "theta2_boom_joint": 3,
    "theta3_arm_joint": 4,
    "q4_big_telescope": 5,
    "q5_small_telescope": 6,
    "theta6_tip_joint": 7,
    "theta7_tilt_joint": 8,
    "theta8_rotator_joint": 9,
    "q11_right_rail_joint": 10,
    "q9_left_rail_joint": 11,
}


def _dec_to_pin_q(pin_model, q_dec: np.ndarray) -> np.ndarray:
    q_pin = np.zeros(pin_model.nq, dtype=float)
    for jid in range(1, pin_model.njoints):
        j = pin_model.joints[jid]
        iq, iv, nq = int(j.idx_q), int(j.idx_v), int(j.nq)
        if nq == 1:
            q_pin[iq] = q_dec[iv]
        elif nq == 2:
            th = q_dec[iv]
            q_pin[iq], q_pin[iq + 1] = np.cos(th), np.sin(th)
    return q_pin


def _q_dec_to_joint_map(model, q_dec: np.ndarray) -> Dict[str, float]:
    q_map: Dict[str, float] = {}
    for jid in range(1, model.njoints):
        j = model.joints[jid]
        iv, nq = int(j.idx_v), int(j.nq)
        if nq in (1, 2):
            q_map[str(model.names[jid])] = float(q_dec[iv])
    return q_map


def _joint_map_to_q_dec(model, q_seed: np.ndarray, q_map: Dict[str, float]) -> np.ndarray:
    q_dec = np.asarray(q_seed, dtype=float).copy()
    for jid in range(1, model.njoints):
        jn = str(model.names[jid])
        if jn not in q_map:
            continue
        q_dec[int(model.joints[jid].idx_v)] = float(q_map[jn])
    q_dec[_JNT_IV["q5_small_telescope"]] = q_dec[_JNT_IV["q4_big_telescope"]]
    return q_dec


def _steady_state_q_dec(kin, ss, acfg, q_seed: np.ndarray) -> np.ndarray:
    q_cur = np.asarray(q_seed, dtype=float).copy()
    for _ in range(8):
        q_pin = _dec_to_pin_q(kin.model, q_cur)
        fk = kin.forward_kinematics(q_pin, base_frame=acfg.base_frame, end_frame=acfg.target_frame)
        T = np.asarray(fk["base_to_end"]["homogeneous"], dtype=float)
        target_pos = np.asarray(T[:3, 3], dtype=float).reshape(3)
        target_yaw = float(np.arctan2(T[1, 0], T[0, 0]))
        ss_res = ss.compute(target_pos=target_pos, target_yaw=target_yaw, q_seed=_q_dec_to_joint_map(kin.model, q_cur))
        if not ss_res.success:
            raise RuntimeError(f"steady-state solve failed: {ss_res.message}")
        q_new = _joint_map_to_q_dec(kin.model, q_cur, ss_res.q_dynamic)
        if np.linalg.norm(q_new - q_cur) < 1e-5:
            return q_new
        q_cur = q_new
    return q_cur


def _build_mujoco_maps(mujoco, model):
    name_to_qadr: Dict[str, int] = {}
    name_to_vadr: Dict[str, int] = {}
    for jid in range(model.njnt):
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        name_to_qadr[jn] = int(model.jnt_qposadr[jid])
        name_to_vadr[jn] = int(model.jnt_dofadr[jid])

    actuator_ids: List[int] = []
    actuator_joint_names: List[str] = []
    actuator_dof_idx: List[int] = []
    actuator_gear: List[float] = []
    for aid in range(model.nu):
        if int(model.actuator_trntype[aid]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        jid = int(model.actuator_trnid[aid, 0])
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        actuator_ids.append(aid)
        actuator_joint_names.append(jn)
        actuator_dof_idx.append(name_to_vadr[jn])
        g = float(model.actuator_gear[aid, 0])
        actuator_gear.append(g if abs(g) > 1e-12 else 1.0)
    return name_to_qadr, name_to_vadr, actuator_ids, actuator_joint_names, actuator_dof_idx, actuator_gear


def _reset_to_q_dec(mujoco, model, data, q_dec: np.ndarray, name_to_qadr: Dict[str, int]) -> None:
    q_dec = np.asarray(q_dec, dtype=float).reshape(-1)
    for jn, iv in _JNT_IV.items():
        if jn in name_to_qadr:
            data.qpos[name_to_qadr[jn]] = float(q_dec[iv])
    tel_half = 0.5 * float(q_dec[_JNT_IV["q4_big_telescope"]])
    data.qpos[name_to_qadr["q4_big_telescope"]] = tel_half
    data.qpos[name_to_qadr["q5_small_telescope"]] = tel_half
    q9_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "q9_left_rail_joint")
    q11_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "q11_right_rail_joint")
    data.qpos[name_to_qadr["q9_left_rail_joint"]] = float(model.jnt_range[q9_jid, 0])
    data.qpos[name_to_qadr["q11_right_rail_joint"]] = float(model.jnt_range[q11_jid, 0])
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _mujoco_refine_passive_equilibrium(
    mujoco,
    model,
    data,
    q_dec: np.ndarray,
    name_to_qadr: Dict[str, int],
    name_to_vadr: Dict[str, int],
) -> np.ndarray:
    _reset_to_q_dec(mujoco, model, data, q_dec, name_to_qadr)

    passive_names = ["theta6_tip_joint", "theta7_tilt_joint"]
    pas_qadr = [name_to_qadr[jn] for jn in passive_names if jn in name_to_qadr]
    pas_vadr = [name_to_vadr[jn] for jn in passive_names if jn in name_to_vadr]
    if not pas_qadr:
        return np.asarray(q_dec, dtype=float).copy()

    pas_jids = [
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn))
        for jn in passive_names
        if jn in name_to_qadr
    ]
    lo = np.asarray([float(model.jnt_range[jid, 0]) for jid in pas_jids], dtype=float)
    hi = np.asarray([float(model.jnt_range[jid, 1]) for jid in pas_jids], dtype=float)

    def residual(q_pas: np.ndarray) -> np.ndarray:
        for i, qa in enumerate(pas_qadr):
            data.qpos[qa] = float(q_pas[i])
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        return np.asarray([float(data.qfrc_bias[va]) for va in pas_vadr], dtype=float)

    q0 = np.asarray([float(data.qpos[qa]) for qa in pas_qadr], dtype=float)
    q0 = np.clip(q0, lo, hi)
    lsq = least_squares(
        residual,
        q0,
        bounds=(lo, hi),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=120,
    )
    q_pas_eq = q0 if not lsq.success else np.asarray(lsq.x, dtype=float)
    q_out = np.asarray(q_dec, dtype=float).copy()
    for i, jn in enumerate(passive_names):
        if jn in _JNT_IV and i < q_pas_eq.shape[0]:
            q_out[_JNT_IV[jn]] = float(q_pas_eq[i])
    return q_out


def test_headless_computed_torque_holds_steady_state_without_joint_drift():
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")

    acfg = create_crane_config()
    kin = CraneKinematics(acfg.urdf_path)
    ss = CraneSteadyState(ModelDescription(acfg), acfg)

    # Deterministic seed within operational envelope.
    q_seed = np.zeros(kin.model.nv, dtype=float)
    q_seed[_JNT_IV["theta1_slewing_joint"]] = 0.15
    q_seed[_JNT_IV["theta2_boom_joint"]] = 0.28
    q_seed[_JNT_IV["theta3_arm_joint"]] = 0.75
    q_seed[_JNT_IV["q4_big_telescope"]] = 0.32
    q_seed[_JNT_IV["q5_small_telescope"]] = q_seed[_JNT_IV["q4_big_telescope"]]
    q_seed[_JNT_IV["theta8_rotator_joint"]] = -0.20

    q_ss = _steady_state_q_dec(kin, ss, acfg, q_seed)

    model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    data = mujoco.MjData(model)
    (
        name_to_qadr,
        name_to_vadr,
        actuator_ids,
        actuator_joint_names,
        actuator_dof_idx,
        actuator_gear,
    ) = _build_mujoco_maps(mujoco, model)
    q_ss = _mujoco_refine_passive_equilibrium(mujoco, model, data, q_ss, name_to_qadr, name_to_vadr)
    _reset_to_q_dec(mujoco, model, data, q_ss, name_to_qadr)

    ctrl_min = model.actuator_ctrlrange[actuator_ids, 0].astype(float)
    ctrl_max = model.actuator_ctrlrange[actuator_ids, 1].astype(float)
    kp_vec = np.full(len(actuator_ids), 80.0, dtype=float)
    kd_vec = np.full(len(actuator_ids), 20.0, dtype=float)
    for i, jn in enumerate(actuator_joint_names):
        if jn in ("q9_left_rail_joint", "q11_right_rail_joint"):
            kp_vec[i] = max(kp_vec[i], 400.0)
            kd_vec[i] = max(kd_vec[i], 80.0)
    controller = ComputedTorqueController(kp=kp_vec, kd=kd_vec, u_min=ctrl_min, u_max=ctrl_max)

    q_des_all = np.asarray(data.qpos, dtype=float).copy()
    dq_des_all = np.zeros(model.nv, dtype=float)
    qdd_des_all = np.zeros(model.nv, dtype=float)

    passive_names = ["theta6_tip_joint", "theta7_tilt_joint"]
    monitor_names = [
        "theta1_slewing_joint",
        "theta2_boom_joint",
        "theta3_arm_joint",
        "q4_big_telescope",
        "theta8_rotator_joint",
        *passive_names,
    ]
    q0 = {jn: float(data.qpos[name_to_qadr[jn]]) for jn in monitor_names}
    max_abs_drift = {jn: 0.0 for jn in monitor_names}

    n_steps = 1500  # ~3s for dt=0.002
    for _ in range(n_steps):
        q_des = np.array([q_des_all[name_to_qadr[jn]] for jn in actuator_joint_names], dtype=float)
        dq_des = np.array([dq_des_all[name_to_vadr[jn]] for jn in actuator_joint_names], dtype=float)
        q_now = np.array([float(data.qpos[name_to_qadr[jn]]) for jn in actuator_joint_names], dtype=float)
        dq_now = np.array([float(data.qvel[name_to_vadr[jn]]) for jn in actuator_joint_names], dtype=float)
        qdd_ff = np.array([qdd_des_all[dof_idx] for dof_idx in actuator_dof_idx], dtype=float)
        qacc_cmd = controller.compute_acceleration(q_des=q_des, dq_des=dq_des, q=q_now, dq=dq_now, qdd_ff=qdd_ff)

        data.qacc[:] = qdd_des_all
        for i, dof_idx in enumerate(actuator_dof_idx):
            data.qacc[dof_idx] = float(qacc_cmd[i])
        mujoco.mj_inverse(model, data)

        tau_vec = np.array([float(data.qfrc_inverse[dof_idx]) for dof_idx in actuator_dof_idx], dtype=float)
        u_cmd, _ = controller.torque_to_control(tau=tau_vec, gear=np.asarray(actuator_gear, dtype=float))
        data.ctrl[:] = 0.0
        for i, aid in enumerate(actuator_ids):
            data.ctrl[aid] = float(u_cmd[i])
        mujoco.mj_step(model, data)

        for jn in monitor_names:
            q_now_j = float(data.qpos[name_to_qadr[jn]])
            max_abs_drift[jn] = max(max_abs_drift[jn], abs(q_now_j - q0[jn]))

    for jn in passive_names:
        assert max_abs_drift[jn] < 2.0e-3, f"{jn} drift too high: {max_abs_drift[jn]:.6f} rad"
    for jn in ("theta1_slewing_joint", "theta2_boom_joint", "theta3_arm_joint", "q4_big_telescope", "theta8_rotator_joint"):
        assert max_abs_drift[jn] < 2.0e-3, f"{jn} drift too high: {max_abs_drift[jn]:.6f} rad"
