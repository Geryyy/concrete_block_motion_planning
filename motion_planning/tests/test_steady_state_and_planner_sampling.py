from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from motion_planning.api import plan as plan_path
from motion_planning.world_model import WorldModel
from motion_planning.mechanics import CraneKinematics
from motion_planning.mechanics import CraneSteadyState, ModelDescription, create_crane_config


_SAMPLE_RANGES: Dict[str, Tuple[float, float]] = {
    "theta1_slewing_joint": (-np.pi / 2.2, np.pi / 2.2),
    "theta2_boom_joint": (0.08, 0.40),
    "theta3_arm_joint": (0.05, 1.10),
    "q4_big_telescope": (0.08, 0.70),
    "theta8_rotator_joint": (-np.pi / 3, np.pi / 3),
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


def _q_dec_to_joint_map(model, q_dec: np.ndarray) -> dict[str, float]:
    q_map: dict[str, float] = {}
    for jid in range(1, model.njoints):
        j = model.joints[jid]
        if int(j.nq) in (1, 2):
            q_map[str(model.names[jid])] = float(q_dec[int(j.idx_v)])
    return q_map


def _sample_q_seed(kin: CraneKinematics, rng: np.random.Generator) -> np.ndarray:
    q = np.zeros(kin.model.nv, dtype=float)
    for jn, (lo, hi) in _SAMPLE_RANGES.items():
        jid = int(kin.model.getJointId(jn))
        q[int(kin.model.joints[jid].idx_v)] = float(rng.uniform(lo, hi))
    # telescope tie
    jid4 = int(kin.model.getJointId("q4_big_telescope"))
    jid5 = int(kin.model.getJointId("q5_small_telescope"))
    q[int(kin.model.joints[jid5].idx_v)] = q[int(kin.model.joints[jid4].idx_v)]
    return q


def _fk_pose_k0(kin: CraneKinematics, q_dec: np.ndarray, acfg) -> tuple[np.ndarray, float]:
    q_pin = _dec_to_pin_q(kin.model, q_dec)
    fk = kin.forward_kinematics(q_pin, base_frame=acfg.base_frame, end_frame=acfg.target_frame)
    T = np.asarray(fk["base_to_end"]["homogeneous"], dtype=float)
    pos = np.asarray(T[:3, 3], dtype=float).reshape(3)
    yaw = float(np.arctan2(T[1, 0], T[0, 0]))
    return pos, yaw


def _sample_valid_targets_with_steady_state(
    kin: CraneKinematics,
    ss: CraneSteadyState,
    acfg,
    *,
    n_keep: int,
    seed: int,
    max_attempts: int = 400,
) -> list[tuple[np.ndarray, float, object]]:
    rng = np.random.default_rng(seed)
    kept: list[tuple[np.ndarray, float, object]] = []
    attempts = 0
    while len(kept) < n_keep and attempts < max_attempts:
        attempts += 1
        q_seed = _sample_q_seed(kin, rng)
        target_pos, target_yaw = _fk_pose_k0(kin, q_seed, acfg)
        res = ss.compute(
            target_pos=np.asarray(target_pos, dtype=float),
            target_yaw=float(target_yaw),
            q_seed=_q_dec_to_joint_map(kin.model, q_seed),
        )
        if not res.success:
            continue
        if float(res.passive_residual) >= 1e-4:
            continue
        if float(res.ik_result.pos_error_m) >= 5e-3:
            continue
        kept.append((target_pos, target_yaw, res))
    return kept


def test_sampled_cartesian_poses_have_valid_steady_state_and_tcp_match():
    acfg = create_crane_config()
    kin = CraneKinematics(acfg.urdf_path)
    ss = CraneSteadyState(ModelDescription(acfg), acfg)
    samples = _sample_valid_targets_with_steady_state(kin, ss, acfg, n_keep=8, seed=7)
    assert len(samples) >= 8, "could not collect enough valid steady-state samples"

    for target_pos, target_yaw, ss_res in samples:
        assert ss_res.success, ss_res.message
        assert float(ss_res.passive_residual) < 1e-4
        assert float(ss_res.ik_result.pos_error_m) < 5e-3
        assert np.isfinite(float(target_pos[0] + target_pos[1] + target_pos[2]))
        assert np.isfinite(float(target_yaw))


def test_geometric_planner_finds_solutions_for_sampled_pose_pairs():
    acfg = create_crane_config()
    kin = CraneKinematics(acfg.urdf_path)
    ss = CraneSteadyState(ModelDescription(acfg), acfg)
    samples = _sample_valid_targets_with_steady_state(kin, ss, acfg, n_keep=10, seed=11)
    assert len(samples) >= 10, "could not collect enough valid steady-state samples"

    scene = WorldModel()
    pairs_checked = 0
    for i in range(0, len(samples) - 1, 2):
        start = np.asarray(samples[i][0], dtype=float)
        goal = np.asarray(samples[i + 1][0], dtype=float)
        if float(np.linalg.norm(goal - start)) < 0.15:
            continue
        res = plan_path(
            start=start,
            end=goal,
            world_model=scene,
            moving_block_size=(0.5, 0.5, 0.5),
            method="POWELL",
            config={"n_vias": 4, "n_samples_curve": 81, "safety_margin": 0.04},
            options={"maxiter": 18, "max_iter": 18, "population_size": 16},
        )
        assert res.success, res.message
        xyz = np.asarray(res.path.sample(81), dtype=float)
        assert xyz.shape[0] == 81
        assert float(np.linalg.norm(xyz[0] - start)) < 5e-2
        assert float(np.linalg.norm(xyz[-1] - goal)) < 5e-2
        pairs_checked += 1

    assert pairs_checked >= 3
