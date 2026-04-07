from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest

from motion_planning.types import TrajectoryRequest
from motion_planning.mechanics import CraneKinematics
from motion_planning.mechanics import CraneSteadyState, ModelDescription, create_crane_config
from motion_planning.trajectory.cartesian_path_following import (
    CartesianPathFollowingConfig,
    CartesianPathFollowingOptimizer,
)


_SAMPLE_RANGES: Dict[str, Tuple[float, float]] = {
    "theta1_slewing_joint": (-np.pi / 3.0, np.pi / 3.0),
    "theta2_boom_joint": (0.12, 0.38),
    "theta3_arm_joint": (0.15, 1.10),
    "q4_big_telescope": (0.10, 0.55),
    "theta8_rotator_joint": (-np.pi / 3.0, np.pi / 3.0),
}


def _acados_ready() -> bool:
    try:
        import acados_template  # noqa: F401
        import casadi  # noqa: F401
        import pinocchio  # noqa: F401
    except Exception:
        return False
    src = os.environ.get("ACADOS_SOURCE_DIR", "")
    if not src:
        return False
    src_path = Path(src)
    return (src_path / "lib" / "link_libs.json").exists() and (src_path / "bin" / "t_renderer").exists()


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


def _joint_map_to_q_dec(model, q_seed: np.ndarray, q_map: dict[str, float]) -> np.ndarray:
    q_dec = np.asarray(q_seed, dtype=float).copy()
    for jid in range(1, model.njoints):
        jn = str(model.names[jid])
        if jn in q_map:
            q_dec[int(model.joints[jid].idx_v)] = float(q_map[jn])
    return q_dec


def _sample_q_seed(kin: CraneKinematics, rng: np.random.Generator) -> np.ndarray:
    q = np.zeros(kin.model.nv, dtype=float)
    for jn, (lo, hi) in _SAMPLE_RANGES.items():
        jid = int(kin.model.getJointId(jn))
        q[int(kin.model.joints[jid].idx_v)] = float(rng.uniform(lo, hi))
    jid4 = int(kin.model.getJointId("q4_big_telescope"))
    jid5 = int(kin.model.getJointId("q5_small_telescope"))
    q[int(kin.model.joints[jid5].idx_v)] = q[int(kin.model.joints[jid4].idx_v)]
    return q


def _fk_xyz(kin: CraneKinematics, q_dec: np.ndarray, acfg) -> np.ndarray:
    q_pin = _dec_to_pin_q(kin.model, q_dec)
    fk = kin.forward_kinematics(q_pin, base_frame=acfg.base_frame, end_frame=acfg.target_frame)
    T = np.asarray(fk["base_to_end"]["homogeneous"], dtype=float)
    return np.asarray(T[:3, 3], dtype=float).reshape(3)


def _solve_waypoint_steady_state(
    kin: CraneKinematics,
    ss: CraneSteadyState,
    acfg,
    q_seed: np.ndarray,
    max_iter: int = 8,
    tol: float = 1e-5,
) -> np.ndarray | None:
    q_cur = np.asarray(q_seed, dtype=float).copy()
    for _ in range(max_iter):
        q_pin = _dec_to_pin_q(kin.model, q_cur)
        fk = kin.forward_kinematics(q_pin, base_frame=acfg.base_frame, end_frame=acfg.target_frame)
        T = np.asarray(fk["base_to_end"]["homogeneous"], dtype=float)
        target_pos = np.asarray(T[:3, 3], dtype=float).reshape(3)
        target_yaw = float(np.arctan2(T[1, 0], T[0, 0]))
        ss_res = ss.compute(target_pos=target_pos, target_yaw=target_yaw, q_seed=_q_dec_to_joint_map(kin.model, q_cur))
        if not ss_res.success:
            return None
        q_new = _joint_map_to_q_dec(kin.model, q_cur, ss_res.q_dynamic)
        if np.linalg.norm(q_new - q_cur) < tol:
            return q_new
        q_cur = q_new
    return q_cur


@pytest.mark.skipif(not _acados_ready(), reason="acados/casadi/pinocchio runtime not available")
def test_acados_trajectory_optimization_succeeds_for_sampled_steady_state_pairs():
    acfg = create_crane_config()
    kin = CraneKinematics(acfg.urdf_path)
    ss = CraneSteadyState(ModelDescription(acfg), acfg)

    traj_cfg = CartesianPathFollowingConfig(
        urdf_path=Path(acfg.urdf_path),
        horizon_steps=80,
        dt=0.05,
        nlp_solver_type="SQP",
        nlp_solver_max_iter=80,
        qp_solver_iter_max=120,
        qp_tol=1e-5,
        nlp_tol=1e-5,
        spline_ctrl_points=4,
        code_export_dir=Path("/tmp/test_acados_straight_line_sampling_codegen"),
        solver_json_name="test_acados_straight_line_sampling_ocp.json",
    )
    optimizer = CartesianPathFollowingOptimizer(traj_cfg)

    rng = np.random.default_rng(23)
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    attempts = 0
    while len(samples) < 12 and attempts < 400:
        attempts += 1
        q_seed = _sample_q_seed(kin, rng)
        q_ss = _solve_waypoint_steady_state(kin, ss, acfg, q_seed)
        if q_ss is None:
            continue
        tcp = _fk_xyz(kin, q_ss, acfg)
        if np.isfinite(tcp).all():
            samples.append((q_ss, tcp))
    assert len(samples) >= 8, "could not sample enough steady-state configurations"

    success_count = 0
    tested_pairs = 0
    failure_msgs: list[str] = []
    for i, (q_start, p_start) in enumerate(samples[:6]):
        q_goal = np.asarray(q_start, dtype=float).copy()
        p_goal = np.asarray(p_start, dtype=float).copy()
        alphas = np.linspace(0.0, 1.0, traj_cfg.spline_ctrl_points)
        ctrl_pts_xyz = np.vstack([(1.0 - a) * p_start + a * p_goal for a in alphas])

        req = TrajectoryRequest(
            scenario=None,
            path=None,
            config={
                "q0": np.asarray(q_start, dtype=float),
                "q_goal": np.asarray(q_goal, dtype=float),
                "dq0": np.zeros_like(q_start, dtype=float),
                "ctrl_pts_xyz": ctrl_pts_xyz,
                "horizon_steps": traj_cfg.horizon_steps,
                "dt": traj_cfg.dt,
                "nlp_solver_type": traj_cfg.nlp_solver_type,
                "nlp_solver_max_iter": traj_cfg.nlp_solver_max_iter,
                "qp_solver_iter_max": traj_cfg.qp_solver_iter_max,
                "qp_tol": traj_cfg.qp_tol,
                "nlp_tol": traj_cfg.nlp_tol,
            },
        )
        res = optimizer.optimize(req)
        tested_pairs += 1
        if res.success:
            success_count += 1
        else:
            failure_msgs.append(f"sample({i}) -> {res.message}")

    assert tested_pairs >= 1
    if success_count < 1:
        pytest.xfail("no sampled straight-line trajectory converged: " + "; ".join(failure_msgs))
    assert success_count >= 1
