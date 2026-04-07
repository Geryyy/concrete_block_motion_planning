from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from motion_planning import CartesianPathFollowingConfig, MotionPlanner, Scene
from motion_planning.mechanics import CraneKinematics
from motion_planning.mechanics import CraneSteadyState, ModelDescription, create_crane_config


def _acados_ready() -> bool:
    acados_src = os.environ.get("ACADOS_SOURCE_DIR", "")
    if not acados_src:
        return False
    src = Path(acados_src).expanduser().resolve()
    return (src / "lib" / "link_libs.json").exists() and (src / "bin" / "t_renderer").exists()


def _dec_to_pin_q(pin_model, q_dec: np.ndarray) -> np.ndarray:
    q_pin = np.zeros(pin_model.nq, dtype=float)
    for jid in range(1, pin_model.njoints):
        j = pin_model.joints[jid]
        iq, iv, nq = int(j.idx_q), int(j.idx_v), int(j.nq)
        if nq == 1:
            q_pin[iq] = q_dec[iv]
        elif nq == 2:
            th = q_dec[iv]
            q_pin[iq] = np.cos(th)
            q_pin[iq + 1] = np.sin(th)
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


def _solve_waypoint_steady_state(kin, ss, acfg, q_seed: np.ndarray, max_iter: int = 8, tol: float = 1e-5) -> np.ndarray:
    q_cur = np.asarray(q_seed, dtype=float).copy()
    for _ in range(max_iter):
        q_pin = _dec_to_pin_q(kin.model, q_cur)
        fk = kin.forward_kinematics(q_pin, base_frame=acfg.base_frame, end_frame=acfg.target_frame)
        T = np.asarray(fk["base_to_end"]["homogeneous"], dtype=float)
        target_pos = np.asarray(T[:3, 3], dtype=float).reshape(3)
        target_yaw = float(np.arctan2(T[1, 0], T[0, 0]))
        ss_res = ss.compute(target_pos=target_pos, target_yaw=target_yaw, q_seed=_q_dec_to_joint_map(kin.model, q_cur))
        if not ss_res.success:
            raise RuntimeError(f"steady-state solve failed: {ss_res.message}")
        q_new = _joint_map_to_q_dec(kin.model, q_cur, ss_res.q_dynamic)
        if np.linalg.norm(q_new - q_cur) < tol:
            return q_new
        q_cur = q_new
    return q_cur


def _point_to_polyline_distance(p: np.ndarray, poly: np.ndarray) -> float:
    best = float("inf")
    for i in range(poly.shape[0] - 1):
        a = poly[i]
        b = poly[i + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-12:
            d = float(np.linalg.norm(p - a))
        else:
            t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
            proj = a + t * ab
            d = float(np.linalg.norm(p - proj))
        if d < best:
            best = d
    return best


def test_geometric_path_and_trajectory_tcp_are_reasonably_aligned():
    if not _acados_ready():
        pytest.skip("acados is not configured (ACADOS_SOURCE_DIR/lib/link_libs.json/bin/t_renderer missing)")

    acfg = create_crane_config()
    urdf = Path(acfg.urdf_path)
    kin = CraneKinematics(urdf)
    ss = CraneSteadyState(ModelDescription(acfg), acfg)

    q_seed_start = np.zeros(kin.model.nv, dtype=float)
    q_seed_goal = np.zeros(kin.model.nv, dtype=float)
    q_seed_start[2] = -0.15
    q_seed_start[3] = 0.22
    q_seed_start[4] = 0.60
    q_seed_start[5] = 0.20
    q_seed_goal[2] = 0.35
    q_seed_goal[3] = 0.30
    q_seed_goal[4] = 0.95
    q_seed_goal[5] = 0.45

    q_start = _solve_waypoint_steady_state(kin, ss, acfg, q_seed_start)
    q_goal = _solve_waypoint_steady_state(kin, ss, acfg, q_seed_goal)

    traj_cfg = CartesianPathFollowingConfig(
        urdf_path=urdf,
        horizon_steps=60,
        nlp_solver_type="SQP",
        nlp_solver_max_iter=50,
        qp_solver_iter_max=80,
        qp_tol=1e-5,
        nlp_tol=1e-5,
        code_export_dir=Path("/tmp/test_path_vs_traj_codegen"),
        solver_json_name="test_path_vs_traj_ocp.json",
    )
    planner = MotionPlanner(
        scene=Scene(),
        method="POWELL",
        traj_config=traj_cfg,
        moving_block_size=(0.5, 0.5, 0.5),
        geometric_options={"accept_straight_line_if_feasible": True, "max_iter": 10, "population_size": 12},
    )
    planner.compile_trajectory_ocp(q_hint=q_start)
    res = planner.plan(q_start, q_goal)
    assert res.success, res.message
    assert res.trajectory is not None
    assert res.geometric_path is not None

    path_world = planner._base_pts_to_world(res.geometric_path.sample(200))

    traj = res.trajectory
    tcp_world = []
    for q in np.asarray(traj.q, dtype=float):
        q_pin = _dec_to_pin_q(kin.model, q)
        fk = kin.forward_kinematics(q_pin, base_frame=acfg.base_frame, end_frame=acfg.target_frame)
        p_base = np.asarray(fk["base_to_end"]["translation"], dtype=float).reshape(3)
        tcp_world.append(planner._base_pts_to_world(p_base.reshape(1, 3))[0])
    tcp_world = np.asarray(tcp_world, dtype=float)

    dists = np.asarray([_point_to_polyline_distance(p, path_world) for p in tcp_world], dtype=float)
    assert dists[0] < 0.05
    assert dists[-1] < 0.05
    assert float(np.median(dists)) < 0.25
    assert float(np.quantile(dists, 0.95)) < 0.60
