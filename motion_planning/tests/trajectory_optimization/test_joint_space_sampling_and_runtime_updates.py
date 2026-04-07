from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from motion_planning.types import TrajectoryRequest
from motion_planning.mechanics import CraneKinematics
from motion_planning.mechanics import CraneSteadyState, ModelDescription, create_crane_config
from motion_planning.trajectory.path_following import (
    CranePathFollowingAcadosConfig,
    CranePathFollowingAcadosOptimizer,
)


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


def _fk_pose(kin: CraneKinematics, q_dec: np.ndarray, acfg) -> tuple[np.ndarray, float]:
    q_pin = _dec_to_pin_q(kin.model, q_dec)
    fk = kin.forward_kinematics(q_pin, base_frame=acfg.base_frame, end_frame=acfg.target_frame)
    T = np.asarray(fk["base_to_end"]["homogeneous"], dtype=float)
    xyz = np.asarray(T[:3, 3], dtype=float).reshape(3)
    yaw = float(np.arctan2(T[1, 0], T[0, 0]))
    return xyz, yaw


def _steady_state_projection(kin: CraneKinematics, ss: CraneSteadyState, acfg, q_seed: np.ndarray) -> np.ndarray | None:
    xyz, yaw = _fk_pose(kin, q_seed, acfg)
    ss_res = ss.compute(
        target_pos=xyz,
        target_yaw=yaw,
        q_seed=_q_dec_to_joint_map(kin.model, q_seed),
    )
    if not ss_res.success:
        return None
    return _joint_map_to_q_dec(kin.model, q_seed, ss_res.q_dynamic)


def _set_joint(model, q: np.ndarray, joint_name: str, value: float) -> None:
    jid = int(model.getJointId(joint_name))
    q[int(model.joints[jid].idx_v)] = float(value)


def _accepted_convergence(res) -> bool:
    status = int(res.diagnostics.get("status", -999))
    if status == 0:
        return True
    if status != 2:
        return False
    residuals = np.asarray(res.diagnostics.get("nlp_residuals", np.array([np.inf])), dtype=float).reshape(-1)
    if residuals.size == 0:
        return False
    return float(residuals[0]) < 2e-4


def _full_to_reduced_q(q_full: np.ndarray, urdf_path: str) -> np.ndarray:
    import pinocchio as pin

    full_model = pin.buildModelFromUrdf(str(urdf_path))
    lock_joint_names = (
        "truck_pitch",
        "truck_roll",
        "q9_left_rail_joint",
        "q11_right_rail_joint",
    )
    lock_ids = [int(full_model.getJointId(n)) for n in lock_joint_names if int(full_model.getJointId(n)) != 0]
    reduced_model = pin.buildReducedModel(full_model, lock_ids, pin.neutral(full_model))
    full_v_map = {str(full_model.names[jid]): int(full_model.joints[jid].idx_v) for jid in range(1, full_model.njoints)}
    keep_names = [str(reduced_model.names[jid]) for jid in range(1, reduced_model.njoints)]
    q_full = np.asarray(q_full, dtype=float).reshape(-1)
    return np.asarray([q_full[full_v_map[n]] for n in keep_names], dtype=float)


def _sample_joint_state_pairs(n_pairs: int = 2) -> list[tuple[np.ndarray, np.ndarray]]:
    acfg = create_crane_config()
    kin = CraneKinematics(acfg.urdf_path)
    ss = CraneSteadyState(ModelDescription(acfg), acfg)

    q_seed = np.zeros(kin.model.nv, dtype=float)
    _set_joint(kin.model, q_seed, "theta1_slewing_joint", 0.10)
    _set_joint(kin.model, q_seed, "theta2_boom_joint", 0.24)
    _set_joint(kin.model, q_seed, "theta3_arm_joint", 0.90)
    _set_joint(kin.model, q_seed, "q4_big_telescope", 0.25)
    _set_joint(kin.model, q_seed, "q5_small_telescope", 0.25)
    _set_joint(kin.model, q_seed, "theta8_rotator_joint", 0.05)

    q_start = _steady_state_projection(kin, ss, acfg, q_seed)
    if q_start is None:
        raise RuntimeError("could not compute steady-state start configuration")

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for dth in [0.05, -0.06, 0.08]:
        q_end_seed = np.asarray(q_start, dtype=float).copy()
        jid = int(kin.model.getJointId("theta1_slewing_joint"))
        q_end_seed[int(kin.model.joints[jid].idx_v)] += dth
        q_end = _steady_state_projection(kin, ss, acfg, q_end_seed)
        if q_end is None:
            continue
        if np.linalg.norm(q_end - q_start) < 0.02:
            continue
        q_start_red = _full_to_reduced_q(q_start, acfg.urdf_path)
        q_end_red = _full_to_reduced_q(q_end, acfg.urdf_path)
        pairs.append((q_start_red, q_end_red))
        if len(pairs) >= n_pairs:
            break
    return pairs


@pytest.mark.skipif(not _acados_ready(), reason="acados/casadi/pinocchio runtime not available")
def test_joint_space_batch_sampled_start_end_configs() -> None:
    acfg = create_crane_config()
    pairs = _sample_joint_state_pairs(n_pairs=2)
    assert len(pairs) == 2, "could not sample enough steady-state joint-state pairs"

    cfg = CranePathFollowingAcadosConfig(
        urdf_path=Path(acfg.urdf_path),
        horizon_steps=80,
        precompile_on_init=True,
        code_export_dir=Path("/tmp/test_joint_path_batch_codegen"),
        solver_json_name="test_joint_path_batch.json",
    )
    optimizer = CranePathFollowingAcadosOptimizer(cfg)

    for q_start, q_end in pairs:
        req = TrajectoryRequest(
            scenario=None,
            path=None,
            config={
                "q0": q_start,
                "q_goal": q_end,
                "dq0": np.zeros_like(q_start),
                "nlp_solver_max_iter": 800,
                "qp_solver_iter_max": 300,
            },
        )
        res = optimizer.optimize(req)
        assert _accepted_convergence(res), f"joint-space sampled solve failed: {res.message}"
        assert float(res.diagnostics["s_trajectory"][-1]) >= 0.999


@pytest.mark.skipif(not _acados_ready(), reason="acados/casadi/pinocchio runtime not available")
def test_joint_space_runtime_parameter_updates_after_compile() -> None:
    acfg = create_crane_config()
    pairs = _sample_joint_state_pairs(n_pairs=1)
    assert len(pairs) == 1
    q_start, q_end = pairs[0]

    cfg = CranePathFollowingAcadosConfig(
        urdf_path=Path(acfg.urdf_path),
        horizon_steps=80,
        precompile_on_init=True,
        code_export_dir=Path("/tmp/test_joint_path_runtime_update_codegen"),
        solver_json_name="test_joint_path_runtime_update.json",
    )
    optimizer = CranePathFollowingAcadosOptimizer(cfg)
    assert optimizer._solver is not None

    req_a = TrajectoryRequest(
        scenario=None,
        path=None,
        config={
            "q0": q_start,
            "q_goal": q_end,
            "dq0": np.zeros_like(q_start),
            "nlp_solver_max_iter": 800,
            "nlp_tol": 1e-5,
            "qp_tol": 1e-5,
        },
    )
    res_a = optimizer.optimize(req_a)
    assert _accepted_convergence(res_a)
    solver_id_a = id(optimizer._solver)
    key_a = optimizer._solver_key

    req_b = TrajectoryRequest(
        scenario=None,
        path=None,
        config={
            "q0": q_start,
            "q_goal": q_end,
            "dq0": np.zeros_like(q_start),
            "nlp_solver_max_iter": 250,
            "nlp_tol": 5e-5,
            "qp_tol": 5e-5,
        },
    )
    res_b = optimizer.optimize(req_b)
    assert _accepted_convergence(res_b)
    assert id(optimizer._solver) == solver_id_a
    assert optimizer._solver_key == key_a
