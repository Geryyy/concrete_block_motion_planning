"""Tests for the Cartesian task-space path-following trajectory optimizer."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from motion_planning.trajectory.cartesian_path_following import (
    CartesianPathFollowingConfig,
    CartesianPathFollowingOptimizer,
)
from motion_planning.mechanics.analytic import create_crane_config


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = Path(create_crane_config().urdf_path)


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
    return (
        DEFAULT_URDF.exists()
        and (src_path / "lib" / "link_libs.json").exists()
        and (src_path / "bin" / "t_renderer").exists()
    )


def _build_small_motion_states(urdf_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (q0, q_goal) full-model decision vectors differing only by a small
    theta1 slewing step (0.3 rad) so the OCP is easy to solve in a short horizon."""
    import pinocchio as pin
    full_model = pin.buildModelFromUrdf(str(urdf_path))
    # Start from a plausible resting pose (all joints at zero = nominal neutral).
    q0 = np.zeros(full_model.nv, dtype=float)
    # Non-zero passive joints at their rough equilibrium values from memory.
    joint_names = [str(full_model.names[jid]) for jid in range(1, full_model.njoints)]
    def _set(q, name, val):
        if name in [str(full_model.names[jid]) for jid in range(1, full_model.njoints)]:
            for jid in range(1, full_model.njoints):
                if str(full_model.names[jid]) == name:
                    q[int(full_model.joints[jid].idx_v)] = val
    _set(q0, "theta2_boom_joint", 0.9)
    _set(q0, "theta6_tip_joint", 0.76)
    _set(q0, "theta7_tilt_joint", 1.57)
    q_goal = q0.copy()
    _set(q_goal, "theta1_slewing_joint", 0.3)   # small slewing step
    return q0, q_goal


def _fk_tcp(urdf_path: Path, q_full: np.ndarray, tool_frame: str = "K8_tool_center_point") -> np.ndarray:
    """Numeric FK: full-model decision q → TCP xyz."""
    import pinocchio as pin
    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    q_pin = np.asarray(pin.neutral(model), dtype=float)
    for jid in range(1, model.njoints):
        jm = model.joints[jid]
        iv, iq, nq = int(jm.idx_v), int(jm.idx_q), int(jm.nq)
        if nq == 1:
            q_pin[iq] = q_full[iv]
        elif nq == 2 and int(jm.nv) == 1:
            th = q_full[iv]
            q_pin[iq], q_pin[iq + 1] = np.cos(th), np.sin(th)
    pin.forwardKinematics(model, data, q_pin)
    pin.updateFramePlacements(model, data)
    fid = model.getFrameId(tool_frame)
    return np.asarray(data.oMf[fid].translation, dtype=float).copy()


def test_config_instantiation() -> None:
    """CartesianPathFollowingConfig can be created with defaults."""
    cfg = CartesianPathFollowingConfig(urdf_path=DEFAULT_URDF)
    assert cfg.xyz_weight > 0
    assert cfg.terminal_xyz_weight > 0
    assert cfg.tool_frame_name == "K8_tool_center_point"
    assert cfg.spline_ctrl_points >= 2


@pytest.mark.skipif(not _acados_ready(), reason="acados/casadi/pinocchio runtime not available")
def test_cartesian_optimizer_smoke_straight_line() -> None:
    """Optimizer solves to status=0 with straight-line Cartesian fallback."""
    from motion_planning.core.types import TrajectoryRequest

    cfg = CartesianPathFollowingConfig(
        urdf_path=DEFAULT_URDF,
        horizon_steps=40,
        hessian_approx="GAUSS_NEWTON",
        nlp_solver_type="SQP",
        nlp_solver_max_iter=200,
        sdot_ref=0.15,
        code_export_dir=Path("/tmp/test_cartesian_pfc_codegen"),
        solver_json_name="test_cartesian_pfc.json",
    )
    q0, q_goal = _build_small_motion_states(DEFAULT_URDF)

    req = TrajectoryRequest(
        scenario=None,
        path=None,  # triggers straight-line Cartesian fallback
        config={"q0": q0, "q_goal": q_goal, "dq0": np.zeros_like(q0)},
    )
    opt = CartesianPathFollowingOptimizer(cfg)
    result = opt.optimize(req)

    assert result.success, (
        f"Solver failed: {result.message} (status={result.diagnostics.get('status')})"
    )

    N = cfg.horizon_steps
    assert result.state.shape == (N + 1, result.state.shape[1])
    assert result.control.shape[0] == N
    assert np.isfinite(result.state).all()
    assert np.isfinite(result.control).all()

    diag = result.diagnostics
    for key in ("q_trajectory", "dq_trajectory", "reduced_joint_names",
                "actuated_joint_names", "xyz_trajectory", "xyz_ref_trajectory"):
        assert key in diag, f"Missing diagnostic key: {key}"
    assert diag["xyz_trajectory"].shape == (N + 1, 3)
    assert diag["xyz_ref_trajectory"].shape == (N + 1, 3)


@pytest.mark.skipif(not _acados_ready(), reason="acados/casadi/pinocchio runtime not available")
def test_cartesian_optimizer_with_explicit_ctrl_pts() -> None:
    """Optimizer solves to status=0 with explicit Cartesian control points."""
    from motion_planning.core.types import TrajectoryRequest

    n_ctrl = 4
    cfg = CartesianPathFollowingConfig(
        urdf_path=DEFAULT_URDF,
        horizon_steps=40,
        spline_ctrl_points=n_ctrl,
        hessian_approx="GAUSS_NEWTON",
        nlp_solver_type="SQP",
        nlp_solver_max_iter=200,
        sdot_ref=0.15,
        code_export_dir=Path("/tmp/test_cartesian_pfc_explicit_codegen"),
        solver_json_name="test_cartesian_pfc_explicit.json",
    )
    q0, q_goal = _build_small_motion_states(DEFAULT_URDF)

    # Control points: straight line between actual TCP positions.
    xyz0 = _fk_tcp(DEFAULT_URDF, q0)
    xyzN = _fk_tcp(DEFAULT_URDF, q_goal)
    ctrl_pts_xyz = np.linspace(xyz0, xyzN, n_ctrl)

    req = TrajectoryRequest(
        scenario=None,
        path=None,
        config={
            "q0": q0,
            "q_goal": q_goal,
            "dq0": np.zeros_like(q0),
            "ctrl_pts_xyz": ctrl_pts_xyz,
        },
    )
    opt = CartesianPathFollowingOptimizer(cfg)
    result = opt.optimize(req)

    assert result.success, (
        f"Solver failed: {result.message} (status={result.diagnostics.get('status')})"
    )
    assert result.diagnostics["ctrl_pts_xyz"].shape == (n_ctrl, 3)
    assert np.isfinite(result.state).all()


def _build_large_motion_states(urdf_path: Path, dtheta1: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (q0, q_goal) differing by dtheta1 rad of slewing."""
    import pinocchio as pin
    full_model = pin.buildModelFromUrdf(str(urdf_path))
    q0 = np.zeros(full_model.nv, dtype=float)

    def _set(q, name, val):
        for jid in range(1, full_model.njoints):
            if str(full_model.names[jid]) == name:
                q[int(full_model.joints[jid].idx_v)] = val
                return

    _set(q0, "theta2_boom_joint", 0.9)
    _set(q0, "theta6_tip_joint", 0.76)
    _set(q0, "theta7_tilt_joint", 1.57)
    q_goal = q0.copy()
    _set(q_goal, "theta1_slewing_joint", dtheta1)
    return q0, q_goal


@pytest.mark.skipif(not _acados_ready(), reason="acados/casadi/pinocchio runtime not available")
def test_cartesian_optimizer_large_motion() -> None:
    """Optimizer solves to status=0 for a 1.5 rad slewing motion (large Cartesian arc).

    The NONLINEAR_LS + GAUSS_NEWTON formulation gives a PSD Hessian J^T W J,
    which avoids the indefinite Hessian that caused MINSTEP failures under the
    previous EXTERNAL + EXACT configuration.
    """
    from motion_planning.core.types import TrajectoryRequest

    cfg = CartesianPathFollowingConfig(
        urdf_path=DEFAULT_URDF,
        horizon_steps=80,
        hessian_approx="GAUSS_NEWTON",
        nlp_solver_type="SQP",
        nlp_solver_max_iter=300,
        sdot_ref=0.1,
        code_export_dir=Path("/tmp/test_cartesian_pfc_large_codegen"),
        solver_json_name="test_cartesian_pfc_large.json",
    )
    q0, q_goal = _build_large_motion_states(DEFAULT_URDF, dtheta1=1.5)

    req = TrajectoryRequest(
        scenario=None,
        path=None,
        config={"q0": q0, "q_goal": q_goal, "dq0": np.zeros_like(q0)},
    )
    opt = CartesianPathFollowingOptimizer(cfg)
    result = opt.optimize(req)

    assert result.success, (
        f"Large-motion solver failed: {result.message} (status={result.diagnostics.get('status')})"
    )
    assert np.isfinite(result.state).all()
    assert np.isfinite(result.control).all()
    assert result.diagnostics["xyz_trajectory"].shape == (cfg.horizon_steps + 1, 3)


@pytest.mark.skipif(not _acados_ready(), reason="acados/casadi/pinocchio runtime not available")
def test_cartesian_optimizer_fixed_time_workaround() -> None:
    """Fixed-time mode keeps the trajectory duration pinned to the requested T."""
    from motion_planning.core.types import TrajectoryRequest

    cfg = CartesianPathFollowingConfig(
        urdf_path=DEFAULT_URDF,
        horizon_steps=60,
        code_export_dir=Path("/tmp/test_cartesian_pfc_fixed_time_codegen"),
        solver_json_name="test_cartesian_pfc_fixed_time.json",
    )
    q0, q_goal = _build_small_motion_states(DEFAULT_URDF)

    req = TrajectoryRequest(
        scenario=None,
        path=None,
        config={
            "q0": q0,
            "q_goal": q_goal,
            "dq0": np.zeros_like(q0),
            "optimize_time": False,
            "fixed_time_duration_s": 10.0,
            "fixed_time_duration_candidates": (10.0,),
            "T_min": 10.0,
            "T_max": 10.0,
            "nlp_solver_max_iter": 300,
        },
    )
    opt = CartesianPathFollowingOptimizer(cfg)
    result = opt.optimize(req)

    assert result.success, (
        f"Fixed-time solver failed: {result.message} (status={result.diagnostics.get('status')})"
    )
    assert result.diagnostics["optimize_time"] is False
    assert float(result.time_s[-1]) == pytest.approx(10.0, abs=1e-6)
    assert np.allclose(np.asarray(result.diagnostics["T_trajectory"], dtype=float), 10.0, atol=1e-6)
