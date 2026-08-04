"""Regression tests for q9's bounded, minimum-time parameterization."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from grip_trajectory import (  # noqa: E402
    GripTrajectoryConfig,
    PHASE_CLOSE,
    PHASE_LIFT,
    gripper_time_optimal_interpolate,
    compute_grip_trajectory,
    minimum_jerk_interpolate,
)


def test_trapezoid_reaches_velocity_limit_and_exact_target() -> None:
    position, velocity, acceleration, time_s = gripper_time_optimal_interpolate(
        0.10, 0.40, max_velocity=0.10, max_acceleration=0.20, dt=0.01
    )

    # 0.30 m reaches v_max: T = d/v + v/a = 3.5 s.
    assert np.isclose(time_s[-1], 3.5)
    assert np.isclose(position[0], 0.10)
    assert np.isclose(position[-1], 0.40)
    assert np.isclose(velocity[0], 0.0)
    assert np.isclose(velocity[-1], 0.0)
    assert np.max(np.abs(velocity)) <= 0.10 + 1.0e-12
    assert np.max(np.abs(acceleration)) <= 0.20 + 1.0e-12


def test_short_move_uses_triangular_profile() -> None:
    _, velocity, acceleration, time_s = gripper_time_optimal_interpolate(
        0.10, 0.11, max_velocity=0.10, max_acceleration=0.20, dt=0.01
    )

    # The 0.01 m path cannot reach 0.10 m/s.
    assert np.isclose(time_s[-1], 2.0 * np.sqrt(0.01 / 0.20))
    assert np.max(np.abs(velocity)) < 0.10
    assert np.max(np.abs(acceleration)) <= 0.20 + 1.0e-12


def test_gripper_uses_current_q9_and_slowdown_only_reduces_speed() -> None:
    cfg = GripTrajectoryConfig(
        dt=0.01,
        gripper_close_position=0.20,
        gripper_max_velocity_mps=0.10,
        gripper_max_acceleration_mps2=0.20,
    )
    q0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.40])
    no_op_ik = lambda *_args: None
    no_op_fk = lambda _q: np.zeros(3)

    normal = compute_grip_trajectory(
        q0, np.zeros(3), 0.0, PHASE_CLOSE, 1.0, no_op_ik, no_op_fk, cfg
    )
    requested_faster = compute_grip_trajectory(
        q0, np.zeros(3), 0.0, PHASE_CLOSE, 0.25, no_op_ik, no_op_fk, cfg
    )
    slower = compute_grip_trajectory(
        q0, np.zeros(3), 0.0, PHASE_CLOSE, 2.0, no_op_ik, no_op_fk, cfg
    )

    assert normal.success and requested_faster.success and slower.success
    assert np.isclose(normal.q_traj[0, 7], 0.40)
    assert np.isclose(normal.q_traj[-1, 7], 0.20)
    assert np.allclose(normal.q_traj[:, :7], q0[:7])
    assert np.isclose(requested_faster.times[-1], normal.times[-1])
    assert np.isclose(slower.times[-1], 2.0 * normal.times[-1])
    assert np.max(np.abs(normal.qd_traj[:, 7])) <= cfg.gripper_max_velocity_mps + 1.0e-12


def test_minimum_jerk_lift_has_zero_start_and_end_acceleration() -> None:
    q0 = np.zeros(8)
    q_end = q0.copy()
    q_end[3] = 0.25
    position, velocity, acceleration, time_s = minimum_jerk_interpolate(
        q0, q_end, duration=4.0, dt=0.01
    )

    assert np.isclose(time_s[-1], 4.0)
    assert np.allclose(position[[0, -1]], np.array([q0, q_end]))
    assert np.allclose(velocity[[0, -1]], 0.0)
    assert np.allclose(acceleration[[0, -1]], 0.0)


def test_lift_uses_minimum_jerk_profile() -> None:
    cfg = GripTrajectoryConfig(dt=0.01, lift_height=0.20, duration_lift=4.0)
    q0 = np.zeros(8)

    def fk_fn(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], q[1], q[3]])

    def ik_solve_fn(target_xyz: np.ndarray, _yaw: float, seed_q: np.ndarray) -> np.ndarray:
        q_target = seed_q.copy()
        q_target[3] = target_xyz[2]
        return q_target

    result = compute_grip_trajectory(
        q0, np.zeros(3), 0.0, PHASE_LIFT, 1.0, ik_solve_fn, fk_fn, cfg
    )

    assert result.success
    assert np.allclose(result.qdd_traj[[0, -1]], 0.0)
    assert np.isclose(result.q_traj[-1, 3], 0.20)
