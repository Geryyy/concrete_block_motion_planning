"""Regression tests for q9's time parameterization and the IK descent."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "scripts"))
sys.path.insert(0, str(PKG))

from grip_trajectory import (  # noqa: E402
    PHASE_CLOSE,
    PHASE_DESCEND,
    PHASE_LIFT,
    TELESCOPE_INDEX,
    GripTrajectoryConfig,
    compute_grip_trajectory,
    gripper_time_optimal_interpolate,
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
    assert (
        np.max(np.abs(normal.qd_traj[:, 7])) <= cfg.gripper_max_velocity_mps + 1.0e-12
    )


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

    def ik_solve_fn(
        target_xyz: np.ndarray, _yaw: float, seed_q: np.ndarray
    ) -> np.ndarray:
        q_target = seed_q.copy()
        q_target[3] = target_xyz[2]
        return q_target

    result = compute_grip_trajectory(
        q0, np.zeros(3), 0.0, PHASE_LIFT, 1.0, ik_solve_fn, fk_fn, cfg
    )

    assert result.success
    assert np.allclose(result.qdd_traj[[0, -1]], 0.0)
    assert np.isclose(result.q_traj[-1, 3], 0.20)


def test_gripper_times_stay_strictly_increasing_as_ros_durations() -> None:
    """Times must remain distinct after truncation to integer nanoseconds.

    JointTrajectory time_from_start is integer nanoseconds. A profile switch
    point and a sampling-grid point can differ by a single float ULP, which is
    distinct to np.unique but collapses onto one stamp in the message, and the
    controller then rejects the trajectory as not strictly increasing in time.
    0.300 -> 0.550 m is a case that used to collide at t = 2.5 s.
    """
    for start, end in [(0.300, 0.550), (0.550, 1.400), (1.400, 0.550), (0.0, 1.400)]:
        _, _, _, time_s = gripper_time_optimal_interpolate(start, end, 0.10, 0.25, 0.01)
        stamps = [(int(t), int((t - int(t)) * 1e9)) for t in time_s]
        assert all(stamps[i] > stamps[i - 1] for i in range(1, len(stamps))), (
            f"duplicate time_from_start for {start} -> {end}"
        )


def test_descend_stays_on_the_straight_line() -> None:
    """A joint-space line bows in task space; the descent must not (wall run 65, 7 cm)."""

    def fk(q):  # slew, radius, height: a joint-space line between two radii is an arc
        return np.array([q[1] * np.cos(q[0]), q[1] * np.sin(q[0]), q[2]])

    def ik(xyz, _yaw, seed, _fixed=None):  # q4 is not this stub's height axis
        q = np.array(seed, dtype=float)
        q[:3] = [np.arctan2(xyz[1], xyz[0]), np.hypot(xyz[0], xyz[1]), xyz[2]]
        return q

    q0 = np.array([0.0, 8.0, 3.7, 0.0, 0.0, 0.0, 0.0, 0.2])
    target = np.array([7.0, 2.0, 0.4])
    result = compute_grip_trajectory(
        q0, target, 0.0, 1, 1.0, ik, fk, GripTrajectoryConfig()
    )
    assert result.success
    start = fk(q0)
    line = (target - start) / np.linalg.norm(target - start)
    off_line = [
        np.linalg.norm((fk(q) - start) - np.dot(fk(q) - start, line) * line)
        for q in result.q_traj
    ]
    assert max(off_line) < 0.005
    assert np.allclose(fk(result.q_traj[-1]), target)
    assert result.q_traj[-1][7] == 0.2, "the gripper does not move"


# ---------------------------------------------------------------------------
# Descent with the real analytic IK.  The stub above cannot show the defect of
# issue 215: q4 steps come out of the solver's d45 grid, not out of _descend.
# ---------------------------------------------------------------------------

#: Joint order of grip_traj_server_simple's CONTROLLED_JOINT_NAMES.
IK_JOINT_NAMES = [
    "theta1_slewing_joint",
    "theta2_boom_joint",
    "theta3_arm_joint",
    "q4_big_telescope",
    "theta6_tip_joint",
    "theta7_tilt_joint",
    "theta8_rotator_joint",
    "q9_left_rail_joint",
]
IK_ACTUATED = [
    "theta1_slewing_joint",
    "theta2_boom_joint",
    "theta3_arm_joint",
    "q4_big_telescope",
    "theta8_rotator_joint",
]
#: theta7 of the free-hanging pendulum.  theta6 follows the arm, so it is left
#: to the equilibrium solve.  phi_tool == theta1 - theta8 holds only while the
#: tool axis is vertical, which is why the IK needs the pair refined per
#: waypoint and not frozen at its start value.
HANGING_TILT_RAD = 0.5 * np.pi


def _real_ik_harness():
    """(ik, fk) closures over the real solver, as grip_traj_server_simple builds them.

    Mirrors ``_ik_solve`` with ``use_passive_steady_state`` on: the passive pair
    is re-settled against the solved arm pose, since a frozen tip angle tilts
    the tool and breaks the solver's phi_tool == theta1 - theta8 assumption.
    """
    import pinocchio as pin
    from motion_planning.mechanics import ModelDescription, create_crane_config
    from motion_planning.mechanics.crane_geometry import DEFAULT_CRANE_GEOMETRY
    from motion_planning.mechanics.inverse_kinematics import AnalyticIKSolver
    from motion_planning.mechanics.pinocchio_utils import (
        fk_homogeneous,
        passive_joint_equilibrium,
    )
    from motion_planning.mechanics.pose_conventions import pose_from_pos_yaw

    config = create_crane_config()
    desc = ModelDescription(config)
    solver = AnalyticIKSolver(desc, config, DEFAULT_CRANE_GEOMETRY)
    eq_data = desc.model.createData()
    frame_cache: dict[str, int] = {}

    def q_map(q: np.ndarray) -> dict[str, float]:
        values = {name: float(q[i]) for i, name in enumerate(IK_JOINT_NAMES[:-1])}
        for follower, leader in config.tied_joints.items():
            values[follower] = values[leader]
        for name in config.locked_joints:
            values.setdefault(name, 0.0)
        return values

    def fk(q: np.ndarray) -> np.ndarray:
        return fk_homogeneous(
            pin_model=desc.model,
            pin_data=desc.data,
            pin_module=pin,
            q_values=q_map(q),
            base_frame="K0_mounting_base",
            end_frame="K8_tool_center_point",
            frame_cache=frame_cache,
        )[:3, 3].copy()

    def settle(values: dict[str, float], seed: dict[str, float]) -> dict[str, float]:
        return passive_joint_equilibrium(
            pin_model=desc.model,
            pin_data=eq_data,
            pin_module=pin,
            q_values=values,
            passive_joint_names=config.passive_joints,
            seed=seed,
        )

    def ik(xyz, yaw, seed_q, fixed_joints=None):
        pose = pose_from_pos_yaw(np.asarray(xyz, dtype=float), float(yaw))
        seed = {name: float(seed_q[i]) for i, name in enumerate(IK_JOINT_NAMES[:-1])}
        pinned = {k: float(v) for k, v in (fixed_joints or {}).items()}
        fixed = {**settle(q_map(seed_q), seed), **pinned}
        result = None
        for _ in range(3):
            result = solver.solve(
                target_T_base_to_end=pose,
                base_frame="K0_mounting_base",
                end_frame="K8_tool_center_point",
                seed=seed,
                act_names=IK_ACTUATED,
                fixed=fixed,
            )
            if result is None or not result.success:
                return None
            values = {**seed, **result.q_dynamic}
            for follower, leader in config.tied_joints.items():
                values[follower] = values[leader]
            for name in config.locked_joints:
                values.setdefault(name, 0.0)
            next_fixed = {**settle(values, fixed), **pinned}
            delta = max(abs(next_fixed[n] - fixed[n]) for n in config.passive_joints)
            fixed = next_fixed
            if delta < 1e-4:
                break
        q_out = np.asarray(seed_q, dtype=float).copy()
        for name, value in {**result.q_dynamic, **fixed}.items():
            if name in IK_JOINT_NAMES:
                q_out[IK_JOINT_NAMES.index(name)] = value
        return q_out

    return ik, fk


def test_real_ik_descent_does_not_step_the_telescope() -> None:
    """The logged jerky descent (issue 215), on the real IK.

    The solver searches d45 on a 120-point grid over the whole q4 range, so a
    free solve quantises q4 to ~19 mm; linear waypoint interpolation turns each
    step into a rectangular velocity pulse (max |ddq4| was 10-14 m/s^2).

    The long-reach case is the second one for a reason: with q4 pinned the d45
    search has no grid left, so the K5 correction has to bootstrap from the
    seed's arm angle there.  Free-solved, it measured 5.2 m/s^2.
    """
    ik, fk = _real_ik_harness()
    yaw = np.deg2rad(90.3)
    descents = [
        (np.array([4.06, 1.91, 2.01]), np.array([4.06, 1.87, 0.24])),  # logged
        (np.array([7.60, 2.60, 1.80]), np.array([7.60, 2.60, 0.40])),  # long reach
    ]

    for start, target in descents:
        seed = np.zeros(8)
        seed[1], seed[2], seed[3] = 0.9, 0.1, 0.4
        seed[5] = HANGING_TILT_RAD
        q0 = ik(start, yaw, seed)
        assert q0 is not None, f"start pose {start} must be solvable"
        assert np.linalg.norm(fk(q0) - start) < 0.02

        result = compute_grip_trajectory(
            q0,
            target,
            yaw,
            PHASE_DESCEND,
            1.0,
            ik,
            fk,
            GripTrajectoryConfig(dt=0.01, duration_descend=5.0),
        )
        assert result.success, result.message
        # A fallback near either end is smeared below the ddq4 bound by ds/dt -> 0,
        # so bound the fallback itself and not only its effect.
        assert result.message == "", "no waypoint may fall back to the free solve"

        q4 = result.q_traj[:, TELESCOPE_INDEX]
        ddq4 = np.gradient(np.gradient(q4, result.times), result.times)
        assert np.max(np.abs(ddq4)) < 1.0, (
            f"q4 steps on {start} -> {target}: max |ddq4|={np.max(np.abs(ddq4)):.2f}"
        )

        direction = (target - start) / np.linalg.norm(target - start)
        off_line = [
            float(np.linalg.norm(d - np.dot(d, direction) * direction))
            for d in (np.array([fk(q) for q in result.q_traj]) - start)
        ]
        assert max(off_line) < 0.005, (
            f"TCP leaves the line by {max(off_line) * 1e3:.1f} mm"
        )
        assert np.linalg.norm(fk(result.q_traj[-1]) - target) < 0.02
