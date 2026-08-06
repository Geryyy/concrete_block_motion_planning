"""Pure trajectory generation for grip movements.

No ROS dependencies. Operates on numpy arrays and returns structured data.
Uses cosine interpolation for DESCEND, a zero-acceleration minimum-jerk
profile for LIFT, and a time-optimal bounded trapezoidal profile for q9.

Four atomic primitives, sequenced by the behavior tree:
  DESCEND (1) — IK move from current position to target, preserves gripper state.
                Handles angled approach naturally when not directly above target.
  CLOSE   (2) — Close gripper at current position.
  OPEN    (3) — Open gripper at current position.
  LIFT    (4) — Lift from current position by lift_height.

Pick sequence:  OPEN → DESCEND → CLOSE → LIFT
Place sequence: DESCEND → OPEN → LIFT
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Phase codes (match BT CalcGripMovement "phase" attribute)
PHASE_DESCEND = 1
PHASE_CLOSE = 2
PHASE_OPEN = 3
PHASE_LIFT = 4


@dataclass
class GripTrajectoryConfig:
    dt: float = 0.01
    lift_height: float = 0.5  # meters above current position
    gripper_open_position: float = 0.15  # meters in q9's command coordinate
    gripper_close_position: float = 0.0
    # q9 limits.  They are deployment parameters, not inferred from a URDF:
    # real hardware commissioning owns their values.
    gripper_max_velocity_mps: float = 0.10
    gripper_max_acceleration_mps2: float = 0.25
    # Per-segment durations in seconds for the multi-joint crane motions.
    duration_descend: float = 5.0
    duration_lift: float = 5.0


@dataclass
class GripTrajectoryResult:
    success: bool
    q_traj: np.ndarray  # (N, n_joints) positions
    qd_traj: np.ndarray  # (N, n_joints) velocities
    qdd_traj: np.ndarray  # (N, n_joints) accelerations
    times: np.ndarray  # (N,) seconds from start
    message: str = ""


def _fail(n_joints: int, message: str) -> GripTrajectoryResult:
    e = np.empty((0, n_joints))
    return GripTrajectoryResult(False, e, e, e.copy(), np.empty(0), message)


def cosine_interpolate(
    q_start: np.ndarray,
    q_end: np.ndarray,
    duration: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cosine interpolation between two joint configurations.

    s(t) = 0.5 * (1 - cos(pi * t / T))  =>  smooth start/stop, zero velocity at endpoints.
    """
    n_pts = max(int(np.ceil(duration / dt)), 2)
    times = np.linspace(0.0, duration, n_pts)
    dq = q_end - q_start

    s = 0.5 * (1.0 - np.cos(np.pi * times / duration))
    s_dot = 0.5 * np.pi / duration * np.sin(np.pi * times / duration)
    s_ddot = 0.5 * (np.pi / duration) ** 2 * np.cos(np.pi * times / duration)

    positions = q_start[np.newaxis, :] + s[:, np.newaxis] * dq[np.newaxis, :]
    velocities = s_dot[:, np.newaxis] * dq[np.newaxis, :]
    accelerations = s_ddot[:, np.newaxis] * dq[np.newaxis, :]

    return positions, velocities, accelerations, times


def minimum_jerk_interpolate(
    q_start: np.ndarray,
    q_end: np.ndarray,
    duration: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate with zero velocity *and acceleration* at both endpoints.

    The quintic profile is deliberately used for lifting a freshly grasped
    block: compared with the cosine profile it avoids an acceleration step at
    lift-off, reducing excitation of the passive tip and tilt joints.
    """
    if not np.isfinite([duration, dt]).all() or duration <= 0.0 or dt <= 0.0:
        raise ValueError("minimum-jerk duration and dt must be positive and finite")

    n_pts = max(int(np.ceil(duration / dt)), 2)
    times = np.linspace(0.0, duration, n_pts)
    u = times / duration
    dq = q_end - q_start

    s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    s_dot = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / duration
    s_ddot = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / (duration**2)
    positions = q_start[np.newaxis, :] + s[:, np.newaxis] * dq[np.newaxis, :]
    velocities = s_dot[:, np.newaxis] * dq[np.newaxis, :]
    accelerations = s_ddot[:, np.newaxis] * dq[np.newaxis, :]
    velocities[[0, -1]] = 0.0
    accelerations[[0, -1]] = 0.0
    return positions, velocities, accelerations, times


def gripper_time_optimal_interpolate(
    q_start: float,
    q_end: float,
    max_velocity: float,
    max_acceleration: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a rest-to-rest, minimum-time q9 trapezoid/triangle profile.

    For one prismatic coordinate there is no path geometry to optimize, so an
    analytic profile is both simpler and exactly time-optimal under the given
    velocity and acceleration bounds.  The caller supplies only positive,
    finite limits; the returned profile includes both acceleration-switch
    points and the exact final time.
    """
    if not np.isfinite([q_start, q_end, max_velocity, max_acceleration, dt]).all():
        raise ValueError("gripper profile values must be finite")
    if max_velocity <= 0.0 or max_acceleration <= 0.0 or dt <= 0.0:
        raise ValueError("gripper velocity, acceleration, and dt must be positive")

    delta = q_end - q_start
    distance = abs(delta)
    if distance <= 1.0e-12:
        times = np.array([0.0])
        return np.array([q_start]), np.zeros(1), np.zeros(1), times

    direction = math.copysign(1.0, delta)
    ramp_time_at_limit = max_velocity / max_acceleration
    ramp_distance_at_limit = max_velocity * ramp_time_at_limit
    if distance <= ramp_distance_at_limit:
        # Triangular profile: the path is too short to reach max velocity.
        ramp_time = math.sqrt(distance / max_acceleration)
        peak_velocity = max_acceleration * ramp_time
        cruise_time = 0.0
    else:
        ramp_time = ramp_time_at_limit
        peak_velocity = max_velocity
        cruise_time = (distance - ramp_distance_at_limit) / max_velocity
    duration = 2.0 * ramp_time + cruise_time

    # Preserve the 100 Hz nominal command spacing, but include each change in
    # acceleration and the exact endpoint so the controller does not have to
    # extrapolate a phase boundary.
    regular_times = np.arange(0.0, duration, dt)
    # Round to nanoseconds before de-duplicating. A switch point and a grid
    # sample can land within one float ULP of each other -- e.g. 2.5 and
    # 2.5000000000000004 -- which np.unique keeps as distinct, but
    # JointTrajectory time_from_start is integer nanoseconds, so both collapse
    # onto the same stamp and the controller rejects the trajectory for not
    # being strictly increasing in time.
    times = np.unique(
        np.round(
            np.concatenate(
                (
                    regular_times,
                    np.array([0.0, ramp_time, ramp_time + cruise_time, duration]),
                )
            ),
            9,
        )
    )
    positions = np.empty_like(times)
    velocities = np.empty_like(times)
    accelerations = np.empty_like(times)
    ramp_distance = 0.5 * max_acceleration * ramp_time**2
    cruise_distance = peak_velocity * cruise_time
    for index, time_s in enumerate(times):
        if time_s <= ramp_time:
            travelled = 0.5 * max_acceleration * time_s**2
            velocity = max_acceleration * time_s
            acceleration = max_acceleration
        elif time_s <= ramp_time + cruise_time:
            travelled = ramp_distance + peak_velocity * (time_s - ramp_time)
            velocity = peak_velocity
            acceleration = 0.0
        else:
            decel_time = time_s - ramp_time - cruise_time
            travelled = (
                ramp_distance
                + cruise_distance
                + peak_velocity * decel_time
                - 0.5 * max_acceleration * decel_time**2
            )
            velocity = max(0.0, peak_velocity - max_acceleration * decel_time)
            acceleration = -max_acceleration
        positions[index] = q_start + direction * travelled
        velocities[index] = direction * velocity
        accelerations[index] = direction * acceleration

    positions[-1] = q_end
    velocities[0] = velocities[-1] = 0.0
    accelerations[-1] = 0.0
    return positions, velocities, accelerations, times


def compute_grip_trajectory(
    q0: np.ndarray,
    target_xyz: np.ndarray,
    phi_tool_n: float,
    phase: int,
    slow_down: float,
    ik_solve_fn,
    fk_fn,
    cfg: GripTrajectoryConfig,
    gripper_index: int = 7,
) -> GripTrajectoryResult:
    """Compute grip trajectory for the given phase.

    Args:
        q0: Current joint positions (n_joints,)
        target_xyz: Target TCP position [x, y, z] (used by DESCEND only)
        phi_tool_n: Target tool yaw (used by DESCEND and LIFT)
        phase: 1=descend, 2=close gripper, 3=open gripper, 4=lift
        slow_down: Speed factor (1.0 = normal, >1 = slower)
        ik_solve_fn: callable(xyz, yaw, seed_q) -> q_target or None
        fk_fn: callable(q) -> xyz (3,)
        cfg: Trajectory configuration
        gripper_index: Index of gripper joint in q vector
    """
    sd = max(slow_down, 0.1)

    if phase == PHASE_DESCEND:
        return _descend(q0, target_xyz, phi_tool_n, sd, ik_solve_fn, cfg, gripper_index)
    elif phase == PHASE_CLOSE:
        return _set_gripper(q0, cfg.gripper_close_position, sd, cfg, gripper_index)
    elif phase == PHASE_OPEN:
        return _set_gripper(q0, cfg.gripper_open_position, sd, cfg, gripper_index)
    elif phase == PHASE_LIFT:
        return _lift(q0, phi_tool_n, sd, ik_solve_fn, fk_fn, cfg, gripper_index)
    else:
        return _fail(len(q0), f"Unknown phase: {phase}")


def _descend(q0, target_xyz, phi_tool_n, sd, ik_solve_fn, cfg, grip_idx):
    """Move from current position to target. Gripper stays as-is.

    When not directly above the target this produces an angled approach,
    which helps slide against existing blocks during placement.
    """
    q_target = ik_solve_fn(target_xyz, phi_tool_n, q0)
    if q_target is None:
        return _fail(len(q0), "IK failed for descend target")
    q_target[grip_idx] = q0[grip_idx]
    pos, vel, acc, times = cosine_interpolate(
        q0,
        q_target,
        cfg.duration_descend * sd,
        cfg.dt,
    )
    return GripTrajectoryResult(True, pos, vel, acc, times)


def _set_gripper(q0, target_position, slow_down, cfg, grip_idx):
    """Move q9 from its measured position in the shortest bounded time."""
    try:
        # ``slow_down`` is an explicit request to be slower.  It may never
        # amplify q9 beyond the YAML safety limits, even if a caller sends a
        # value below one.  Scaling v by s and a by s² preserves the same
        # shape while scaling total duration by s.
        scale = max(float(slow_down), 1.0)
        positions, velocities, accelerations, times = gripper_time_optimal_interpolate(
            float(q0[grip_idx]),
            float(target_position),
            cfg.gripper_max_velocity_mps / scale,
            cfg.gripper_max_acceleration_mps2 / (scale * scale),
            cfg.dt,
        )
    except (IndexError, TypeError, ValueError) as error:
        return _fail(len(q0), f"invalid q9 time-parameterization: {error}")

    q_traj = np.repeat(q0[np.newaxis, :], len(times), axis=0)
    qd_traj = np.zeros_like(q_traj)
    qdd_traj = np.zeros_like(q_traj)
    q_traj[:, grip_idx] = positions
    qd_traj[:, grip_idx] = velocities
    qdd_traj[:, grip_idx] = accelerations
    return GripTrajectoryResult(True, q_traj, qd_traj, qdd_traj, times)


def _lift(q0, phi_tool_n, sd, ik_solve_fn, fk_fn, cfg, grip_idx):
    """Lift vertically from current position. Gripper stays as-is."""
    current_xyz = fk_fn(q0)

    # Try progressively smaller lift heights if IK fails
    for scale in [1.0, 0.75, 0.5, 0.25]:
        lift_xyz = current_xyz.copy()
        lift_xyz[2] += cfg.lift_height * scale
        q_lifted = ik_solve_fn(lift_xyz, phi_tool_n, q0)
        if q_lifted is not None:
            q_lifted[grip_idx] = q0[grip_idx]
            pos, vel, acc, times = minimum_jerk_interpolate(
                q0,
                q_lifted,
                cfg.duration_lift * sd,
                cfg.dt,
            )
            actual_lift = float(lift_xyz[2] - current_xyz[2])
            return GripTrajectoryResult(
                True,
                pos,
                vel,
                acc,
                times,
                message=f"lift_scale={scale:.2f} actual_lift={actual_lift:.3f}",
            )

    return _fail(len(q0), "IK failed for lift target")
