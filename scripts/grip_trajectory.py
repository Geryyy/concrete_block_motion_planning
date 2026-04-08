"""Pure trajectory generation for grip movements.

No ROS dependencies. Operates on numpy arrays and returns structured data.
Uses cosine interpolation for smooth start/stop motion profiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GripTrajectoryConfig:
    dt: float = 0.01
    max_joint_velocity: float = 0.3  # rad/s (or m/s for prismatic)
    lift_height: float = 0.5  # meters above current position
    jaw_open_angle: float = 0.15  # radians
    jaw_grip_angle: float = 0.0
    min_segment_duration: float = 1.0  # seconds
    default_block_radius: float = 0.30
    default_block_length: float = 0.90


@dataclass
class GripTrajectoryResult:
    success: bool
    q_traj: np.ndarray  # (N, n_joints) positions
    qd_traj: np.ndarray  # (N, n_joints) velocities
    qdd_traj: np.ndarray  # (N, n_joints) accelerations
    times: np.ndarray  # (N,) seconds from start
    message: str = ""


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

    # (n_pts, n_joints)
    positions = q_start[np.newaxis, :] + s[:, np.newaxis] * dq[np.newaxis, :]
    velocities = s_dot[:, np.newaxis] * dq[np.newaxis, :]
    accelerations = s_ddot[:, np.newaxis] * dq[np.newaxis, :]

    return positions, velocities, accelerations, times


def estimate_duration(
    q_start: np.ndarray,
    q_end: np.ndarray,
    max_vel: float,
    slow_down: float,
    min_duration: float = 1.0,
) -> float:
    """Estimate trajectory duration from max joint displacement."""
    max_disp = float(np.max(np.abs(q_end - q_start)))
    effective_vel = max_vel / max(slow_down, 0.1)
    if max_disp < 1e-6:
        return min_duration
    return max(max_disp / effective_vel, min_duration)


def concatenate_segments(
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate trajectory segments, adjusting time offsets."""
    if not segments:
        return np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)), np.empty(0)

    all_pos, all_vel, all_acc, all_t = [], [], [], []
    t_offset = 0.0

    for i, (pos, vel, acc, times) in enumerate(segments):
        if i > 0 and len(pos) > 0:
            # Skip first point of subsequent segments (duplicate of previous last)
            pos = pos[1:]
            vel = vel[1:]
            acc = acc[1:]
            times = times[1:]
        if len(pos) == 0:
            continue
        all_pos.append(pos)
        all_vel.append(vel)
        all_acc.append(acc)
        all_t.append(times + t_offset)
        t_offset = all_t[-1][-1]

    return (
        np.vstack(all_pos),
        np.vstack(all_vel),
        np.vstack(all_acc),
        np.concatenate(all_t),
    )


def compute_grip_trajectory(
    q0: np.ndarray,
    target_xyz: np.ndarray,
    phi_tool_n: float,
    select_phases: int,
    slow_down: float,
    ik_solve_fn,
    fk_fn,
    cfg: GripTrajectoryConfig,
    jaw_index: int = 7,
) -> GripTrajectoryResult:
    """Compute grip trajectory for the given phase.

    Args:
        q0: Current joint positions (n_joints,)
        target_xyz: Target TCP position [x, y, z]
        phi_tool_n: Target tool yaw
        select_phases: 0=laydown, 1=approach+grip, 2=lift, 3=open+lift
        slow_down: Speed reduction factor (1.0 = full speed)
        ik_solve_fn: callable(xyz, yaw, seed_q) -> q_target or None
        fk_fn: callable(q) -> xyz (3,)
        cfg: Trajectory configuration
        jaw_index: Index of jaw joint in q vector
    """
    n_joints = len(q0)
    current_xyz = fk_fn(q0)

    if select_phases == 1:
        # Phase 1: approach + grip (descend to target, open jaw, close)
        return _phase_grip(q0, target_xyz, phi_tool_n, slow_down, ik_solve_fn, cfg, jaw_index)

    elif select_phases == 2:
        # Phase 2: lift (move up with block)
        return _phase_lift(q0, current_xyz, phi_tool_n, slow_down, ik_solve_fn, fk_fn, cfg, jaw_index)

    elif select_phases == 0:
        # Phase 0: laydown (descend, open jaw, retract up)
        return _phase_laydown(q0, target_xyz, phi_tool_n, slow_down, ik_solve_fn, cfg, jaw_index)

    elif select_phases == 3:
        # Phase 3: open + lift (failure recovery)
        return _phase_open_and_lift(q0, current_xyz, phi_tool_n, slow_down, ik_solve_fn, fk_fn, cfg, jaw_index)

    else:
        return GripTrajectoryResult(
            success=False, q_traj=np.empty((0, n_joints)), qd_traj=np.empty((0, n_joints)),
            qdd_traj=np.empty((0, n_joints)), times=np.empty(0),
            message=f"Unknown phase: {select_phases}",
        )


def _phase_grip(q0, target_xyz, phi_tool_n, slow_down, ik_solve_fn, cfg, jaw_idx):
    """Phase 1: descend to target + close jaw."""
    n = len(q0)

    # Solve IK for target position
    q_target = ik_solve_fn(target_xyz, phi_tool_n, q0)
    if q_target is None:
        return GripTrajectoryResult(
            success=False, q_traj=np.empty((0, n)), qd_traj=np.empty((0, n)),
            qdd_traj=np.empty((0, n)), times=np.empty(0), message="IK failed for grip target",
        )

    # Segment 1: descend with jaw opening
    q_descend_end = q_target.copy()
    q_descend_end[jaw_idx] = cfg.jaw_open_angle
    q_descend_start = q0.copy()
    q_descend_start[jaw_idx] = cfg.jaw_open_angle  # open jaw at start too

    # First open jaw in place
    q_open = q0.copy()
    q_open[jaw_idx] = cfg.jaw_open_angle
    dur_open = estimate_duration(q0, q_open, cfg.max_joint_velocity, slow_down, 0.5)
    seg_open = cosine_interpolate(q0, q_open, dur_open, cfg.dt)

    # Then descend
    dur_descend = estimate_duration(q_open, q_descend_end, cfg.max_joint_velocity, slow_down)
    seg_descend = cosine_interpolate(q_open, q_descend_end, dur_descend, cfg.dt)

    # Segment 2: close jaw
    q_closed = q_target.copy()
    q_closed[jaw_idx] = cfg.jaw_grip_angle
    dur_close = estimate_duration(q_descend_end, q_closed, cfg.max_joint_velocity, slow_down, 0.5)
    seg_close = cosine_interpolate(q_descend_end, q_closed, dur_close, cfg.dt)

    pos, vel, acc, times = concatenate_segments([seg_open, seg_descend, seg_close])
    return GripTrajectoryResult(success=True, q_traj=pos, qd_traj=vel, qdd_traj=acc, times=times)


def _phase_lift(q0, current_xyz, phi_tool_n, slow_down, ik_solve_fn, fk_fn, cfg, jaw_idx):
    """Phase 2: lift block vertically."""
    n = len(q0)

    # Try progressively smaller lift heights if IK fails
    q_lifted = None
    for scale in [1.0, 0.75, 0.5, 0.25]:
        lift_xyz = current_xyz.copy()
        lift_xyz[2] += cfg.lift_height * scale
        q_lifted = ik_solve_fn(lift_xyz, phi_tool_n, q0)
        if q_lifted is not None:
            break
    if q_lifted is None:
        return GripTrajectoryResult(
            success=False, q_traj=np.empty((0, n)), qd_traj=np.empty((0, n)),
            qdd_traj=np.empty((0, n)), times=np.empty(0), message="IK failed for lift target",
        )
    # Keep jaw at current angle
    q_lifted[jaw_idx] = q0[jaw_idx]

    dur = estimate_duration(q0, q_lifted, cfg.max_joint_velocity, slow_down)
    pos, vel, acc, times = cosine_interpolate(q0, q_lifted, dur, cfg.dt)
    return GripTrajectoryResult(success=True, q_traj=pos, qd_traj=vel, qdd_traj=acc, times=times)


def _phase_laydown(q0, target_xyz, phi_tool_n, slow_down, ik_solve_fn, cfg, jaw_idx):
    """Phase 0: descend to target, open jaw, retract up."""
    n = len(q0)

    # Segment 1: descend to target
    q_target = ik_solve_fn(target_xyz, phi_tool_n, q0)
    if q_target is None:
        return GripTrajectoryResult(
            success=False, q_traj=np.empty((0, n)), qd_traj=np.empty((0, n)),
            qdd_traj=np.empty((0, n)), times=np.empty(0), message="IK failed for laydown target",
        )
    q_target[jaw_idx] = q0[jaw_idx]  # keep jaw closed during descent
    dur_desc = estimate_duration(q0, q_target, cfg.max_joint_velocity, slow_down)
    seg_desc = cosine_interpolate(q0, q_target, dur_desc, cfg.dt)

    # Segment 2: open jaw
    q_open = q_target.copy()
    q_open[jaw_idx] = cfg.jaw_open_angle
    dur_open = estimate_duration(q_target, q_open, cfg.max_joint_velocity, slow_down, 0.5)
    seg_open = cosine_interpolate(q_target, q_open, dur_open, cfg.dt)

    # Segment 3: lift
    lift_xyz = target_xyz.copy()
    lift_xyz[2] += cfg.lift_height
    q_lifted = ik_solve_fn(lift_xyz, phi_tool_n, q_open)
    if q_lifted is None:
        # Still return what we have
        pos, vel, acc, times = concatenate_segments([seg_desc, seg_open])
        return GripTrajectoryResult(
            success=True, q_traj=pos, qd_traj=vel, qdd_traj=acc, times=times,
            message="Lift IK failed, returning descent+open only",
        )
    q_lifted[jaw_idx] = cfg.jaw_open_angle
    dur_lift = estimate_duration(q_open, q_lifted, cfg.max_joint_velocity, slow_down)
    seg_lift = cosine_interpolate(q_open, q_lifted, dur_lift, cfg.dt)

    pos, vel, acc, times = concatenate_segments([seg_desc, seg_open, seg_lift])
    return GripTrajectoryResult(success=True, q_traj=pos, qd_traj=vel, qdd_traj=acc, times=times)


def _phase_open_and_lift(q0, current_xyz, phi_tool_n, slow_down, ik_solve_fn, fk_fn, cfg, jaw_idx):
    """Phase 3: open jaw + lift (failure recovery)."""
    n = len(q0)

    # Segment 1: open jaw
    q_open = q0.copy()
    q_open[jaw_idx] = cfg.jaw_open_angle
    dur_open = estimate_duration(q0, q_open, cfg.max_joint_velocity, slow_down, 0.5)
    seg_open = cosine_interpolate(q0, q_open, dur_open, cfg.dt)

    # Segment 2: lift
    lift_xyz = current_xyz.copy()
    lift_xyz[2] += cfg.lift_height
    q_lifted = ik_solve_fn(lift_xyz, phi_tool_n, q_open)
    if q_lifted is None:
        pos, vel, acc, times = cosine_interpolate(q0, q_open, dur_open, cfg.dt)
        return GripTrajectoryResult(
            success=True, q_traj=pos, qd_traj=vel, qdd_traj=acc, times=times,
            message="Lift IK failed, returning jaw-open only",
        )
    q_lifted[jaw_idx] = cfg.jaw_open_angle
    dur_lift = estimate_duration(q_open, q_lifted, cfg.max_joint_velocity, slow_down)
    seg_lift = cosine_interpolate(q_open, q_lifted, dur_lift, cfg.dt)

    pos, vel, acc, times = concatenate_segments([seg_open, seg_lift])
    return GripTrajectoryResult(success=True, q_traj=pos, qd_traj=vel, qdd_traj=acc, times=times)
