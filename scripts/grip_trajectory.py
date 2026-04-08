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
    lift_height: float = 0.5  # meters above current position
    gripper_open_angle: float = 0.15  # radians
    gripper_close_angle: float = 0.0
    default_block_radius: float = 0.30
    default_block_length: float = 0.90
    # Per-segment durations in seconds (for commissioning)
    duration_gripper_open: float = 2.0
    duration_descend: float = 5.0
    duration_gripper_close: float = 2.0
    duration_lift: float = 5.0


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


def scale_duration(base_duration: float, slow_down: float) -> float:
    """Apply slow_down factor to a base duration."""
    return base_duration * max(slow_down, 0.1)


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
    gripper_index: int = 7,
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
        gripper_index: Index of gripper joint in q vector
    """
    n_joints = len(q0)
    current_xyz = fk_fn(q0)

    if select_phases == 1:
        # Phase 1: approach + grip (descend to target, open gripper, close)
        return _phase_grip(q0, target_xyz, phi_tool_n, slow_down, ik_solve_fn, cfg, gripper_index)

    elif select_phases == 2:
        # Phase 2: lift (move up with block)
        return _phase_lift(q0, current_xyz, phi_tool_n, slow_down, ik_solve_fn, fk_fn, cfg, gripper_index)

    elif select_phases == 0:
        # Phase 0: laydown (descend, open gripper, retract up)
        return _phase_laydown(q0, target_xyz, phi_tool_n, slow_down, ik_solve_fn, cfg, gripper_index)

    elif select_phases == 3:
        # Phase 3: open + lift (failure recovery)
        return _phase_open_and_lift(q0, current_xyz, phi_tool_n, slow_down, ik_solve_fn, fk_fn, cfg, gripper_index)

    else:
        return GripTrajectoryResult(
            success=False, q_traj=np.empty((0, n_joints)), qd_traj=np.empty((0, n_joints)),
            qdd_traj=np.empty((0, n_joints)), times=np.empty(0),
            message=f"Unknown phase: {select_phases}",
        )


def _phase_grip(q0, target_xyz, phi_tool_n, slow_down, ik_solve_fn, cfg, grip_idx):
    """Phase 1: open gripper → descend → close gripper → lift up."""
    n = len(q0)

    # Solve IK for target position
    q_target = ik_solve_fn(target_xyz, phi_tool_n, q0)
    if q_target is None:
        return GripTrajectoryResult(
            success=False, q_traj=np.empty((0, n)), qd_traj=np.empty((0, n)),
            qdd_traj=np.empty((0, n)), times=np.empty(0), message="IK failed for grip target",
        )

    # Segment 1: open gripper in place
    q_open = q0.copy()
    q_open[grip_idx] = cfg.gripper_open_angle
    seg_open = cosine_interpolate(q0, q_open, scale_duration(cfg.duration_gripper_open, slow_down), cfg.dt)

    # Segment 2: descend to target (gripper stays open)
    q_at_target = q_target.copy()
    q_at_target[grip_idx] = cfg.gripper_open_angle
    seg_descend = cosine_interpolate(q_open, q_at_target, scale_duration(cfg.duration_descend, slow_down), cfg.dt)

    # Segment 3: close gripper at target
    q_closed = q_target.copy()
    q_closed[grip_idx] = cfg.gripper_close_angle
    seg_close = cosine_interpolate(q_at_target, q_closed, scale_duration(cfg.duration_gripper_close, slow_down), cfg.dt)

    # Segment 4: lift up with closed gripper
    lift_xyz = target_xyz.copy()
    lift_xyz[2] += cfg.lift_height
    q_lifted = ik_solve_fn(lift_xyz, phi_tool_n, q_closed)
    if q_lifted is not None:
        q_lifted[grip_idx] = cfg.gripper_close_angle
        seg_lift = cosine_interpolate(q_closed, q_lifted, scale_duration(cfg.duration_lift, slow_down), cfg.dt)
        pos, vel, acc, times = concatenate_segments([seg_open, seg_descend, seg_close, seg_lift])
    else:
        # Lift IK failed — return grip without lift
        pos, vel, acc, times = concatenate_segments([seg_open, seg_descend, seg_close])

    return GripTrajectoryResult(success=True, q_traj=pos, qd_traj=vel, qdd_traj=acc, times=times)


def _phase_lift(q0, current_xyz, phi_tool_n, slow_down, ik_solve_fn, fk_fn, cfg, grip_idx):
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
    # Keep gripper at current angle
    q_lifted[grip_idx] = q0[grip_idx]

    pos, vel, acc, times = cosine_interpolate(q0, q_lifted, scale_duration(cfg.duration_lift, slow_down), cfg.dt)
    return GripTrajectoryResult(success=True, q_traj=pos, qd_traj=vel, qdd_traj=acc, times=times)


def _phase_laydown(q0, target_xyz, phi_tool_n, slow_down, ik_solve_fn, cfg, grip_idx):
    """Phase 0: descend to target, open gripper, retract up."""
    n = len(q0)

    # Segment 1: descend to target
    q_target = ik_solve_fn(target_xyz, phi_tool_n, q0)
    if q_target is None:
        return GripTrajectoryResult(
            success=False, q_traj=np.empty((0, n)), qd_traj=np.empty((0, n)),
            qdd_traj=np.empty((0, n)), times=np.empty(0), message="IK failed for laydown target",
        )
    q_target[grip_idx] = q0[grip_idx]  # keep gripper closed during descent
    seg_desc = cosine_interpolate(q0, q_target, scale_duration(cfg.duration_descend, slow_down), cfg.dt)

    # Segment 2: open gripper
    q_open = q_target.copy()
    q_open[grip_idx] = cfg.gripper_open_angle
    seg_open = cosine_interpolate(q_target, q_open, scale_duration(cfg.duration_gripper_open, slow_down), cfg.dt)

    # Segment 3: lift
    lift_xyz = target_xyz.copy()
    lift_xyz[2] += cfg.lift_height
    q_lifted = ik_solve_fn(lift_xyz, phi_tool_n, q_open)
    if q_lifted is None:
        pos, vel, acc, times = concatenate_segments([seg_desc, seg_open])
        return GripTrajectoryResult(
            success=True, q_traj=pos, qd_traj=vel, qdd_traj=acc, times=times,
            message="Lift IK failed, returning descent+open only",
        )
    q_lifted[grip_idx] = cfg.gripper_open_angle
    seg_lift = cosine_interpolate(q_open, q_lifted, scale_duration(cfg.duration_lift, slow_down), cfg.dt)

    pos, vel, acc, times = concatenate_segments([seg_desc, seg_open, seg_lift])
    return GripTrajectoryResult(success=True, q_traj=pos, qd_traj=vel, qdd_traj=acc, times=times)


def _phase_open_and_lift(q0, current_xyz, phi_tool_n, slow_down, ik_solve_fn, fk_fn, cfg, grip_idx):
    """Phase 3: open gripper + lift (failure recovery)."""
    n = len(q0)

    # Segment 1: open gripper
    q_open = q0.copy()
    q_open[grip_idx] = cfg.gripper_open_angle
    seg_open = cosine_interpolate(q0, q_open, scale_duration(cfg.duration_gripper_open, slow_down), cfg.dt)

    # Segment 2: lift
    lift_xyz = current_xyz.copy()
    lift_xyz[2] += cfg.lift_height
    q_lifted = ik_solve_fn(lift_xyz, phi_tool_n, q_open)
    if q_lifted is None:
        pos, vel, acc, times = seg_open
        return GripTrajectoryResult(
            success=True, q_traj=pos, qd_traj=vel, qdd_traj=acc, times=times,
            message="Lift IK failed, returning gripper-open only",
        )
    q_lifted[grip_idx] = cfg.gripper_open_angle
    seg_lift = cosine_interpolate(q_open, q_lifted, scale_duration(cfg.duration_lift, slow_down), cfg.dt)

    pos, vel, acc, times = concatenate_segments([seg_open, seg_lift])
    return GripTrajectoryResult(success=True, q_traj=pos, qd_traj=vel, qdd_traj=acc, times=times)
