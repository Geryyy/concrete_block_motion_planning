"""Pure trajectory generation for grip movements.

No ROS dependencies. Operates on numpy arrays and returns structured data.
Uses cosine interpolation for smooth start/stop motion profiles.

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
    gripper_open_position: float = 0.15  # radians
    gripper_close_position: float = 0.0
    # Per-segment durations in seconds (for commissioning)
    duration_gripper: float = 2.0
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
        return _set_gripper(q0, cfg.gripper_close_position, sd, cfg)
    elif phase == PHASE_OPEN:
        return _set_gripper(q0, cfg.gripper_open_position, sd, cfg)
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
        q0, q_target, cfg.duration_descend * sd, cfg.dt,
    )
    return GripTrajectoryResult(True, pos, vel, acc, times)


def _set_gripper(q0, angle, sd, cfg):
    """Move gripper to target angle, all other joints stay."""
    q_end = q0.copy()
    q_end[-1] = angle
    pos, vel, acc, times = cosine_interpolate(
        q0, q_end, cfg.duration_gripper * sd, cfg.dt,
    )
    return GripTrajectoryResult(True, pos, vel, acc, times)


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
            pos, vel, acc, times = cosine_interpolate(
                q0, q_lifted, cfg.duration_lift * sd, cfg.dt,
            )
            return GripTrajectoryResult(True, pos, vel, acc, times)

    return _fail(len(q0), "IK failed for lift target")
