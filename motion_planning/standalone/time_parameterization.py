from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin

from motion_planning.pipeline import JointGoalStage
from motion_planning.trajectory.limits import load_joint_accel_limits_yaml

from .types import StandalonePlanResult


def apply_simple_time_scaling(
    result: StandalonePlanResult,
    *,
    joint_names: list[str] | None = None,
    accel_limits_yaml: Path | None = None,
) -> StandalonePlanResult:
    q = np.asarray(result.q_waypoints, dtype=float)
    if q.ndim != 2 or q.shape[0] < 2:
        result.time_s = np.zeros(q.shape[0], dtype=float)
        result.dq_waypoints = np.zeros_like(q)
        return result

    stage = JointGoalStage()
    names = list(joint_names or stage.config.actuated_joints)
    vel_limits = []
    for name in names:
        jid = int(stage._kin.model.getJointId(name))
        joint = stage._kin.model.joints[jid]
        iq = int(joint.idx_q)
        vel_limits.append(abs(float(stage._kin.model.velocityLimit[iq])) if iq < stage._kin.model.velocityLimit.shape[0] else 1.0)
    vel = np.maximum(np.asarray(vel_limits, dtype=float), 1e-3)

    accel_path = accel_limits_yaml or (
        Path(__file__).resolve().parents[1] / "trajectory" / "joint_accel_limits.yaml"
    )
    accel_dict, accel_default = load_joint_accel_limits_yaml(accel_path)
    acc = []
    for name in names:
        lo, hi = accel_dict.get(name, accel_default)
        acc.append(max(abs(float(lo)), abs(float(hi)), 1e-3))
    acc_arr = np.asarray(acc, dtype=float)

    dq = np.diff(q, axis=0)
    seg_dt = np.zeros(dq.shape[0], dtype=float)
    for i in range(dq.shape[0]):
        dq_abs = np.abs(dq[i, :])
        vel_dt = np.max(dq_abs / vel)
        acc_dt = np.max(np.sqrt(2.0 * dq_abs / acc_arr))
        seg_dt[i] = max(float(vel_dt), float(acc_dt), 1e-3)

    time_s = np.concatenate([[0.0], np.cumsum(seg_dt)], dtype=float)
    dq_waypoints = np.zeros_like(q)
    dq_waypoints[1:, :] = dq / seg_dt.reshape(-1, 1)
    if dq_waypoints.shape[0] > 1:
        dq_waypoints[0, :] = dq_waypoints[1, :]

    result.time_s = time_s
    result.dq_waypoints = dq_waypoints
    result.diagnostics["timing_backend"] = "simple_limits"
    result.diagnostics["duration_s"] = float(time_s[-1])
    result.diagnostics["max_joint_speed_cmd"] = float(np.max(np.abs(dq_waypoints)))
    return result
