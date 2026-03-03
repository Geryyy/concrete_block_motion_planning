from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import yaml


def load_joint_accel_limits_yaml(path: Path) -> Tuple[Dict[str, Tuple[float, float]], Tuple[float, float]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload = payload or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid limits yaml format in {path}: expected mapping at top level.")

    default_block = payload.get("default", {})
    if not isinstance(default_block, Mapping):
        raise ValueError(f"Invalid limits yaml format in {path}: 'default' must be a mapping.")
    default_min = float(default_block.get("min", -1.0))
    default_max = float(default_block.get("max", 1.0))
    if default_min > default_max:
        raise ValueError(f"Invalid default limits in {path}: min > max.")

    joint_block = payload.get("joint_acceleration_limits", {})
    if not isinstance(joint_block, Mapping):
        raise ValueError(f"Invalid limits yaml format in {path}: 'joint_acceleration_limits' must be a mapping.")

    limits: Dict[str, Tuple[float, float]] = {}
    for jn, bounds in joint_block.items():
        if not isinstance(bounds, Mapping):
            raise ValueError(f"Invalid limits for joint '{jn}' in {path}: expected mapping.")
        lo = float(bounds["min"])
        hi = float(bounds["max"])
        if lo > hi:
            raise ValueError(f"Invalid limits for joint '{jn}' in {path}: min > max.")
        limits[str(jn)] = (lo, hi)
    return limits, (default_min, default_max)


def prepare_control_bounds_from_limits(
    *,
    req_config: Mapping[str, Any],
    actuated_joints: Sequence[str],
    act_v_idx: Sequence[int],
    reduced_name_to_vidx: Mapping[str, int],
    velocity_limits: np.ndarray,
    dt: float,
    joint_accel_limits_yaml: Path,
    validate_joint_limits_with_urdf: bool,
    qdd_u_min: float,
    qdd_u_max: float,
    v_min: float,
    v_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Path]:
    warnings: List[str] = []
    limits_yaml = Path(req_config.get("joint_accel_limits_yaml", joint_accel_limits_yaml))
    qdd_lims, default_lim = load_joint_accel_limits_yaml(limits_yaml)

    missing_yaml_actuated = [jn for jn in actuated_joints if jn not in qdd_lims]
    if missing_yaml_actuated:
        for jn in missing_yaml_actuated:
            qdd_lims[jn] = default_lim
        warnings.append(
            f"Missing per-joint accel limits in yaml for {missing_yaml_actuated}; using default {default_lim}."
        )

    if bool(req_config.get("validate_joint_limits_with_urdf", validate_joint_limits_with_urdf)):
        unknown_yaml_joints = sorted([jn for jn in qdd_lims if jn not in reduced_name_to_vidx])
        if unknown_yaml_joints:
            raise ValueError(f"joint_accel_limits_yaml has joints not present in reduced URDF model: {unknown_yaml_joints}")

    n_act = len(actuated_joints)
    lbu = np.zeros(n_act + 1, dtype=float)
    ubu = np.zeros(n_act + 1, dtype=float)
    accel_abs = np.zeros(n_act, dtype=float)
    for i, jn in enumerate(actuated_joints):
        lo_def, hi_def = qdd_lims.get(jn, (qdd_u_min, qdd_u_max))
        lbu[i] = float(req_config.get(f"{jn}_u_min", lo_def))
        ubu[i] = float(req_config.get(f"{jn}_u_max", hi_def))
        accel_abs[i] = max(abs(lbu[i]), abs(ubu[i]))
        vi = int(act_v_idx[i])
        v_urdf = abs(float(velocity_limits[vi]))
        if v_urdf > 0.0 and accel_abs[i] * dt > v_urdf:
            warnings.append(
                f"Joint '{jn}': |qdd|max*dt={accel_abs[i] * dt:.3f} exceeds URDF |dq|max={v_urdf:.3f}."
            )

    lbu[-1] = float(req_config.get("v_min", v_min))
    ubu[-1] = float(req_config.get("v_max", v_max))
    return lbu, ubu, accel_abs, warnings, limits_yaml
