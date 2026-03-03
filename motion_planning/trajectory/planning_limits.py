from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import yaml


def load_planning_limits_yaml(
    path: Path,
) -> Tuple[Dict[str, Tuple[Optional[float], Optional[float]]], Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload = payload or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid planning limits yaml format in {path}: expected mapping at top level.")

    joint_limits: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    joint_block = payload.get("joint_position_limits", {})
    if joint_block is not None:
        if not isinstance(joint_block, Mapping):
            raise ValueError(f"Invalid planning limits yaml format in {path}: 'joint_position_limits' must be a mapping.")
        for jn, bounds in joint_block.items():
            if not isinstance(bounds, Mapping):
                raise ValueError(f"Invalid position limits for joint '{jn}' in {path}: expected mapping.")
            lo = bounds.get("min", None)
            hi = bounds.get("max", None)
            lo_val = None if lo is None else float(lo)
            hi_val = None if hi is None else float(hi)
            if lo_val is not None and hi_val is not None and lo_val > hi_val:
                raise ValueError(f"Invalid position limits for joint '{jn}' in {path}: min > max.")
            joint_limits[str(jn)] = (lo_val, hi_val)

    taskspace_bounds = None
    ts_block = payload.get("taskspace_limits", None)
    if ts_block is not None:
        if not isinstance(ts_block, Mapping):
            raise ValueError(f"Invalid planning limits yaml format in {path}: 'taskspace_limits' must be a mapping.")
        min_xyz = ts_block.get("min_xyz", None)
        max_xyz = ts_block.get("max_xyz", None)
        if min_xyz is not None and max_xyz is not None:
            if len(min_xyz) != 3 or len(max_xyz) != 3:
                raise ValueError(f"Invalid taskspace limits in {path}: expected 3D min/max.")
            min_xyz_t = (float(min_xyz[0]), float(min_xyz[1]), float(min_xyz[2]))
            max_xyz_t = (float(max_xyz[0]), float(max_xyz[1]), float(max_xyz[2]))
            if any(a > b for a, b in zip(min_xyz_t, max_xyz_t)):
                raise ValueError(f"Invalid taskspace limits in {path}: min_xyz > max_xyz.")
            taskspace_bounds = (min_xyz_t, max_xyz_t)

    return joint_limits, taskspace_bounds
