from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml


_CONFIG_DIR = Path("/workspaces/ros2_baustelle_ws/src/epsilon_crane_description/config")
_KEY_MAP = {
    "theta1_0": "theta1_slewing_joint",
    "theta2_0": "theta2_boom_joint",
    "theta3_0": "theta3_arm_joint",
    "q4_0": "q4_big_telescope",
    "theta6_0": "theta6_tip_joint",
    "theta7_0": "theta7_tilt_joint",
    "theta8_0": "theta8_rotator_joint",
    "q9_0": "q9_left_rail_joint",
    "q11_0": "q11_right_rail_joint",
    "boom_cylinder_piston_in_barrel_linear_joint": "boom_cylinder_piston_in_barrel_linear_joint",
    "boom_cylinder_mounting_on_slewing_column": "boom_cylinder_mounting_on_slewing_column",
    "boom_cylinder_linkage_big_mounting_on_slewing_column": "boom_cylinder_linkage_big_mounting_on_slewing_column",
    "boom_cylinder_linkage_small_mounting_on_boom": "boom_cylinder_linkage_small_mounting_on_boom",
}


@dataclass(frozen=True)
class ReferenceState:
    name: str
    q_map: dict[str, float]


def _load_reference_state(path: Path) -> ReferenceState:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    q_map = {
        joint_name: float(data[yaml_key])
        for yaml_key, joint_name in _KEY_MAP.items()
        if yaml_key in data
    }
    if "q4_big_telescope" in q_map:
        q_map["q5_small_telescope"] = q_map["q4_big_telescope"]
    arm_cyl = data.get("length_arm_cylinder_piston_in_barrel_linear_joint")
    if arm_cyl is not None:
        q_map["arm_cylinder_piston_in_barrel_linear_joint_left"] = float(arm_cyl)
        q_map["arm_cylinder_piston_in_barrel_linear_joint_right"] = float(arm_cyl)
    return ReferenceState(path.stem, q_map)


@lru_cache(maxsize=1)
def load_reference_states() -> tuple[ReferenceState, ...]:
    if not _CONFIG_DIR.exists():
        return ()
    return tuple(_load_reference_state(path) for path in sorted(_CONFIG_DIR.glob("*.yaml")))


def _reference_score(reference: Mapping[str, float], query: Mapping[str, float]) -> float:
    keys = [name for name in query if name in reference]
    if not keys:
        return float("inf")
    return float(np.linalg.norm([float(reference[name]) - float(query[name]) for name in keys]))


def best_reference_state(query: Mapping[str, float] | None = None) -> ReferenceState | None:
    states = load_reference_states()
    if not states:
        return None
    if not query:
        return states[0]
    return min(states, key=lambda state: _reference_score(state.q_map, query))


def merge_reference_seed(
    q_seed: Mapping[str, float] | None = None,
    *,
    q_actuated: Mapping[str, float] | None = None,
) -> dict[str, float]:
    merged = dict(q_seed or {})
    query = dict(q_actuated or {})
    query.update({k: float(v) for k, v in merged.items() if k in {
        "theta1_slewing_joint",
        "theta2_boom_joint",
        "theta3_arm_joint",
        "q4_big_telescope",
        "theta8_rotator_joint",
    }})
    reference = best_reference_state(query)
    if reference is None:
        return merged
    out = dict(reference.q_map)
    out.update(merged)
    if "q4_big_telescope" in out and "q5_small_telescope" not in out:
        out["q5_small_telescope"] = float(out["q4_big_telescope"])
    return out
