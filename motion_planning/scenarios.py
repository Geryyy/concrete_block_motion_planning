from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from motion_planning.geometry import Scene


@dataclass(frozen=True)
class ScenarioConfig:
    scene: Scene
    start: Tuple[float, float, float]
    goal: Tuple[float, float, float]
    moving_block_size: Tuple[float, float, float]
    start_yaw_deg: float
    goal_yaw_deg: float
    goal_normals: Tuple[Tuple[float, float, float], ...]


DEFAULT_SCENARIOS_FILE = Path(__file__).with_name("data").joinpath("generated_scenarios.yaml")


class ScenarioLibrary:
    def __init__(self, scenarios_file: Path | str = DEFAULT_SCENARIOS_FILE):
        self.scenarios_file = Path(scenarios_file)
        self.payload = _load_yaml_payload(self.scenarios_file)

    def list_scenarios(self) -> List[str]:
        return sorted(self.payload["scenarios"].keys())

    def build_scenario(self, name: str) -> ScenarioConfig:
        scenarios = self.payload["scenarios"]
        key = str(name).strip().lower()
        if key not in scenarios:
            available = ", ".join(sorted(scenarios.keys()))
            raise ValueError(f"Unknown scenario '{name}'. Available: {available}")

        defaults = self.payload.get("defaults", {})
        base_size = tuple(float(v) for v in defaults.get("base_size", [0.6, 0.9, 0.6]))
        cfg = scenarios[key]

        scene = Scene()
        _load_blocks(scene, cfg.get("blocks", []), base_size)

        moving_cfg = cfg["moving_block"]
        moving_size = tuple(float(v) for v in moving_cfg.get("size", base_size))
        start = tuple(float(v) for v in moving_cfg["start"])
        start_yaw_deg = float(moving_cfg.get("start_yaw_deg", 0.0))
        goal_yaw_deg = float(moving_cfg.get("goal_yaw_deg", 0.0))

        goal = _resolve_goal(scene, moving_size, moving_cfg["goal"])
        goal_normals = _parse_goal_normals(moving_cfg=moving_cfg, scenario_name=key)

        return ScenarioConfig(
            scene=scene,
            start=start,
            goal=goal,
            moving_block_size=moving_size,
            start_yaw_deg=start_yaw_deg,
            goal_yaw_deg=goal_yaw_deg,
            goal_normals=goal_normals,
        )


def list_scenarios(scenarios_file: Path | str = DEFAULT_SCENARIOS_FILE) -> List[str]:
    return ScenarioLibrary(scenarios_file=scenarios_file).list_scenarios()


def build_scenario(name: str, scenarios_file: Path | str = DEFAULT_SCENARIOS_FILE) -> ScenarioConfig:
    return ScenarioLibrary(scenarios_file=scenarios_file).build_scenario(name)


# Backward compatibility for older imports.
WorldModel = ScenarioLibrary


def _load_yaml_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    if not isinstance(payload, dict) or "scenarios" not in payload:
        raise ValueError(f"Invalid scenarios YAML: {path}")
    if not isinstance(payload["scenarios"], dict):
        raise ValueError("'scenarios' must be a mapping")
    return payload


def _load_blocks(scene: Scene, blocks: List[Dict[str, Any]], base_size: Tuple[float, float, float]) -> None:
    for item in blocks:
        size = tuple(float(v) for v in item.get("size", base_size))
        position = tuple(float(v) for v in item["position"])
        quat = tuple(float(v) for v in item.get("quat", [0.0, 0.0, 0.0, 1.0]))
        object_id = str(item["id"])
        scene.add_block(size=size, position=position, quat=quat, object_id=object_id)


def _resolve_goal(scene: Scene, moving_size: Tuple[float, float, float], goal_cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    goal_type = str(goal_cfg.get("type", "point")).lower()

    if goal_type == "point":
        return tuple(float(v) for v in goal_cfg["position"])

    if goal_type == "face":
        base = goal_cfg["base"]
        face = str(goal_cfg["face"]).lower()
        gap = float(goal_cfg.get("gap", 0.0))
        tangential_offset = tuple(float(v) for v in goal_cfg.get("tangential_offset", [0.0, 0.0]))
        size_for_goal = tuple(float(v) for v in goal_cfg.get("size", moving_size))
        pos = scene.get_stack_point_on_face(
            base=base,
            new_size=size_for_goal,
            face=face,
            gap=gap,
            tangential_offset=tangential_offset,
        )
        return tuple(float(v) for v in pos.tolist())

    if goal_type == "between":
        ids = goal_cfg["ids"]
        p0 = np.asarray(scene.get_block(ids[0]).position, dtype=float)
        p1 = np.asarray(scene.get_block(ids[1]).position, dtype=float)
        mid = 0.5 * (p0 + p1)
        if "position" in goal_cfg:
            # Use provided coordinates with 'null' values meaning midpoint component.
            out = []
            provided = goal_cfg["position"]
            for i in range(3):
                out.append(float(mid[i]) if provided[i] is None else float(provided[i]))
            return tuple(out)
        return tuple(float(v) for v in mid.tolist())

    raise ValueError(f"Unknown goal type: {goal_type}")


def _parse_goal_normals(
    moving_cfg: Dict[str, Any],
    scenario_name: str,
) -> Tuple[Tuple[float, float, float], ...]:
    normals_raw = moving_cfg.get("goal_normals")
    if normals_raw is None:
        raise ValueError(
            f"Scenario '{scenario_name}' is missing moving_block.goal_normals. "
            "Provide one or more 3D vectors in YAML."
        )

    if not isinstance(normals_raw, list) or not normals_raw:
        raise ValueError(f"Scenario '{scenario_name}' has invalid moving_block.goal_normals (must be a non-empty list).")

    normals: List[Tuple[float, float, float]] = []
    for idx, raw in enumerate(normals_raw):
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise ValueError(
                f"Scenario '{scenario_name}' goal_normals[{idx}] is invalid; expected 3 values."
            )

        vec = np.asarray([float(v) for v in raw], dtype=float)
        mag = float(np.linalg.norm(vec))
        if mag < 1e-12:
            raise ValueError(f"Scenario '{scenario_name}' goal_normals[{idx}] must be non-zero.")

        unit = vec / mag
        normals.append((float(unit[0]), float(unit[1]), float(unit[2])))

    return tuple(normals)
