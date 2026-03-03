from __future__ import annotations

from typing import Any

from .types import Scenario


def from_worldmodel_scenario(cfg: Any) -> Scenario:
    """Convert WorldModel ScenarioConfig-like object to canonical Scenario."""
    return Scenario(
        scene=cfg.scene,
        start=tuple(float(v) for v in cfg.start),
        goal=tuple(float(v) for v in cfg.goal),
        moving_block_size=tuple(float(v) for v in cfg.moving_block_size),
        start_yaw_deg=float(cfg.start_yaw_deg),
        goal_yaw_deg=float(cfg.goal_yaw_deg),
        goal_normals=tuple(tuple(float(x) for x in n) for n in cfg.goal_normals),
    )
