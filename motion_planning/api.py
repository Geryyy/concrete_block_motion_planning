from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple, Union

from motion_planning.core.types import PlannerResult, Scenario
from motion_planning.core.world_model import WorldModel
from motion_planning.geometry.scene import Scene as GeometryScene
from motion_planning.io.optimized_params import canonical_method_name, load_optimized_planner_params
from motion_planning.pipeline.geometric_stage import run_geometric_planning


def _vec3(v: Sequence[float]) -> Tuple[float, float, float]:
    if len(v) != 3:
        raise ValueError("Expected 3 values for a 3D vector.")
    return (float(v[0]), float(v[1]), float(v[2]))


def _normals(values: Iterable[Sequence[float]] | None) -> Tuple[Tuple[float, float, float], ...]:
    if values is None:
        return ()
    out = []
    for n in values:
        out.append(_vec3(n))
    return tuple(out)


def _scene_from_input(world_model: Union[WorldModel, GeometryScene]) -> GeometryScene:
    if isinstance(world_model, WorldModel):
        return world_model.scene
    if isinstance(world_model, GeometryScene):
        return world_model
    raise TypeError(
        "world_model must be WorldModel or geometry.scene.Scene, "
        f"got {type(world_model)!r}"
    )


def plan(
    start: Sequence[float],
    end: Sequence[float],
    *,
    method: str = "Powell",
    world_model: Union[WorldModel, GeometryScene],
    moving_block_size: Sequence[float],
    start_yaw_deg: float = 0.0,
    goal_yaw_deg: float = 0.0,
    goal_normals: Iterable[Sequence[float]] | None = None,
    config: Mapping[str, Any] | None = None,
    options: Mapping[str, Any] | None = None,
    optimized_params_file: str | Path | None = None,
) -> PlannerResult:
    """Plan a geometric path in the current world model.

    This is the stable high-level API intended for online query use
    (e.g. ROS request handlers): ``plan(start, end, method=..., ...)``.
    """
    canonical = canonical_method_name(method)
    cfg: Dict[str, Any] = dict(config or {})
    opts: Dict[str, Any] = dict(options or {})

    if optimized_params_file is not None:
        entries = load_optimized_planner_params(optimized_params_file)
        if canonical not in entries:
            raise KeyError(f"Method '{canonical}' not found in optimized params: {optimized_params_file}")
        entry = entries[canonical]
        if not cfg:
            cfg = dict(entry["config"])
        if not opts:
            opts = dict(entry["options"])

    scenario = Scenario(
        scene=_scene_from_input(world_model),
        start=_vec3(start),
        goal=_vec3(end),
        moving_block_size=_vec3(moving_block_size),
        start_yaw_deg=float(start_yaw_deg),
        goal_yaw_deg=float(goal_yaw_deg),
        goal_normals=_normals(goal_normals),
    )
    return run_geometric_planning(
        scenario=scenario,
        method=canonical,
        config=cfg,
        options=opts,
    )
