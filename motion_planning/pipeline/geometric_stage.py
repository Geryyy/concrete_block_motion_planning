from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from motion_planning.core.scenario import from_worldmodel_scenario
from motion_planning.core.types import PlannerRequest, PlannerResult, Scenario
from motion_planning.io.optimized_params import canonical_method_name, load_optimized_planner_params
from motion_planning.planners.factory import create_planner


def run_geometric_planning(
    *,
    scenario: Scenario,
    method: str,
    config: Dict[str, Any],
    options: Dict[str, Any],
) -> PlannerResult:
    planner = create_planner(method)
    req = PlannerRequest(scenario=scenario, config=config, options=options)
    return planner.plan(req)


def run_geometric_planning_from_benchmark_params(
    *,
    world_scenario: Any,
    method: str,
    optimized_params_file: str | Path,
) -> PlannerResult:
    scenario = from_worldmodel_scenario(world_scenario)
    params = load_optimized_planner_params(optimized_params_file)
    canonical = canonical_method_name(method)
    if canonical not in params:
        raise KeyError(f"Method '{canonical}' not found in optimized params: {optimized_params_file}")
    entry = params[canonical]
    return run_geometric_planning(
        scenario=scenario,
        method=canonical,
        config=dict(entry["config"]),
        options=dict(entry["options"]),
    )
