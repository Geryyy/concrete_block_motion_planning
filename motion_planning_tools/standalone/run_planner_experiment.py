#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import copy


def _repo_pythonpath() -> tuple[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    pkg_root = repo_root
    scripts_root = repo_root / "scripts"
    return str(pkg_root), str(scripts_root)


pkg_root, scripts_root = _repo_pythonpath()
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)
if scripts_root not in sys.path:
    sys.path.insert(0, scripts_root)

from motion_planning.standalone.compare_solvers import compare_solver_suite
from motion_planning.standalone.plotting import plot_plan_result, plot_solver_results
from motion_planning.standalone.scenarios import make_default_scenarios
from motion_planning.standalone.stacks import STACK_REGISTRY, apply_simple_time_scaling
from motion_planning.scenarios import ScenarioLibrary


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone planner/sovler experiment runner.")
    parser.add_argument("--scenario", default="single_block_transfer", help="Scenario name.")
    parser.add_argument(
        "--stack",
        default="cartesian_anchor_joint_spline",
        choices=sorted(STACK_REGISTRY.keys()),
        help="Planner stack to run.",
    )
    parser.add_argument(
        "--mode",
        default="planner",
        choices=["planner", "solver_compare"],
        help="Run a planner stack or a solver comparison.",
    )
    parser.add_argument(
        "--timing",
        default="none",
        choices=["none", "simple"],
        help="Optional standalone timing backend applied after path planning.",
    )
    parser.add_argument(
        "--overlay-scene",
        default="",
        help="Optional existing scene name from motion_planning.scenarios for 3D block overlay.",
    )
    parser.add_argument("--plot", action="store_true", help="Show matplotlib plots.")
    args = parser.parse_args()

    scenarios = make_default_scenarios()
    if args.scenario not in scenarios:
        raise SystemExit(f"Unknown scenario '{args.scenario}'. Available: {', '.join(sorted(scenarios))}")
    scenario = scenarios[args.scenario]

    if args.mode == "solver_compare":
        results = compare_solver_suite(scenario)
        payload = [
            {
                "name": item.name,
                "success": item.success,
                "message": item.message,
                "position_error_m": item.position_error_m,
                "yaw_error_deg": item.yaw_error_deg,
                "fk_xyz": item.fk_xyz.tolist(),
                "fk_yaw_rad": item.fk_yaw_rad,
                "metadata": item.metadata,
            }
            for item in results
        ]
        print(json.dumps(payload, indent=2))
        if args.plot:
            plot_solver_results(results)
        return

    result = STACK_REGISTRY[args.stack](scenario)
    if result.success and args.timing == "simple":
        apply_simple_time_scaling(result)
    payload = {
        "stack_name": result.stack_name,
        "success": result.success,
        "message": result.message,
        "diagnostics": result.diagnostics,
        "evaluation": None if result.evaluation is None else {
            "final_position_error_m": result.evaluation.final_position_error_m,
            "final_yaw_error_deg": result.evaluation.final_yaw_error_deg,
            "max_position_error_m": result.evaluation.max_position_error_m,
            "mean_position_error_m": result.evaluation.mean_position_error_m,
            "max_path_deviation_m": result.evaluation.max_path_deviation_m,
            "path_length_m": result.evaluation.path_length_m,
        },
        "timing": None if result.time_s is None else {
            "duration_s": float(result.time_s[-1]),
            "num_samples": int(result.time_s.shape[0]),
        },
    }
    print(json.dumps(payload, indent=2))
    if args.plot:
        overlay_scene = None
        overlay_scene_name = None
        overlay_name = args.overlay_scene or scenario.overlay_scene_name or ""
        if overlay_name:
            sc = ScenarioLibrary().build_scenario(overlay_name)
            overlay_scene = copy.deepcopy(sc.scene)
            if scenario.overlay_scene_translation is not None:
                dx, dy, dz = (float(v) for v in scenario.overlay_scene_translation)
                for block in overlay_scene.blocks:
                    px, py, pz = block.position
                    block.position = (px + dx, py + dy, pz + dz)
            overlay_scene_name = overlay_name
        plot_plan_result(result, scene=overlay_scene, scene_name=overlay_scene_name)


if __name__ == "__main__":
    main()
