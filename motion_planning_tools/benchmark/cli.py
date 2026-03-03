#!/usr/bin/env python3
"""Benchmark path planning strategies and run simple hyperparameter search.

Strategies:
- Powell (local spline optimizer)
- Nelder-Mead (simplex local optimizer)
- CEM (stochastic spline optimizer)
- VP-STO (via-point stochastic optimizer library)
- OMPL-RRT (sampling-based geometric planner)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ImportError:
    yaml = None

from motion_planning_tools.benchmark import SEED_OFFSETS, VALID_METHODS, benchmark_best, hyperopt
from motion_planning.scenarios import ScenarioLibrary


DEFAULT_OPTUNA_JOBS = max(1, int(os.cpu_count() or 1))


def _parse_methods(methods_raw: str) -> List[str]:
    methods = [m.strip() for m in methods_raw.split(",") if m.strip()]
    for m in methods:
        if m.upper() not in VALID_METHODS:
            raise ValueError(f"Unsupported method '{m}'. Use Powell, Nelder-Mead, CEM, VP-STO, and/or OMPL-RRT.")
    return methods


def _split_train_test(scenario_names: List[str]) -> tuple[List[str], List[str]]:
    if len(scenario_names) > 1:
        return scenario_names[:-1], scenario_names
    return scenario_names, scenario_names


def _load_params_yaml(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required for --params-in. Install with: pip install pyyaml")
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    methods = data.get("methods", {})
    return methods if isinstance(methods, dict) else {}


def _save_params_yaml(path: str, methods_payload: Dict[str, Any]) -> None:
    if not path:
        return
    if yaml is None:
        raise RuntimeError("PyYAML is required for --params-out. Install with: pip install pyyaml")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"methods": methods_payload}
    p.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark path planners and run hyperparameter search.")
    parser.add_argument(
        "--scenarios-file",
        default=str(Path(__file__).resolve().parents[1] / "data" / "generated_scenarios.yaml"),
        help="Path to scenarios YAML file.",
    )
    parser.add_argument(
        "--methods",
        default="Powell,Nelder-Mead,CEM,VP-STO,OMPL-RRT",
        help="Comma-separated methods to benchmark (supported: Powell,Nelder-Mead,CEM,VP-STO,OMPL-RRT).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Hyperparameter trials per method.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for hyperparameter search.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("benchmark_results.json")),
        help="Output JSON file.",
    )
    parser.add_argument(
        "--scenarios",
        default="",
        help="Optional comma-separated scenario names to evaluate (subset of YAML).",
    )
    parser.add_argument(
        "--params-in",
        default="",
        help="Optional YAML file with per-method best config/options for warm start.",
    )
    parser.add_argument(
        "--params-out",
        default=str(Path(__file__).with_name("optimized_params.yaml")),
        help="YAML file to write per-method best config/options.",
    )
    parser.add_argument(
        "--optuna-storage",
        default="",
        help="Optional Optuna storage URL (e.g. sqlite:///optuna_bench.db).",
    )
    parser.add_argument(
        "--optuna-jobs",
        type=int,
        default=DEFAULT_OPTUNA_JOBS,
        help=(
            f"Number of parallel Optuna workers per method (default: {DEFAULT_OPTUNA_JOBS}). "
            "Set 0 or negative to auto-use all CPU cores."
        ),
    )
    parser.add_argument(
        "--study-prefix",
        default="benchmark",
        help="Prefix for per-method Optuna study names.",
    )
    args = parser.parse_args()

    wm = ScenarioLibrary(scenarios_file=args.scenarios_file)
    all_scenarios = wm.list_scenarios()

    if args.scenarios.strip():
        wanted = [s.strip() for s in args.scenarios.split(",") if s.strip()]
        missing = [s for s in wanted if s not in all_scenarios]
        if missing:
            raise ValueError(f"Unknown scenario(s) in --scenarios: {', '.join(missing)}")
        all_scenarios = wanted
    if not all_scenarios:
        raise ValueError("No scenarios found for benchmark.")

    methods = _parse_methods(args.methods)
    train_scenarios, test_scenarios = _split_train_test(all_scenarios)
    warm_params = _load_params_yaml(args.params_in)

    hyperopt_results: Dict[str, Any] = {}
    benchmark_results: Dict[str, Any] = {}
    optimized_params: Dict[str, Any] = {}

    for method in methods:
        print(f"[hyperopt] method={method} trials={args.trials} train_scenarios={len(train_scenarios)}")
        warm_entry = warm_params.get(method) or warm_params.get(method.upper()) or None
        hres = hyperopt(
            wm=wm,
            train_scenarios=train_scenarios,
            method=method,
            n_trials=args.trials,
            seed=args.seed + SEED_OFFSETS.get(method.upper(), 0),
            warm_start=warm_entry,
            study_name=f"{args.study_prefix}_{method.lower().replace('-', '_')}",
            storage_url=(args.optuna_storage or None),
            n_jobs=int(args.optuna_jobs),
        )
        hyperopt_results[method] = hres

        best = hres["best"]
        optimized_params[method] = {"config": dict(best["config"]), "options": dict(best["options"])}
        print(
            f"[best] method={method} mean_score={best['mean_score']:.4f} "
            f"success_rate={best['success_rate']:.2f}"
        )

        bres = benchmark_best(wm, test_scenarios, best)
        benchmark_results[method] = bres
        agg = bres["aggregate"]
        print(
            f"[benchmark] method={method} mean_score={agg['mean_score']:.4f} "
            f"std={agg['std_score']:.4f} success_rate={agg['success_rate']:.2f}"
        )

    payload = {
        "scenarios_file": str(args.scenarios_file),
        "train_scenarios": train_scenarios,
        "test_scenarios": test_scenarios,
        "methods": methods,
        "trials_per_method": int(args.trials),
        "seed": int(args.seed),
        "hyperopt": hyperopt_results,
        "benchmark": benchmark_results,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote benchmark results to {out}")
    _save_params_yaml(args.params_out, optimized_params)
    if args.params_out:
        print(f"Wrote optimized parameters to {args.params_out}")


if __name__ == "__main__":
    main()
