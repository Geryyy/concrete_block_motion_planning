#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _resolve_default_cases_file() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("concrete_block_motion_planning"))
        return share / "motion_planning" / "data" / "acados_bench_cases.yaml"
    except Exception:
        repo_guess = Path(__file__).resolve().parents[1] / "motion_planning" / "data" / "acados_bench_cases.yaml"
        return repo_guess


def _acados_ready() -> tuple[bool, str]:
    try:
        import acados_template  # noqa: F401
        import casadi  # noqa: F401
        import pinocchio  # noqa: F401
    except Exception as exc:
        return False, f"missing runtime dependency: {exc}"
    src = os.environ.get("ACADOS_SOURCE_DIR", "")
    if not src:
        return False, "ACADOS_SOURCE_DIR is not set"
    src_path = Path(src)
    if not (src_path / "lib" / "link_libs.json").exists():
        return False, f"missing acados link_libs.json under {src_path / 'lib'}"
    if not (src_path / "bin" / "t_renderer").exists():
        return False, f"missing acados t_renderer under {src_path / 'bin'}"
    return True, "ok"


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text())
    cases = list(payload.get("cases", []))
    if not cases:
      raise ValueError(f"no benchmark cases found in {path}")
    return cases


def _run_case(case: dict[str, Any], output_root: Path) -> dict[str, Any]:
    from motion_planning.types import TrajectoryRequest
    from motion_planning.mechanics import create_crane_config
    from motion_planning.trajectory.cartesian_path_following import (
        CartesianPathFollowingConfig,
        CartesianPathFollowingOptimizer,
    )

    q0 = np.asarray(case["q0"], dtype=float)
    q_goal = np.asarray(case["q_goal"], dtype=float)
    if q0.shape != q_goal.shape:
        raise ValueError(f"case '{case['name']}' has incompatible q0/q_goal shapes")

    crane_cfg = create_crane_config()
    codegen_dir = output_root / case["name"]
    codegen_dir.mkdir(parents=True, exist_ok=True)
    solver_cfg = CartesianPathFollowingConfig(
        urdf_path=Path(crane_cfg.urdf_path),
        horizon_steps=int(case.get("horizon_steps", 80)),
        optimize_time=bool(case.get("optimize_time", False)),
        fixed_time_duration_s=float(case.get("duration_s", 10.0)),
        fixed_time_duration_candidates=(float(case.get("duration_s", 10.0)),),
        code_export_dir=codegen_dir,
        solver_json_name=f"{case['name']}_ocp.json",
        precompile_on_init=False,
    )
    optimizer = CartesianPathFollowingOptimizer(solver_cfg)
    req = TrajectoryRequest(
        scenario=None,
        path=None,
        config={
            "q0": q0,
            "q_goal": q_goal,
            "dq0": np.zeros_like(q0),
            "optimize_time": bool(case.get("optimize_time", False)),
            "fixed_time_duration_s": float(case.get("duration_s", 10.0)),
            "fixed_time_duration_candidates": (float(case.get("duration_s", 10.0)),),
            "T_min": float(case.get("duration_s", 10.0)),
            "T_max": float(case.get("duration_s", 10.0)),
            "nlp_solver_max_iter": int(case.get("nlp_solver_max_iter", 300)),
            "qp_solver_iter_max": int(case.get("qp_solver_iter_max", 120)),
        },
    )
    t0 = time.perf_counter()
    result = optimizer.optimize(req)
    dt = time.perf_counter() - t0

    final_q_error = None
    if result.state.size > 0:
        q_terminal = np.asarray(result.state[-1, : q_goal.shape[0]], dtype=float)
        final_q_error = float(np.linalg.norm(q_terminal - q_goal))

    return {
        "name": case["name"],
        "success": bool(result.success),
        "message": result.message,
        "solve_time_s": dt,
        "reported_cost": None if result.cost is None else float(result.cost),
        "terminal_q_error_l2": final_q_error,
        "diagnostics_status": result.diagnostics.get("status"),
        "diagnostics_keys": sorted(result.diagnostics.keys()),
        "case": case,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run standalone acados trajectory benchmark cases.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=_resolve_default_cases_file(),
        help="YAML file containing benchmark cases.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for JSON benchmark results.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "concrete_block_acados_bench",
        help="Directory for per-case acados codegen artifacts.",
    )
    args = parser.parse_args()

    ready, reason = _acados_ready()
    if not ready:
        print(json.dumps({"success": False, "message": reason}, indent=2))
        return 2

    cases = _load_cases(args.cases)
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = [_run_case(case, args.output_root) for case in cases]
    payload = {
        "success": True,
        "cases_file": str(args.cases),
        "results": results,
    }

    text = json.dumps(payload, indent=2)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
