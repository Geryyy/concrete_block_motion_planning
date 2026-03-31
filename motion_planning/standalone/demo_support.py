from __future__ import annotations

from typing import Any, Callable

import numpy as np


DEFAULT_PLANNER_CFG = {
    "goal_approach_window_fraction": 0.1,
    "contact_window_fraction": 0.1,
}
FALLBACK_DIAGNOSTICS = {
    "reference_path_fallback_used": 1.0,
    "joint_anchor_fallback_used": 0.0,
}
_STACK_NAME = "joint_space_global_path"


def make_straight_curve_sampler(start_xyz: np.ndarray, goal_xyz: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    start = np.asarray(start_xyz, dtype=float).reshape(3)
    goal = np.asarray(goal_xyz, dtype=float).reshape(3)

    def _sample(uq: np.ndarray) -> np.ndarray:
        u = np.asarray(uq, dtype=float).reshape(-1, 1)
        return (1.0 - u) * start.reshape(1, 3) + u * goal.reshape(1, 3)

    return _sample


def make_linear_yaw_fn(start_yaw_deg: float, goal_yaw_deg: float) -> Callable[[np.ndarray], np.ndarray]:
    def _yaw(uq: np.ndarray) -> np.ndarray:
        u = np.asarray(uq, dtype=float).reshape(-1)
        return np.asarray(start_yaw_deg + (goal_yaw_deg - start_yaw_deg) * u, dtype=float)

    return _yaw


def is_cbs_stack(method: str) -> bool:
    return method.lower().replace("-", "_") == _STACK_NAME


def plan_cbs_stack(method: str, demo_scenario_name: str) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, dict[str, Any]]:
    from motion_planning.standalone.scenarios import make_default_scenarios
    from motion_planning.standalone.stacks import STACK_REGISTRY

    stack_name = method.lower().replace("-", "_")
    if stack_name != _STACK_NAME:
        raise ValueError(f"Only '{_STACK_NAME}' is supported in the standalone demo")
    fn = STACK_REGISTRY[stack_name]
    scenarios = make_default_scenarios()
    scenario = next((sc for sc in scenarios.values() if sc.overlay_scene_name == demo_scenario_name), None)
    if scenario is None:
        raise ValueError(f"No standalone scenario found for '{demo_scenario_name}'")

    result = fn(scenario)
    if not result.success:
        raise RuntimeError(result.message)

    tcp_xyz = np.asarray(result.tcp_xyz, dtype=float).reshape(-1, 3)
    tcp_yaw_rad = np.asarray(result.tcp_yaw_rad, dtype=float).reshape(-1)
    t = np.linspace(0.0, 1.0, len(tcp_xyz))

    def curve_sampler(uq: np.ndarray) -> np.ndarray:
        u = np.asarray(uq, dtype=float).reshape(-1)
        return np.column_stack([np.interp(u, t, tcp_xyz[:, i]) for i in range(3)])

    def yaw_fn(uq: np.ndarray) -> np.ndarray:
        u = np.asarray(uq, dtype=float).reshape(-1)
        return np.interp(u, t, np.degrees(tcp_yaw_rad))

    return curve_sampler, np.asarray(result.diagnostics.get("via_tcp_xyz", np.empty((0, 3))), dtype=float), {
        "yaw_fn": yaw_fn,
        "success": bool(result.success),
        "message": str(result.message),
        "nit": int(result.diagnostics.get("optimizer_iterations", 0)),
        "preferred_clearance": 0.05,
        "diagnostics": dict(result.diagnostics),
        "joint_anchor_fallback_used": float(result.diagnostics.get("joint_anchor_fallback_used", 0.0)),
        "reference_path_fallback_used": float(result.diagnostics.get("reference_path_fallback_used", 0.0)),
    }
