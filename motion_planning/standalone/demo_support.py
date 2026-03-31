from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from motion_planning.core.types import PlannerRequest, Scenario
from motion_planning.io.optimized_params import load_optimized_planner_params
from motion_planning.planners.factory import create_planner
from motion_planning_tools.benchmark.metrics import (
    evaluate_path_metrics,
    make_eval_context,
)


DEFAULT_PLANNER_CFG = {
    "goal_approach_window_fraction": 0.1,
    "contact_window_fraction": 0.1,
}
FALLBACK_DIAGNOSTICS = {
    "reference_path_fallback_used": 1.0,
    "joint_anchor_fallback_used": 0.0,
}
_CBS_STACK_NAMES = {"vpsto_path_planning", "vpsto_ilqr"}


def make_straight_curve_sampler(
    start_xyz: np.ndarray,
    goal_xyz: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    start = np.asarray(start_xyz, dtype=float).reshape(3)
    goal = np.asarray(goal_xyz, dtype=float).reshape(3)

    def _sample(uq: np.ndarray) -> np.ndarray:
        u = np.asarray(uq, dtype=float).reshape(-1, 1)
        return (1.0 - u) * start.reshape(1, 3) + u * goal.reshape(1, 3)

    return _sample


def make_curve_sampler(curve: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    curve_arr = np.asarray(curve, dtype=float)
    t = np.linspace(0.0, 1.0, curve_arr.shape[0])

    def _sample(uq: np.ndarray) -> np.ndarray:
        u = np.asarray(uq, dtype=float).reshape(-1)
        return np.column_stack([np.interp(u, t, curve_arr[:, i]) for i in range(3)])

    return _sample


def make_linear_yaw_fn(
    start_yaw_deg: float,
    goal_yaw_deg: float,
) -> Callable[[np.ndarray], np.ndarray]:
    def _yaw(uq: np.ndarray) -> np.ndarray:
        u = np.asarray(uq, dtype=float).reshape(-1)
        return np.asarray(start_yaw_deg + (goal_yaw_deg - start_yaw_deg) * u, dtype=float)

    return _yaw


def is_cbs_stack(method: str) -> bool:
    return method.lower().replace("-", "_") in _CBS_STACK_NAMES


def planner_entry(
    method: str,
    optimized_params_file: Path,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    method_raw = str(method).strip()
    method_upper = method_raw.upper()
    if method_upper in {"NELDER", "NELDER_MEAD", "NELDERMEAD", "NM"}:
        canonical = "Nelder-Mead"
    elif method_upper == "POWELL":
        canonical = "Powell"
    elif method_upper == "CEM":
        canonical = "CEM"
    elif method_upper in {"VP-STO", "VPSTO"}:
        canonical = "VP-STO"
    elif method_upper in {"OMPL-RRT", "OMPL", "RRT"}:
        canonical = "OMPL-RRT"
    else:
        raise ValueError("Unsupported --planner. Use Powell, CEM, Nelder-Mead, VP-STO, or OMPL-RRT.")

    entries = load_optimized_planner_params(optimized_params_file)
    entry = entries.get(canonical)
    if entry is None:
        raise KeyError(f"Method '{canonical}' not found in optimized params: {optimized_params_file}")
    return canonical, dict(entry["config"]), dict(entry["options"])


def plan_cbs_stack(
    method: str,
    demo_scenario_name: str,
) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, dict[str, Any]]:
    from motion_planning.standalone.scenarios import make_default_scenarios
    from motion_planning.standalone.stacks import STACK_REGISTRY

    stack_name = method.lower().replace("-", "_")
    fn = STACK_REGISTRY.get(stack_name)
    if fn is None:
        raise KeyError(f"CBS stack '{stack_name}' not in STACK_REGISTRY")

    cbs_scenarios = make_default_scenarios()
    cbs_scenario = next(
        (
            sc for sc in cbs_scenarios.values()
            if getattr(sc, "overlay_scene_name", None) == demo_scenario_name
        ),
        None,
    )
    if cbs_scenario is None:
        raise ValueError(
            f"No CBS scenario found for '{demo_scenario_name}'. "
            f"Available: {list(cbs_scenarios.keys())}"
        )

    print(f"Running CBS stack '{stack_name}' on scenario '{cbs_scenario.name}'")
    result = fn(cbs_scenario)
    if not result.success:
        raise RuntimeError(f"CBS stack '{stack_name}' failed: {result.message}")

    print(f"  {result.message}")
    if result.evaluation:
        ev = result.evaluation
        print(
            f"  pos err: {ev.final_position_error_m * 100:.1f} cm  "
            f"path len: {ev.path_length_m:.2f} m"
        )

    tcp_xyz = np.asarray(result.tcp_xyz, dtype=float).reshape(-1, 3)
    tcp_yaw_rad = np.asarray(result.tcp_yaw_rad, dtype=float).ravel()
    t = np.linspace(0.0, 1.0, tcp_xyz.shape[0])

    def curve_sampler(uq: np.ndarray) -> np.ndarray:
        u = np.asarray(uq, dtype=float).reshape(-1)
        return np.column_stack([np.interp(u, t, tcp_xyz[:, i]) for i in range(3)])

    def yaw_fn(uq: np.ndarray) -> np.ndarray:
        u = np.asarray(uq, dtype=float).reshape(-1)
        return np.interp(u, t, np.degrees(tcp_yaw_rad))

    info: dict[str, Any] = {
        "yaw_fn": yaw_fn,
        "success": result.success,
        "message": result.message,
        "nit": int(result.diagnostics.get("ilqr_iterations", result.diagnostics.get("vpsto_iterations", 0))),
        "preferred_clearance": 0.05,
        "joint_anchor_fallback_used": float(result.diagnostics.get("joint_anchor_fallback_used", 0.0)),
        "reference_path_fallback_used": float(result.diagnostics.get("reference_path_fallback_used", 0.0)),
    }
    return curve_sampler, np.empty((0, 3), dtype=float), info


def plan_spline_method(
    planner_method: str,
    planner_cfg: dict[str, Any],
    planner_options: dict[str, Any],
    scenario: Any,
) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, dict[str, Any]]:
    req = PlannerRequest(
        scenario=Scenario(
            scene=scenario.scene,
            start=tuple(float(v) for v in scenario.start),
            goal=tuple(float(v) for v in scenario.goal),
            moving_block_size=tuple(float(v) for v in scenario.moving_block_size),
            start_yaw_deg=float(scenario.start_yaw_deg),
            goal_yaw_deg=float(scenario.goal_yaw_deg),
            goal_normals=tuple(tuple(float(x) for x in n) for n in scenario.goal_normals),
        ),
        config=dict(planner_cfg),
        options=dict(planner_options),
    )
    p_res = create_planner(planner_method).plan(req)
    info = dict(p_res.metrics)
    info["success"] = bool(p_res.success)
    info["message"] = str(p_res.message)
    info["nit"] = int(p_res.diagnostics.get("nit", 0))
    info["diagnostics"] = dict(p_res.diagnostics)
    info["joint_anchor_fallback_used"] = float(p_res.diagnostics.get("joint_anchor_fallback_used", 0.0))
    info["reference_path_fallback_used"] = float(p_res.diagnostics.get("reference_path_fallback_used", 0.0))
    if p_res.path.yaw_fn is not None:
        info["yaw_fn"] = p_res.path.yaw_fn
    return p_res.path.xyz_fn, np.empty((0, 3), dtype=float), info


def import_vpsto():
    from vpsto.vpsto import VPSTO, VPSTOOptions  # type: ignore

    return VPSTO, VPSTOOptions


def import_ompl():
    import ompl.base as ob  # type: ignore
    import ompl.geometric as og  # type: ignore

    return ob, og


def plan_vpsto(
    scenario: Any,
    planner_cfg: dict[str, Any],
    planner_options: dict[str, Any],
) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, dict[str, Any]]:
    VPSTO, VPSTOOptions = import_vpsto()
    vp_opt = VPSTOOptions(ndof=3)
    vp_opt.N_eval = int(planner_options.get("N_eval", planner_cfg.get("n_samples_curve", 101)))
    vp_opt.N_via = int(planner_options.get("N_via", 6))
    vp_opt.pop_size = int(planner_options.get("pop_size", 32))
    vp_opt.sigma_init = float(planner_options.get("sigma_init", 0.4))
    vp_opt.max_iter = int(planner_options.get("max_iter", 180))
    vp_opt.CMA_diagonal = bool(planner_options.get("CMA_diagonal", False))
    vp_opt.verbose = False
    vp_opt.log = False
    planner = VPSTO(vp_opt)

    start_arr = np.asarray(scenario.start, dtype=float)
    goal_arr = np.asarray(scenario.goal, dtype=float)
    eval_ctx = make_eval_context(
        scene=scenario.scene,
        goal=goal_arr,
        moving_block_size=scenario.moving_block_size,
        start_yaw_deg=scenario.start_yaw_deg,
        goal_yaw_deg=scenario.goal_yaw_deg,
        goal_normals=np.asarray(scenario.goal_normals, dtype=float),
        config=planner_cfg,
    )

    def loss(candidates: dict[str, np.ndarray]) -> np.ndarray:
        qs = np.asarray(candidates["pos"], dtype=float)
        vals = np.empty((qs.shape[0],), dtype=float)
        for i in range(qs.shape[0]):
            info = evaluate_path_metrics(
                ctx=eval_ctx,
                P=qs[i],
                message="VP-STO",
                nit=0,
            )
            vals[i] = float(info["fun"])
        return vals

    sol = planner.minimize(loss, q0=start_arr, qT=goal_arr, dq0=np.zeros(3), dqT=np.zeros(3), T=None)
    ts = np.linspace(0.0, float(sol.T_best), int(planner_cfg.get("n_samples_curve", 101)))
    curve, _, _ = sol.get_posvelacc(ts)
    curve = np.asarray(curve, dtype=float)
    info = evaluate_path_metrics(
        ctx=eval_ctx,
        P=curve,
        message="VP-STO",
        nit=int(planner_options.get("max_iter", 180)),
        solver_success=True,
    )
    info["joint_anchor_fallback_used"] = 0.0
    info["reference_path_fallback_used"] = 0.0
    return make_curve_sampler(curve), np.empty((0, 3), dtype=float), info


def plan_rrt(
    scenario: Any,
    planner_cfg: dict[str, Any],
    planner_options: dict[str, Any],
    yaw_deg_to_quat: Callable[[float], tuple[float, float, float, float]],
) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, dict[str, Any]]:
    ob, og = import_ompl()
    start_arr = np.asarray(scenario.start, dtype=float)
    goal_arr = np.asarray(scenario.goal, dtype=float)
    block_positions = np.asarray([np.asarray(b.position, dtype=float) for b in scenario.scene.blocks], dtype=float)
    all_pts = np.vstack([block_positions, start_arr[None, :], goal_arr[None, :]])
    bmin = np.min(all_pts, axis=0)
    bmax = np.max(all_pts, axis=0)
    margin = float(planner_options.get("bounds_margin", 0.5))

    space = ob.RealVectorStateSpace(3)
    bounds = ob.RealVectorBounds(3)
    for i in range(3):
        bounds.setLow(i, float(bmin[i] - margin))
        bounds.setHigh(i, float(bmax[i] + margin))
    space.setBounds(bounds)

    si = ob.SpaceInformation(space)
    fixed_quat = yaw_deg_to_quat(float(scenario.start_yaw_deg))

    def is_valid(state) -> bool:
        p = np.array([state[i] for i in range(3)], dtype=float)
        d = scenario.scene.signed_distance_block(size=scenario.moving_block_size, position=p, quat=fixed_quat)
        return bool(d >= -1e-4)

    si.setStateValidityChecker(ob.StateValidityCheckerFn(is_valid))
    si.setup()

    pdef = ob.ProblemDefinition(si)
    s0 = ob.State(space)
    s1 = ob.State(space)
    for i in range(3):
        s0[i] = float(start_arr[i])
        s1[i] = float(goal_arr[i])
    pdef.setStartAndGoalStates(s0, s1, 0.06)

    planner = og.RRT(si)
    if "range" in planner_options:
        planner.setRange(float(planner_options["range"]))
    planner.setProblemDefinition(pdef)
    planner.setup()

    solved = planner.solve(float(planner_options.get("solve_time", 0.8)))
    if not solved:
        raise RuntimeError("OMPL-RRT failed to find a solution")

    path = pdef.getSolutionPath()
    path.interpolate(int(max(path.getStateCount(), int(planner_options.get("interpolate_points", 101)))))
    curve = np.array([[path.getState(i)[j] for j in range(3)] for i in range(path.getStateCount())], dtype=float)

    eval_ctx = make_eval_context(
        scene=scenario.scene,
        goal=goal_arr,
        moving_block_size=scenario.moving_block_size,
        start_yaw_deg=scenario.start_yaw_deg,
        goal_yaw_deg=scenario.goal_yaw_deg,
        goal_normals=np.asarray(scenario.goal_normals, dtype=float),
        config=planner_cfg,
    )
    info = evaluate_path_metrics(
        ctx=eval_ctx,
        P=curve,
        message="OMPL-RRT",
        nit=0,
        solver_success=True,
    )
    info["joint_anchor_fallback_used"] = 0.0
    info["reference_path_fallback_used"] = 0.0
    return make_curve_sampler(curve), np.empty((0, 3), dtype=float), info
