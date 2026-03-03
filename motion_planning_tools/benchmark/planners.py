from __future__ import annotations

import time
from typing import Any, Dict
from pathlib import Path
import sys

import numpy as np

from motion_planning.geometry.spline_opt import yaw_deg_to_quat
try:
    from motion_planning.core.types import PlannerRequest, Scenario
    from motion_planning.planners.factory import create_planner
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from motion_planning.core.types import PlannerRequest, Scenario
    from motion_planning.planners.factory import create_planner

from .metrics import evaluate_path_metrics, make_eval_context, scenario_score


METHOD_ALIASES = {
    "POWELL": "POWELL",
    "NELDER-MEAD": "NELDER-MEAD",
    "NELDER_MEAD": "NELDER-MEAD",
    "NELDERMEAD": "NELDER-MEAD",
    "NELDER": "NELDER-MEAD",
    "NM": "NELDER-MEAD",
    "CEM": "CEM",
    "VP-STO": "VP-STO",
    "OMPL-RRT": "OMPL-RRT",
    "OMPL": "OMPL-RRT",
    "RRT": "OMPL-RRT",
}


def _import_vpsto():
    from vpsto.vpsto import VPSTO, VPSTOOptions  # type: ignore

    return VPSTO, VPSTOOptions


def _import_ompl():
    import ompl.base as ob  # type: ignore
    import ompl.geometric as og  # type: ignore

    return ob, og


def _make_base_row(
    *,
    scenario_name: str,
    P: np.ndarray,
    metrics: Dict[str, Any],
    message: str,
    nit: int,
    start: np.ndarray,
    goal: np.ndarray,
) -> Dict[str, Any]:
    straight_len = float(np.linalg.norm(np.asarray(goal, dtype=float) - np.asarray(start, dtype=float)))
    return {
        "scenario": scenario_name,
        "success": bool(metrics["success"]),
        "fun": float(metrics["fun"]),
        "length": float(metrics["length"]),
        "path_efficiency": float(metrics["length"]) / max(straight_len, 1e-9),
        "curvature_cost": float(metrics["curvature_cost"]),
        "turn_angle_mean_deg": float(metrics["turn_angle_mean_deg"]),
        "yaw_smoothness_cost": float(metrics["yaw_smoothness_cost"]),
        "safety_cost": float(metrics["safety_cost"]),
        "preferred_safety_cost": float(metrics["preferred_safety_cost"]),
        "approach_rebound_cost": float(metrics["approach_rebound_cost"]),
        "goal_clearance_cost": float(metrics["goal_clearance_cost"]),
        "goal_clearance_target_cost": float(metrics["goal_clearance_target_cost"]),
        "approach_clearance_cost": float(metrics["approach_clearance_cost"]),
        "approach_collision_cost": float(metrics["approach_collision_cost"]),
        "goal_approach_normal_cost": float(metrics["goal_approach_normal_cost"]),
        "min_clearance": float(metrics["min_clearance"]),
        "mean_clearance": float(metrics["mean_clearance"]),
        "min_clearance_raw": float(metrics.get("min_clearance_raw", metrics["min_clearance"])),
        "mean_clearance_raw": float(metrics.get("mean_clearance_raw", metrics["mean_clearance"])),
        "nit": int(nit),
        "message": str(message),
        "trajectory_xyz": np.asarray(P, dtype=float).tolist(),
    }


def _run_spline_method(sc, scenario_name: str, method: str, config: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    planner = create_planner(method)
    req = PlannerRequest(
        scenario=Scenario(
            scene=sc.scene,
            start=tuple(float(v) for v in sc.start),
            goal=tuple(float(v) for v in sc.goal),
            moving_block_size=tuple(float(v) for v in sc.moving_block_size),
            start_yaw_deg=float(sc.start_yaw_deg),
            goal_yaw_deg=float(sc.goal_yaw_deg),
            goal_normals=tuple(tuple(float(x) for x in n) for n in sc.goal_normals),
        ),
        config=dict(config),
        options=dict(options),
    )
    p_res = planner.plan(req)

    ctx = make_eval_context(
        scene=sc.scene,
        goal=sc.goal,
        moving_block_size=sc.moving_block_size,
        start_yaw_deg=sc.start_yaw_deg,
        goal_yaw_deg=sc.goal_yaw_deg,
        goal_normals=np.asarray(sc.goal_normals, dtype=float),
        config=config,
    )
    n_curve = int(config.get("n_samples_curve", 101))
    P = p_res.path.sample(n_curve)
    if p_res.path.yaw_fn is not None:
        yaw_samples = p_res.path.sample_yaw(n_curve)
    else:
        yaw_samples = np.linspace(float(sc.start_yaw_deg), float(sc.goal_yaw_deg), n_curve)
    metrics = evaluate_path_metrics(
        ctx=ctx,
        P=P,
        message=str(p_res.message),
        nit=int(p_res.diagnostics.get("nit", 0)),
        yaw_samples_deg=yaw_samples,
        solver_success=bool(p_res.success),
    )
    return _make_base_row(
        scenario_name=scenario_name,
        P=P,
        metrics=metrics,
        message=str(p_res.message),
        nit=int(p_res.diagnostics.get("nit", 0)),
        start=np.asarray(sc.start, dtype=float),
        goal=np.asarray(sc.goal, dtype=float),
    )


def _run_vpsto_method(sc, scenario_name: str, config: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    VPSTO, VPSTOOptions = _import_vpsto()

    vp_opt = VPSTOOptions(ndof=3)
    vp_opt.N_eval = int(options.get("N_eval", config.get("n_samples_curve", 101)))
    vp_opt.N_via = int(options.get("N_via", 6))
    vp_opt.pop_size = int(options.get("pop_size", 32))
    vp_opt.sigma_init = float(options.get("sigma_init", 0.4))
    vp_opt.max_iter = int(options.get("max_iter", 180))
    vp_opt.CMA_diagonal = bool(options.get("CMA_diagonal", False))
    vp_opt.verbose = False
    vp_opt.log = False

    planner = VPSTO(vp_opt)
    start = np.asarray(sc.start, dtype=float)
    goal = np.asarray(sc.goal, dtype=float)
    ctx = make_eval_context(
        scene=sc.scene,
        goal=goal,
        moving_block_size=sc.moving_block_size,
        start_yaw_deg=sc.start_yaw_deg,
        goal_yaw_deg=sc.goal_yaw_deg,
        goal_normals=np.asarray(sc.goal_normals, dtype=float),
        config=config,
    )

    def loss(candidates: Dict[str, np.ndarray]) -> np.ndarray:
        qs = np.asarray(candidates["pos"], dtype=float)
        costs = np.empty((qs.shape[0],), dtype=float)
        for i in range(qs.shape[0]):
            info = evaluate_path_metrics(
                ctx=ctx,
                P=qs[i],
                message="VP-STO",
                nit=0,
            )
            costs[i] = float(info["fun"])
        return costs

    sol = planner.minimize(loss, q0=start, qT=goal, dq0=np.zeros(3), dqT=np.zeros(3), T=None)
    ts = np.linspace(0.0, float(sol.T_best), int(config.get("n_samples_curve", 101)))
    P, _, _ = sol.get_posvelacc(ts)
    P = np.asarray(P, dtype=float)

    info = evaluate_path_metrics(
        ctx=ctx,
        P=P,
        message="VP-STO",
        nit=int(options.get("max_iter", 180)),
        solver_success=True,
    )
    return _make_base_row(
        scenario_name=scenario_name,
        P=P,
        metrics=info,
        message=str(info["message"]),
        nit=int(info["nit"]),
        start=start,
        goal=goal,
    )


def _run_ompl_rrt_method(sc, scenario_name: str, config: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    ob, og = _import_ompl()

    start = np.asarray(sc.start, dtype=float)
    goal = np.asarray(sc.goal, dtype=float)
    block_positions = np.asarray([np.asarray(b.position, dtype=float) for b in sc.scene.blocks], dtype=float)
    all_pts = np.vstack([block_positions, start[None, :], goal[None, :]])
    bmin = np.min(all_pts, axis=0)
    bmax = np.max(all_pts, axis=0)
    margin = float(options.get("bounds_margin", 0.5))

    space = ob.RealVectorStateSpace(3)
    bounds = ob.RealVectorBounds(3)
    for i in range(3):
        bounds.setLow(i, float(bmin[i] - margin))
        bounds.setHigh(i, float(bmax[i] + margin))
    space.setBounds(bounds)

    si = ob.SpaceInformation(space)

    fixed_quat = yaw_deg_to_quat(float(sc.start_yaw_deg))

    def is_valid(state) -> bool:
        p = np.array([state[i] for i in range(3)], dtype=float)
        d = sc.scene.signed_distance_block(size=sc.moving_block_size, position=p, quat=fixed_quat)
        return bool(d >= -1e-4)

    si.setStateValidityChecker(ob.StateValidityCheckerFn(is_valid))
    si.setup()

    pdef = ob.ProblemDefinition(si)
    s0 = ob.State(space)
    s1 = ob.State(space)
    for i in range(3):
        s0[i] = float(start[i])
        s1[i] = float(goal[i])
    pdef.setStartAndGoalStates(s0, s1, 0.06)

    planner = og.RRT(si)
    if "range" in options:
        planner.setRange(float(options["range"]))
    planner.setProblemDefinition(pdef)
    planner.setup()

    solved = planner.solve(float(options.get("solve_time", 0.8)))
    if not solved:
        raise RuntimeError("OMPL-RRT failed to find a solution")

    path = pdef.getSolutionPath()
    interp_n = int(max(path.getStateCount(), int(options.get("interpolate_points", config.get("n_samples_curve", 101)))))
    path.interpolate(interp_n)

    P = np.array([[path.getState(i)[j] for j in range(3)] for i in range(path.getStateCount())], dtype=float)
    ctx = make_eval_context(
        scene=sc.scene,
        goal=goal,
        moving_block_size=sc.moving_block_size,
        start_yaw_deg=sc.start_yaw_deg,
        goal_yaw_deg=sc.goal_yaw_deg,
        goal_normals=np.asarray(sc.goal_normals, dtype=float),
        config=config,
    )
    info = evaluate_path_metrics(
        ctx=ctx,
        P=P,
        message="OMPL-RRT",
        nit=0,
        solver_success=True,
    )
    return _make_base_row(
        scenario_name=scenario_name,
        P=P,
        metrics=info,
        message=str(info["message"]),
        nit=int(info["nit"]),
        start=start,
        goal=goal,
    )


def run_single(
    wm: Any,
    scenario_name: str,
    method: str,
    config: Dict[str, Any],
    options: Dict[str, Any],
) -> Dict[str, Any]:
    sc = wm.build_scenario(scenario_name)
    t0 = time.perf_counter()

    canonical = METHOD_ALIASES.get(method.upper())
    if canonical is None:
        raise ValueError(f"Unsupported method: {method}")

    if canonical in {"POWELL", "NELDER-MEAD", "CEM"}:
        row = _run_spline_method(sc, scenario_name, method=canonical.title() if canonical != "CEM" else "CEM", config=config, options=options)
    elif canonical == "VP-STO":
        row = _run_vpsto_method(sc, scenario_name, config=config, options=options)
    else:
        row = _run_ompl_rrt_method(sc, scenario_name, config=config, options=options)

    dt = float(time.perf_counter() - t0)
    row["runtime_s"] = dt
    row["score"] = scenario_score(row, dt)
    return row
