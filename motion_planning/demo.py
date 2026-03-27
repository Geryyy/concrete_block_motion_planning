from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from motion_planning.core.types import PlannerRequest, Scenario
from motion_planning.geometry import plot_scene
from motion_planning.geometry.spline_opt import yaw_deg_to_quat
from motion_planning.geometry.utils import quat_to_rot
from motion_planning.io.optimized_params import load_optimized_planner_params
from motion_planning.planners.factory import create_planner
from motion_planning.scenarios import ScenarioLibrary
from motion_planning_tools.benchmark.metrics import (
    evaluate_path_metrics,
    evaluated_clearance_subset,
    make_eval_context,
)


DEFAULT_SCENARIOS_FILE = Path(__file__).with_name("data").joinpath("generated_scenarios.yaml")
DEFAULT_OPTIMIZED_PARAMS_FILE = Path(__file__).with_name("data").joinpath("optimized_params.yaml")
_LAST_ANIMATION: FuncAnimation | None = None


def _import_vpsto():
    from vpsto.vpsto import VPSTO, VPSTOOptions  # type: ignore

    return VPSTO, VPSTOOptions


def _import_ompl():
    import ompl.base as ob  # type: ignore
    import ompl.geometric as og  # type: ignore

    return ob, og


def _make_curve_sampler(curve: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    curve = np.asarray(curve, dtype=float)
    t = np.linspace(0.0, 1.0, curve.shape[0])

    def _sample(uq: np.ndarray) -> np.ndarray:
        u = np.asarray(uq, dtype=float).reshape(-1)
        x = np.interp(u, t, curve[:, 0])
        y = np.interp(u, t, curve[:, 1])
        z = np.interp(u, t, curve[:, 2])
        return np.column_stack([x, y, z])

    return _sample


def _make_linear_yaw_fn(start_yaw_deg: float, goal_yaw_deg: float) -> Callable[[np.ndarray], np.ndarray]:
    def _yaw(uq: np.ndarray) -> np.ndarray:
        u = np.asarray(uq, dtype=float).reshape(-1)
        return np.asarray(start_yaw_deg + (goal_yaw_deg - start_yaw_deg) * u, dtype=float)

    return _yaw


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(arr))
    if n < eps:
        return np.zeros_like(arr)
    return arr / n


def _approach_alignment_vectors(
    curve: np.ndarray,
    goal_normals: np.ndarray,
    terminal_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tail_n = max(3, int(np.ceil(float(terminal_fraction) * curve.shape[0])))
    v_approach = _normalize(np.sum(np.diff(curve[-tail_n:], axis=0), axis=0))

    normals = np.asarray(goal_normals, dtype=float).reshape(-1, 3)
    if normals.size == 0:
        summed_normal = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        summed_normal = _normalize(np.sum(normals, axis=0))
        if not np.any(summed_normal):
            summed_normal = _normalize(normals[0])
    desired_approach = -summed_normal
    return v_approach, summed_normal, desired_approach


def _fill_info_defaults(info: Dict[str, Any], *, start_yaw_deg: float, goal_yaw_deg: float, planner_cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(info)
    out.setdefault("yaw_fn", _make_linear_yaw_fn(start_yaw_deg, goal_yaw_deg))
    out.setdefault("required_clearance", float(planner_cfg.get("safety_margin", 0.0)))
    out.setdefault(
        "preferred_clearance",
        float(planner_cfg.get("preferred_safety_margin", planner_cfg.get("safety_margin", 0.0))),
    )
    out.setdefault("approach_only_clearance", planner_cfg.get("approach_only_clearance", None))
    out.setdefault("via_deviation_cost", 0.0)
    out.setdefault("yaw_deviation_cost", 0.0)
    out.setdefault("yaw_monotonic_cost", 0.0)
    out.setdefault("yaw_schedule_cost", 0.0)
    out.setdefault("goal_approach_normal_cost", 0.0)
    out.setdefault("preferred_safety_cost", 0.0)
    out.setdefault("approach_rebound_cost", 0.0)
    out.setdefault("goal_clearance_cost", 0.0)
    out.setdefault("goal_clearance_target_cost", 0.0)
    out.setdefault("approach_clearance_cost", 0.0)
    out.setdefault("approach_collision_cost", 0.0)
    out.setdefault("yaw_smoothness_cost", 0.0)
    out.setdefault("turn_angle_mean_deg", 0.0)
    out.setdefault("nit", 0)
    out.setdefault("message", "")
    return out


# CBS standalone stack names that bypass the external planner infrastructure
_CBS_STACK_NAMES = {"vpsto_path_planning", "vpsto_ilqr"}


def _is_cbs_stack(method: str) -> bool:
    return method.lower().replace("-", "_") in _CBS_STACK_NAMES


def _plan_cbs_stack(
    method: str,
    demo_scenario_name: str,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], Dict[str, Any]]:
    """Run a CBS standalone stack and return (curve_sampler, yaw_fn, info).

    All CBS scenarios and their block scenes are defined in K0_mounting_base
    frame, so no coordinate transformation is needed.
    """
    from motion_planning.standalone.scenarios import make_default_scenarios
    from motion_planning.standalone.stacks import STACK_REGISTRY

    stack_name = method.lower().replace("-", "_")
    fn = STACK_REGISTRY.get(stack_name)
    if fn is None:
        raise KeyError(f"CBS stack '{stack_name}' not in STACK_REGISTRY")

    cbs_scenarios = make_default_scenarios()
    # Match by overlay_scene_name (e.g. "step_01_first_on_ground")
    cbs_scenario = next(
        (sc for sc in cbs_scenarios.values()
         if getattr(sc, "overlay_scene_name", None) == demo_scenario_name),
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
        print(f"  pos err: {ev.final_position_error_m * 100:.1f} cm  "
              f"path len: {ev.path_length_m:.2f} m")

    tcp_xyz = np.asarray(result.tcp_xyz, dtype=float).reshape(-1, 3)
    tcp_yaw_rad = np.asarray(result.tcp_yaw_rad, dtype=float).ravel()
    t = np.linspace(0.0, 1.0, tcp_xyz.shape[0])

    def curve_sampler(uq: np.ndarray) -> np.ndarray:
        u = np.asarray(uq, dtype=float).reshape(-1)
        return np.column_stack([np.interp(u, t, tcp_xyz[:, i]) for i in range(3)])

    def yaw_fn(uq: np.ndarray) -> np.ndarray:
        u = np.asarray(uq, dtype=float).reshape(-1)
        return np.interp(u, t, np.degrees(tcp_yaw_rad))

    info: Dict[str, Any] = {
        "yaw_fn": yaw_fn,
        "success": result.success,
        "message": result.message,
        "nit": int(result.diagnostics.get("ilqr_iterations",
                    result.diagnostics.get("vpsto_iterations", 0))),
        "preferred_clearance": 0.05,
    }
    return curve_sampler, np.empty((0, 3), dtype=float), info


def _planner_entry(
    method: str,
    optimized_params_file: Path,
) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
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


def _plan_spline_method(
    planner_method: str,
    planner_cfg: Dict[str, Any],
    planner_options: Dict[str, Any],
    scenario,
) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, Dict[str, Any]]:
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
    if p_res.path.yaw_fn is not None:
        info["yaw_fn"] = p_res.path.yaw_fn
    return p_res.path.xyz_fn, np.empty((0, 3), dtype=float), info


def _plan_vpsto(
    scenario,
    planner_cfg: Dict[str, Any],
    planner_options: Dict[str, Any],
) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, Dict[str, Any]]:
    VPSTO, VPSTOOptions = _import_vpsto()
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

    def loss(candidates: Dict[str, np.ndarray]) -> np.ndarray:
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
    return _make_curve_sampler(curve), np.empty((0, 3), dtype=float), info


def _plan_rrt(
    scenario,
    planner_cfg: Dict[str, Any],
    planner_options: Dict[str, Any],
) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, Dict[str, Any]]:
    ob, og = _import_ompl()
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
    return _make_curve_sampler(curve), np.empty((0, 3), dtype=float), info


def _box_vertices(center: np.ndarray, size: tuple[float, float, float], yaw_deg: float) -> np.ndarray:
    cx, cy, cz = np.asarray(center, dtype=float).reshape(3)
    sx, sy, sz = (float(size[0]), float(size[1]), float(size[2]))
    hx, hy, hz = 0.5 * sx, 0.5 * sy, 0.5 * sz
    local = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=float,
    )
    R = quat_to_rot(yaw_deg_to_quat(yaw_deg))
    center_vec = np.array([cx, cy, cz], dtype=float)
    return (R @ local.T).T + center_vec


def _box_faces(vertices: np.ndarray) -> list[list[np.ndarray]]:
    return [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[4], vertices[7], vertices[3], vertices[0]],
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visual 3D demo for geometric planning scenarios.")
    parser.add_argument(
        "--scenario",
        default="step_02_second_in_front",
        help="Scenario name to run.",
    )
    parser.add_argument(
        "--scenarios-file",
        default=str(DEFAULT_SCENARIOS_FILE),
        help="Path to scenarios YAML file.",
    )
    parser.add_argument(
        "--planner",
        default="CEM",
        help="Planner backend: Powell, CEM, Nelder-Mead, VP-STO, or OMPL-RRT.",
    )
    parser.add_argument(
        "--optimized-params-file",
        default=str(DEFAULT_OPTIMIZED_PARAMS_FILE),
        help="Path to optimized planner params YAML.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=250,
        help="Number of path samples for plotting.",
    )
    parser.add_argument(
        "--animation-frames",
        type=int,
        default=180,
        help="Number of payload animation frames.",
    )
    parser.add_argument(
        "--save",
        default="",
        help="Optional path to save the rendered figure.",
    )
    return parser


def main() -> None:
    global _LAST_ANIMATION
    args = build_parser().parse_args()
    lib = ScenarioLibrary(args.scenarios_file)
    scenario_names = lib.list_scenarios()
    if args.scenario not in scenario_names:
        available = ", ".join(scenario_names)
        raise ValueError(f"Unknown scenario '{args.scenario}'. Available: {available}")

    scenario = lib.build_scenario(args.scenario)

    t_start = time.time()
    if _is_cbs_stack(args.planner):
        planner_method = args.planner.lower().replace("-", "_")
        planner_cfg: Dict[str, Any] = {"goal_approach_window_fraction": 0.1, "contact_window_fraction": 0.1}
        curve_sampler, vias_opt, info = _plan_cbs_stack(planner_method, args.scenario)
    else:
        planner_method, planner_cfg, planner_options = _planner_entry(
            method=args.planner,
            optimized_params_file=Path(args.optimized_params_file),
        )
        if planner_method in {"Powell", "CEM", "Nelder-Mead"}:
            curve_sampler, vias_opt, info = _plan_spline_method(
                planner_method=planner_method,
                planner_cfg=planner_cfg,
                planner_options=planner_options,
                scenario=scenario,
            )
        elif planner_method == "VP-STO":
            curve_sampler, vias_opt, info = _plan_vpsto(
                scenario=scenario,
                planner_cfg=planner_cfg,
                planner_options=planner_options,
            )
        else:
            curve_sampler, vias_opt, info = _plan_rrt(
                scenario=scenario,
                planner_cfg=planner_cfg,
                planner_options=planner_options,
            )
    opt_duration = time.time() - t_start
    print(f"Optimization took {opt_duration:.2f} seconds")

    info = _fill_info_defaults(
        info,
        start_yaw_deg=scenario.start_yaw_deg,
        goal_yaw_deg=scenario.goal_yaw_deg,
        planner_cfg=planner_cfg,
    )

    u = np.linspace(0.0, 1.0, max(10, int(args.samples)))
    curve = curve_sampler(u)
    goal_normals = np.asarray(scenario.goal_normals, dtype=float)
    v_approach, summed_normal, desired_approach = _approach_alignment_vectors(
        curve=curve,
        goal_normals=goal_normals,
        terminal_fraction=float(planner_cfg.get("goal_approach_window_fraction", 0.1)),
    )
    align_cos = float(np.clip(np.dot(v_approach, desired_approach), -1.0, 1.0))
    align_angle_deg = float(np.degrees(np.arccos(align_cos)))
    print(
        f"Approach alignment angle: {align_angle_deg:.2f} deg "
        f"(0 deg means perfectly aligned with -summed surface normals)"
    )

    fig = plt.figure(figsize=(13, 5.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax = plot_scene(scenario.scene, ax=ax, start=scenario.start, goal=scenario.goal)
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], "k-", lw=2, label=f"C2 B-spline ({planner_method})")

    for i, vp in enumerate(vias_opt):
        ax.scatter(*vp, s=30, label=f"v{i + 1} (opt)")

    normal_len = 0.35 * max(float(np.linalg.norm(np.asarray(scenario.moving_block_size, dtype=float))), 1e-6)
    g = np.asarray(scenario.goal, dtype=float)
    for n in goal_normals:
        nn = _normalize(np.asarray(n, dtype=float))
        ax.quiver(g[0], g[1], g[2], nn[0], nn[1], nn[2], length=normal_len, color="deepskyblue", linewidth=2.0)
    ax.quiver(
        g[0],
        g[1],
        g[2],
        summed_normal[0],
        summed_normal[1],
        summed_normal[2],
        length=normal_len,
        color="magenta",
        linewidth=2.5,
    )
    ax.quiver(
        g[0],
        g[1],
        g[2],
        v_approach[0],
        v_approach[1],
        v_approach[2],
        length=normal_len,
        color="red",
        linewidth=2.5,
    )
    ax.plot([], [], [], color="deepskyblue", lw=2, label="surface normals @ goal")
    ax.plot([], [], [], color="magenta", lw=2, label="resultant normal (sum normals)")
    ax.plot([], [], [], color="red", lw=2, label="actual approach direction")

    anim_u = np.linspace(0.0, 1.0, max(20, int(args.animation_frames)))
    anim_pts = curve_sampler(anim_u)
    yaw_fn = info["yaw_fn"]
    anim_yaw = np.asarray(yaw_fn(anim_u), dtype=float)
    anim_dists = np.array(
        [
            scenario.scene.signed_distance_block(
                size=scenario.moving_block_size,
                position=p,
                quat=yaw_deg_to_quat(float(anim_yaw[i])),
            )
            for i, p in enumerate(anim_pts)
        ],
        dtype=float,
    )
    anim_dists_eval = evaluated_clearance_subset(
        P=anim_pts,
        d=anim_dists,
        goal=np.asarray(scenario.goal, dtype=float),
        contact_window_fraction=float(planner_cfg.get("contact_window_fraction", 0.1)),
        goal_contact_radius=0.08,
    )
    min_anim_eval = float(np.min(anim_dists_eval))
    print(f"Min evaluated clearance along animation path: {min_anim_eval:+.3f} m")

    goal_dist_anim = np.linalg.norm(anim_pts - np.asarray(scenario.goal, dtype=float).reshape(1, 3), axis=1)
    contact_window_fraction = float(planner_cfg.get("contact_window_fraction", 0.1))
    contact_u_mask = anim_u < (1.0 - contact_window_fraction)
    goal_contact_mask = goal_dist_anim > 0.08
    eval_mask = contact_u_mask & goal_contact_mask
    anim_dists_eval_plot = np.full_like(anim_dists, np.nan, dtype=float)
    anim_dists_eval_plot[eval_mask] = anim_dists[eval_mask]

    ax_clear = fig.add_subplot(1, 2, 2)
    ax_clear.plot(anim_u, anim_dists, "b-", lw=2, label="signed distance (raw)")
    ax_clear.plot(anim_u, anim_dists_eval_plot, "k--", lw=2, label="signed distance (evaluated)")
    ax_clear.axhline(0.0, color="r", lw=1, ls="--", label="collision boundary")
    ax_clear.axhline(info["preferred_clearance"], color="orange", lw=1, ls="--", label="preferred clearance")
    if info.get("approach_only_clearance") is not None:
        ax_clear.axhline(info["approach_only_clearance"], color="green", lw=1, ls="--", label="approach clearance")
    clear_marker, = ax_clear.plot([anim_u[0]], [anim_dists[0]], "ko", ms=6)
    ax_clear.set_xlabel("path parameter u")
    ax_clear.set_ylabel("signed distance [m]")
    ax_clear.set_title("Block Clearance Along Path")
    ax_clear.grid(True, alpha=0.3)
    ax_clear.legend(loc="best")

    v0 = _box_vertices(anim_pts[0], scenario.moving_block_size, float(anim_yaw[0]))
    moving_poly = Poly3DCollection(_box_faces(v0), alpha=0.25, facecolor="limegreen", edgecolor="k", linewidths=0.8)
    ax.add_collection3d(moving_poly)
    moving_center = ax.scatter([anim_pts[0, 0]], [anim_pts[0, 1]], [anim_pts[0, 2]], s=40, c="k", label="moving block")
    dist_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def _frame_color(dist: float) -> str:
        if dist < 0.0:
            return "crimson"
        if dist < 0.03:
            return "darkorange"
        return "limegreen"

    def _update(frame_idx: int):
        p = anim_pts[frame_idx]
        dist = float(anim_dists[frame_idx])
        vv = _box_vertices(p, scenario.moving_block_size, float(anim_yaw[frame_idx]))
        moving_poly.set_verts(_box_faces(vv))
        moving_poly.set_facecolor(_frame_color(dist))
        moving_center._offsets3d = ([p[0]], [p[1]], [p[2]])
        dist_text.set_text(f"clearance: {dist:+.3f} m, yaw: {anim_yaw[frame_idx]:+.1f} deg")
        clear_marker.set_data([anim_u[frame_idx]], [dist])
        return moving_poly, moving_center, dist_text, clear_marker

    ax.legend(loc="upper right")
    ax.view_init(elev=25, azim=45)
    plt.tight_layout()
    backend = matplotlib.get_backend().lower()
    is_headless_backend = backend in {"agg", "pdf", "pgf", "ps", "svg", "template", "cairo"}
    if not is_headless_backend:
        anim = FuncAnimation(fig=ax.figure, func=_update, frames=len(anim_pts), interval=50, blit=False, repeat=True)
        _LAST_ANIMATION = anim
        setattr(fig, "_motion_planning_animation", anim)
    if args.save:
        out = Path(args.save).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180)
        print(f"Saved figure to {out}")
    if is_headless_backend:
        plt.close(fig)
        return
    plt.show(block=True)


if __name__ == "__main__":
    main()
