from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from motion_planning.geometry import plot_scene
from motion_planning.standalone import (
    DEFAULT_PLANNER_CFG,
    is_cbs_stack as demo_is_cbs_stack,
    make_linear_yaw_fn as demo_make_linear_yaw_fn,
    plan_cbs_stack as demo_plan_cbs_stack,
)
from motion_planning.demo_viz import (
    add_arm_visuals,
    attach_animation,
    build_animation_data,
    draw_goal_vectors,
    fill_info_defaults,
    print_path_diagnostics,
    plot_clearance_axis,
    set_demo_bounds,
)
from motion_planning.scenarios import ScenarioLibrary


DEFAULT_SCENARIOS_FILE = Path(__file__).with_name("data").joinpath("generated_scenarios.yaml")
_LAST_ANIMATION: FuncAnimation | None = None
HEADLESS_BACKENDS = {"agg", "pdf", "pgf", "ps", "svg", "template", "cairo"}


@dataclass(frozen=True)
class PlannedDemo:
    planner_method: str
    curve_sampler: Any
    vias_opt: np.ndarray
    info: Dict[str, Any]
    diagnostics: Dict[str, Any]


def _start_q_map_for_demo_scenario(scene_name: str, arm_model) -> dict[str, float] | None:
    from motion_planning import make_default_scenarios

    sc = next(
        (s for s in make_default_scenarios().values() if getattr(s, "overlay_scene_name", None) == scene_name),
        None,
    )
    if sc is None or sc.planner_start_q is None:
        return None
    if sc.planner_start_q_seed_map is not None:
        return dict(sc.planner_start_q_seed_map)
    return arm_model.complete_joint_map(np.asarray(sc.planner_start_q, dtype=float))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visual 3D demo for geometric planning scenarios.")
    parser.add_argument(
        "--scenario",
        default="step_01_first_on_ground",
        help="Scenario name to run.",
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
    parser.add_argument(
        "--start-only",
        action="store_true",
        help="Skip planning and visualize only the start pose.",
    )
    return parser


def _load_scenario(library: ScenarioLibrary, scenario_name: str):
    scenario_names = library.list_scenarios()
    if scenario_name not in scenario_names:
        available = ", ".join(scenario_names)
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {available}")
    return library.build_scenario(scenario_name)


def _constant_curve_sampler(point: np.ndarray):
    point = np.asarray(point, dtype=float).reshape(3)

    def curve_sampler(uq: np.ndarray) -> np.ndarray:
        u = np.asarray(uq, dtype=float).reshape(-1)
        return np.repeat(point.reshape(1, 3), u.size, axis=0)

    return curve_sampler


def _constant_yaw_fn(yaw_deg: float):
    yaw_deg = float(yaw_deg)
    return lambda uq: np.full(np.asarray(uq, dtype=float).reshape(-1).shape, yaw_deg, dtype=float)


def _start_only_plan(scenario_name: str, scenario, arm_model) -> PlannedDemo:
    start_xyz = np.asarray(scenario.start, dtype=float).reshape(3)
    start_yaw = float(scenario.start_yaw_deg)
    start_q_map = _start_q_map_for_demo_scenario(scenario_name, arm_model)
    diagnostics = {}
    if start_q_map is not None:
        diagnostics = {
            "q_maps_path": [start_q_map],
            "tcp_xyz_path": [start_xyz.tolist()],
            "tcp_yaw_path_rad": [np.radians(start_yaw)],
        }
    info = {
        "success": True,
        "message": "Visualizing start pose only.",
        "yaw_fn": _constant_yaw_fn(start_yaw),
        "planner_fallback": True,
        "diagnostics": diagnostics,
    }
    return PlannedDemo(
        planner_method="start_only",
        curve_sampler=_constant_curve_sampler(start_xyz),
        vias_opt=np.empty((0, 3), dtype=float),
        info=info,
        diagnostics=diagnostics,
    )


def _plan_demo(scenario_name: str, scenario, *, start_only: bool) -> PlannedDemo:
    from motion_planning.geometry.arm_model import CraneArmCollisionModel

    planner_method = "joint_space_global_path"
    if start_only:
        return _start_only_plan(scenario_name, scenario, CraneArmCollisionModel())
    if not demo_is_cbs_stack(planner_method):
        raise ValueError(f"Unsupported demo planner '{planner_method}'")
    curve_sampler, vias_opt, info = demo_plan_cbs_stack(planner_method, scenario_name)
    info = dict(info)
    info["planner_fallback"] = False
    diagnostics = dict(info.get("diagnostics", {}))
    return PlannedDemo(
        planner_method=planner_method,
        curve_sampler=curve_sampler,
        vias_opt=np.asarray(vias_opt, dtype=float),
        info=info,
        diagnostics=diagnostics,
    )


def main() -> None:
    global _LAST_ANIMATION
    from motion_planning.geometry.arm_model import CraneArmCollisionModel

    args = build_parser().parse_args()
    lib = ScenarioLibrary(DEFAULT_SCENARIOS_FILE)
    scenario = _load_scenario(lib, args.scenario)
    planner_cfg = dict(DEFAULT_PLANNER_CFG)
    t_start = time.time()
    planned = _plan_demo(args.scenario, scenario, start_only=args.start_only)
    opt_duration = time.time() - t_start
    print(f"Optimization took {opt_duration:.2f} seconds")

    info = fill_info_defaults(
        planned.info,
        start_yaw_deg=scenario.start_yaw_deg,
        goal_yaw_deg=scenario.goal_yaw_deg,
        make_linear_yaw_fn=demo_make_linear_yaw_fn,
        planner_cfg=planner_cfg,
    )

    u = np.linspace(0.0, 1.0, max(10, int(args.samples)))
    curve = planned.curve_sampler(u)
    diagnostics = dict(info.get("diagnostics", planned.diagnostics))
    print_path_diagnostics(curve, np.asarray(scenario.goal_normals, dtype=float), diagnostics, planner_cfg)

    fig = plt.figure(figsize=(13, 5.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax = plot_scene(scenario.scene, ax=ax, start=scenario.start, goal=scenario.goal)
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], "k-", lw=2, label=f"TCP path ({planned.planner_method})")
    ax.set_title("Stage-1 Planner in K0_mounting_base")

    for i, vp in enumerate(planned.vias_opt):
        ax.scatter(*vp, s=30, label=f"v{i + 1} (opt)")
    arm_model = CraneArmCollisionModel()
    draw_goal_vectors(ax, scenario, curve, planner_cfg)
    animation, q_maps_path = build_animation_data(
        planned.curve_sampler,
        info["yaw_fn"],
        diagnostics,
        scenario,
        planner_cfg,
        arm_model,
        args.animation_frames,
    )
    add_arm_visuals(ax, q_maps_path, arm_model)
    set_demo_bounds(
        ax,
        scenario.scene,
        curve,
        animation,
        np.asarray(scenario.start, dtype=float),
        np.asarray(scenario.goal, dtype=float),
        planned.vias_opt,
        q_maps_path,
        arm_model,
    )

    min_anim_eval = float(np.min(animation.payload_clearance_eval))
    print(f"Min evaluated clearance along animation path: {min_anim_eval:+.3f} m")

    ax_clear = fig.add_subplot(1, 2, 2)
    clear_marker = plot_clearance_axis(ax_clear, animation, info)

    ax.legend(loc="upper right")
    ax.view_init(elev=25, azim=45)
    plt.tight_layout()
    backend = matplotlib.get_backend().lower()
    is_headless_backend = backend in HEADLESS_BACKENDS
    if not is_headless_backend:
        anim = attach_animation(ax, fig, animation, scenario, clear_marker)
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
