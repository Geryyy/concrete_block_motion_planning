from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from motion_planning.geometry import plot_scene
from motion_planning.geometry.spline_opt import yaw_deg_to_quat
from motion_planning.geometry.utils import quat_to_rot
from motion_planning.standalone.demo_support import (
    DEFAULT_PLANNER_CFG,
    is_cbs_stack as demo_is_cbs_stack,
    make_linear_yaw_fn as demo_make_linear_yaw_fn,
    plan_cbs_stack as demo_plan_cbs_stack,
)
from motion_planning.scenarios import ScenarioLibrary
from motion_planning_tools.benchmark.metrics import (
    evaluated_clearance_subset,
)


DEFAULT_SCENARIOS_FILE = Path(__file__).with_name("data").joinpath("generated_scenarios.yaml")
_LAST_ANIMATION: FuncAnimation | None = None


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
    out.setdefault("yaw_fn", demo_make_linear_yaw_fn(start_yaw_deg, goal_yaw_deg))
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


def _draw_capsule_segments(ax, segments, *, color: str, alpha: float, label: str | None = None) -> None:
    first = True
    for seg in segments:
        p1 = np.asarray(seg.p1, dtype=float)
        p2 = np.asarray(seg.p2, dtype=float)
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color=color,
            alpha=alpha,
            lw=max(1.5, 16.0 * float(seg.radius)),
            solid_capstyle="round",
            label=label if first else None,
        )
        first = False


def _draw_approach_arrow(ax, origin: np.ndarray, direction: np.ndarray, length: float, *, color: str, label: str) -> None:
    d = _normalize(direction)
    ax.quiver(
        float(origin[0]),
        float(origin[1]),
        float(origin[2]),
        float(d[0]),
        float(d[1]),
        float(d[2]),
        length=float(length),
        color=color,
        linewidth=2.5,
    )
    ax.plot([], [], [], color=color, lw=2.5, label=label)


def _draw_frame_axes(ax, tf: np.ndarray, length: float, *, label: str) -> None:
    tf = np.asarray(tf, dtype=float)
    o = tf[:3, 3]
    R = tf[:3, :3]
    colors = ["red", "green", "blue"]
    for i, color in enumerate(colors):
        d = R[:, i] * float(length)
        ax.quiver(float(o[0]), float(o[1]), float(o[2]), float(d[0]), float(d[1]), float(d[2]), color=color, linewidth=2.0)
    ax.scatter([o[0]], [o[1]], [o[2]], c="k", s=18, label=label)


def _set_demo_axes(ax, scene, *point_sets: np.ndarray, margin: float = 0.35) -> None:
    pts = []
    for blk in getattr(scene, "blocks", []):
        pts.append(np.asarray(blk.vertices_world(), dtype=float).reshape(-1, 3))
    for arr in point_sets:
        a = np.asarray(arr, dtype=float)
        if a.size:
            pts.append(a.reshape(-1, 3))
    if not pts:
        return
    all_pts = np.vstack(pts)
    mins = np.min(all_pts, axis=0)
    maxs = np.max(all_pts, axis=0)
    center = 0.5 * (mins + maxs)
    span = np.max(maxs - mins) + 2.0 * float(margin)
    half = 0.5 * span
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))


def _start_q_map_for_demo_scenario(scene_name: str, arm_model) -> dict[str, float] | None:
    from motion_planning.standalone.scenarios import make_default_scenarios

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
        default="pzs100_live_block_2_top",
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


def main() -> None:
    global _LAST_ANIMATION
    from motion_planning.geometry.arm_model import CraneArmCollisionModel

    args = build_parser().parse_args()
    lib = ScenarioLibrary(DEFAULT_SCENARIOS_FILE)
    scenario_names = lib.list_scenarios()
    if args.scenario not in scenario_names:
        available = ", ".join(scenario_names)
        raise ValueError(f"Unknown scenario '{args.scenario}'. Available: {available}")

    scenario = lib.build_scenario(args.scenario)

    planner_method = "joint_space_global_path"
    planner_cfg = dict(DEFAULT_PLANNER_CFG)
    t_start = time.time()
    if args.start_only:
        start_xyz = np.asarray(scenario.start, dtype=float).reshape(3)
        start_yaw = float(scenario.start_yaw_deg)
        arm_model = CraneArmCollisionModel()
        start_q_map = _start_q_map_for_demo_scenario(args.scenario, arm_model)

        def curve_sampler(uq: np.ndarray) -> np.ndarray:
            u = np.asarray(uq, dtype=float).reshape(-1)
            return np.repeat(start_xyz.reshape(1, 3), u.size, axis=0)

        info = {
            "success": True,
            "message": "Visualizing start pose only.",
            "yaw_fn": lambda uq: np.full(np.asarray(uq, dtype=float).reshape(-1).shape, start_yaw, dtype=float),
            "planner_fallback": True,
            "diagnostics": {} if start_q_map is None else {
                "q_maps_path": [start_q_map],
                "tcp_xyz_path": [start_xyz.tolist()],
                "tcp_yaw_path_rad": [np.radians(start_yaw)],
            },
        }
        vias_opt = np.empty((0, 3), dtype=float)
        planner_method = "start_only"
    else:
        if not demo_is_cbs_stack(planner_method):
            raise ValueError(f"Unsupported demo planner '{planner_method}'")
        curve_sampler, vias_opt, info = demo_plan_cbs_stack(planner_method, args.scenario)
        info["planner_fallback"] = False
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
    diagnostics = dict(info.get("diagnostics", {}))
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
    if diagnostics.get("combined_min_clearance_m") is not None:
        print(
            "Capsule clearance summary: "
            f"arm={float(diagnostics.get('arm_min_clearance_m', np.nan)):+.3f} m, "
            f"payload={float(diagnostics.get('payload_min_clearance_m', np.nan)):+.3f} m, "
            f"combined={float(diagnostics.get('combined_min_clearance_m', np.nan)):+.3f} m"
        )

    fig = plt.figure(figsize=(13, 5.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax = plot_scene(scenario.scene, ax=ax, start=scenario.start, goal=scenario.goal)
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], "k-", lw=2, label=f"TCP path ({planner_method})")
    ax.set_title("Stage-1 Planner in K0_mounting_base")

    for i, vp in enumerate(vias_opt):
        ax.scatter(*vp, s=30, label=f"v{i + 1} (opt)")

    normal_len = 0.35 * max(float(np.linalg.norm(np.asarray(scenario.moving_block_size, dtype=float))), 1e-6)
    g = np.asarray(scenario.goal, dtype=float)
    for n in goal_normals:
        nn = _normalize(np.asarray(n, dtype=float))
        ax.quiver(g[0], g[1], g[2], nn[0], nn[1], nn[2], length=normal_len, color="deepskyblue", linewidth=2.0)
    _draw_approach_arrow(ax, g, summed_normal, normal_len, color="magenta", label="resultant normal (sum normals)")
    _draw_approach_arrow(ax, g, desired_approach, normal_len, color="darkorange", label="requested approach direction")
    _draw_approach_arrow(ax, g, v_approach, normal_len, color="red", label="realized approach direction")
    ax.plot([], [], [], color="deepskyblue", lw=2, label="surface normals @ goal")

    anim_u = np.linspace(0.0, 1.0, max(20, int(args.animation_frames)))
    anim_pts = curve_sampler(anim_u)
    yaw_fn = info["yaw_fn"]
    anim_yaw = np.asarray(yaw_fn(anim_u), dtype=float)
    arm_model = CraneArmCollisionModel()
    q_maps_path = diagnostics.get("q_maps_path", [])
    if q_maps_path:
        dense_u = np.linspace(0.0, 1.0, len(q_maps_path), dtype=float)
        tcp_xyz_dense = np.asarray(diagnostics["tcp_xyz_path"], dtype=float)
        tcp_yaw_dense = np.asarray(diagnostics["tcp_yaw_path_rad"], dtype=float).reshape(-1)
        arm_clear_dense = np.asarray(
            [arm_model.clearance(q_map, scenario.scene, ignore_ids=["table"]) for q_map in q_maps_path],
            dtype=float,
        )
        payload_clear_dense = np.asarray(
            [
                arm_model.payload_clearance(
                    tcp_xyz_dense[i],
                    float(tcp_yaw_dense[i]),
                    scenario.moving_block_size,
                    scenario.scene,
                )
                for i in range(len(q_maps_path))
            ],
            dtype=float,
        )
        combined_clear_dense = np.minimum(arm_clear_dense, payload_clear_dense)
        arm_clear_anim = np.interp(anim_u, dense_u, arm_clear_dense)
        combined_clear_anim = np.interp(anim_u, dense_u, combined_clear_dense)
        _draw_capsule_segments(
            ax,
            arm_model.capsule_segments(q_maps_path[0]),
            color="royalblue",
            alpha=0.30,
            label="crane capsules (start)",
        )
        _draw_capsule_segments(
            ax,
            arm_model.capsule_segments(q_maps_path[-1]),
            color="forestgreen",
            alpha=0.30,
            label="crane capsules (goal)",
        )
        frame_len = 0.18
        _draw_frame_axes(
            ax,
            arm_model._frame_tf(q_maps_path[0], "K8_tool_center_point"),
            frame_len,
            label="K8 frame (start)",
        )
        if len(q_maps_path) > 1:
            _draw_frame_axes(
                ax,
                arm_model._frame_tf(q_maps_path[-1], "K8_tool_center_point"),
                frame_len,
                label="K8 frame (goal)",
            )
    else:
        arm_clear_anim = np.full(anim_u.shape, np.nan, dtype=float)
        combined_clear_anim = np.full(anim_u.shape, np.nan, dtype=float)

    bounds_pts = [
        np.asarray(curve, dtype=float),
        np.asarray(anim_pts, dtype=float),
        np.asarray([scenario.start, scenario.goal], dtype=float),
        np.asarray(vias_opt, dtype=float),
    ]
    if q_maps_path:
        capsule_pts = []
        for q_map in [q_maps_path[0], q_maps_path[-1]]:
            for seg in arm_model.capsule_segments(q_map):
                capsule_pts.extend([seg.p1, seg.p2])
        if capsule_pts:
            bounds_pts.append(np.asarray(capsule_pts, dtype=float))
    _set_demo_axes(ax, scenario.scene, *bounds_pts)

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
    ax_clear.plot(anim_u, anim_dists, "b-", lw=1.5, label="payload clearance (raw)")
    ax_clear.plot(anim_u, anim_dists_eval_plot, "k--", lw=1.5, label="payload clearance (evaluated)")
    if np.any(np.isfinite(arm_clear_anim)):
        ax_clear.plot(anim_u, arm_clear_anim, color="purple", lw=2.0, label="arm capsule clearance")
        ax_clear.plot(anim_u, combined_clear_anim, color="green", lw=2.0, label="combined clearance")
    ax_clear.axhline(0.0, color="r", lw=1, ls="--", label="collision boundary")
    ax_clear.axhline(info["preferred_clearance"], color="orange", lw=1, ls="--", label="preferred clearance")
    if info.get("approach_only_clearance") is not None:
        ax_clear.axhline(info["approach_only_clearance"], color="green", lw=1, ls="--", label="approach clearance")
    clear_marker, = ax_clear.plot([anim_u[0]], [combined_clear_anim[0] if np.isfinite(combined_clear_anim[0]) else anim_dists[0]], "ko", ms=6)
    ax_clear.set_xlabel("path parameter u")
    ax_clear.set_ylabel("signed distance [m]")
    ax_clear.set_title("Clearance Along Path")
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
        arm_dist = float(arm_clear_anim[frame_idx]) if np.isfinite(arm_clear_anim[frame_idx]) else float("nan")
        combined_dist = float(combined_clear_anim[frame_idx]) if np.isfinite(combined_clear_anim[frame_idx]) else dist
        vv = _box_vertices(p, scenario.moving_block_size, float(anim_yaw[frame_idx]))
        moving_poly.set_verts(_box_faces(vv))
        moving_poly.set_facecolor(_frame_color(combined_dist))
        moving_center._offsets3d = ([p[0]], [p[1]], [p[2]])
        dist_text.set_text(
            f"payload: {dist:+.3f} m | arm: {arm_dist:+.3f} m | combined: {combined_dist:+.3f} m"
        )
        clear_marker.set_data([anim_u[frame_idx]], [combined_dist])
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
