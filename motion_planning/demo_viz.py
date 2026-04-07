from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from motion_planning.geometry.spline_opt import yaw_deg_to_quat
from motion_planning.geometry.utils import quat_to_rot
from motion_planning_tools.benchmark.metrics import evaluated_clearance_subset


@dataclass(frozen=True)
class AnimationData:
    u: np.ndarray
    pts: np.ndarray
    yaw_deg: np.ndarray
    payload_clearance: np.ndarray
    payload_clearance_eval: np.ndarray
    payload_clearance_eval_plot: np.ndarray
    arm_clearance: np.ndarray
    combined_clearance: np.ndarray


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(arr))
    return np.zeros_like(arr) if n < eps else arr / n


def approach_alignment_vectors(
    curve: np.ndarray,
    goal_normals: np.ndarray,
    terminal_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tail_n = max(3, int(np.ceil(float(terminal_fraction) * curve.shape[0])))
    v_approach = normalize(np.sum(np.diff(curve[-tail_n:], axis=0), axis=0))
    normals = np.asarray(goal_normals, dtype=float).reshape(-1, 3)
    if normals.size == 0:
        summed_normal = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        summed_normal = normalize(np.sum(normals, axis=0))
        if not np.any(summed_normal):
            summed_normal = normalize(normals[0])
    return v_approach, summed_normal, -summed_normal


def fill_info_defaults(info: dict[str, Any], *, start_yaw_deg: float, goal_yaw_deg: float, make_linear_yaw_fn, planner_cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(info)
    out.setdefault("yaw_fn", make_linear_yaw_fn(start_yaw_deg, goal_yaw_deg))
    out.setdefault("required_clearance", float(planner_cfg.get("safety_margin", 0.0)))
    out.setdefault("preferred_clearance", float(planner_cfg.get("preferred_safety_margin", planner_cfg.get("safety_margin", 0.0))))
    out.setdefault("approach_only_clearance", planner_cfg.get("approach_only_clearance", None))
    for key in (
        "via_deviation_cost",
        "yaw_deviation_cost",
        "yaw_monotonic_cost",
        "yaw_schedule_cost",
        "goal_approach_normal_cost",
        "preferred_safety_cost",
        "approach_rebound_cost",
        "goal_clearance_cost",
        "goal_clearance_target_cost",
        "approach_clearance_cost",
        "approach_collision_cost",
        "yaw_smoothness_cost",
    ):
        out.setdefault(key, 0.0)
    out.setdefault("turn_angle_mean_deg", 0.0)
    out.setdefault("nit", 0)
    out.setdefault("message", "")
    return out


def box_vertices(center: np.ndarray, size: tuple[float, float, float], yaw_deg: float) -> np.ndarray:
    cx, cy, cz = np.asarray(center, dtype=float).reshape(3)
    sx, sy, sz = (float(size[0]), float(size[1]), float(size[2]))
    hx, hy, hz = 0.5 * sx, 0.5 * sy, 0.5 * sz
    local = np.array(
        [[-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz], [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]],
        dtype=float,
    )
    return (quat_to_rot(yaw_deg_to_quat(yaw_deg)) @ local.T).T + np.array([cx, cy, cz], dtype=float)


def box_faces(vertices: np.ndarray) -> list[list[np.ndarray]]:
    return [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[4], vertices[7], vertices[3], vertices[0]],
    ]


def draw_capsule_segments(ax, segments, *, color: str, alpha: float, label: str | None = None) -> None:
    first = True
    for seg in segments:
        p1 = np.asarray(seg.p1, dtype=float)
        p2 = np.asarray(seg.p2, dtype=float)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, alpha=alpha, lw=max(1.5, 16.0 * float(seg.radius)), solid_capstyle="round", label=label if first else None)
        first = False


def draw_approach_arrow(ax, origin: np.ndarray, direction: np.ndarray, length: float, *, color: str, label: str) -> None:
    d = normalize(direction)
    ax.quiver(float(origin[0]), float(origin[1]), float(origin[2]), float(d[0]), float(d[1]), float(d[2]), length=float(length), color=color, linewidth=2.5)
    ax.plot([], [], [], color=color, lw=2.5, label=label)


def draw_frame_axes(ax, tf: np.ndarray, length: float, *, label: str) -> None:
    tf = np.asarray(tf, dtype=float)
    o, R = tf[:3, 3], tf[:3, :3]
    for i, color in enumerate(["red", "green", "blue"]):
        d = R[:, i] * float(length)
        ax.quiver(float(o[0]), float(o[1]), float(o[2]), float(d[0]), float(d[1]), float(d[2]), color=color, linewidth=2.0)
    ax.scatter([o[0]], [o[1]], [o[2]], c="k", s=18, label=label)


def tool_frame_with_z_down(tf: np.ndarray) -> np.ndarray:
    corrected = np.asarray(tf, dtype=float).copy()
    corrected[:3, :3] = corrected[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    return corrected


def set_demo_axes(ax, scene, *point_sets: np.ndarray, margin: float = 0.35) -> None:
    pts = [np.asarray(blk.vertices_world(), dtype=float).reshape(-1, 3) for blk in getattr(scene, "blocks", [])]
    pts.extend(np.asarray(arr, dtype=float).reshape(-1, 3) for arr in point_sets if np.asarray(arr, dtype=float).size)
    if not pts:
        return
    all_pts = np.vstack(pts)
    mins, maxs = np.min(all_pts, axis=0), np.max(all_pts, axis=0)
    center = 0.5 * (mins + maxs)
    half = 0.5 * (np.max(maxs - mins) + 2.0 * float(margin))
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))


def print_path_diagnostics(curve: np.ndarray, goal_normals: np.ndarray, diagnostics: dict[str, Any], planner_cfg: dict[str, Any]) -> None:
    v_approach, _, desired_approach = approach_alignment_vectors(curve=curve, goal_normals=goal_normals, terminal_fraction=float(planner_cfg.get("goal_approach_window_fraction", 0.1)))
    align_angle_deg = float(np.degrees(np.arccos(float(np.clip(np.dot(v_approach, desired_approach), -1.0, 1.0)))))
    print(f"Approach alignment angle: {align_angle_deg:.2f} deg (0 deg means perfectly aligned with -summed surface normals)")
    if diagnostics.get("combined_min_clearance_m") is not None:
        print("Capsule clearance summary: " f"arm={float(diagnostics.get('arm_min_clearance_m', np.nan)):+.3f} m, " f"payload={float(diagnostics.get('payload_min_clearance_m', np.nan)):+.3f} m, " f"combined={float(diagnostics.get('combined_min_clearance_m', np.nan)):+.3f} m")


def draw_goal_vectors(ax, scenario, curve: np.ndarray, planner_cfg: dict[str, Any]) -> None:
    goal_normals = np.asarray(scenario.goal_normals, dtype=float)
    v_approach, summed_normal, desired_approach = approach_alignment_vectors(curve, goal_normals, float(planner_cfg.get("goal_approach_window_fraction", 0.1)))
    normal_len = 0.35 * max(float(np.linalg.norm(np.asarray(scenario.moving_block_size, dtype=float))), 1e-6)
    goal = np.asarray(scenario.goal, dtype=float)
    for normal in goal_normals:
        nn = normalize(np.asarray(normal, dtype=float))
        ax.quiver(goal[0], goal[1], goal[2], nn[0], nn[1], nn[2], length=normal_len, color="deepskyblue", linewidth=2.0)
    draw_approach_arrow(ax, goal, summed_normal, normal_len, color="magenta", label="resultant normal (sum normals)")
    draw_approach_arrow(ax, goal, desired_approach, normal_len, color="darkorange", label="requested approach direction")
    draw_approach_arrow(ax, goal, v_approach, normal_len, color="red", label="realized approach direction")
    ax.plot([], [], [], color="deepskyblue", lw=2, label="surface normals @ goal")


def build_animation_data(curve_sampler: Any, yaw_fn: Any, diagnostics: dict[str, Any], scenario, planner_cfg: dict[str, Any], arm_model, frame_count: int) -> tuple[AnimationData, list[Any]]:
    anim_u = np.linspace(0.0, 1.0, max(20, int(frame_count)))
    anim_pts = curve_sampler(anim_u)
    anim_yaw = np.asarray(yaw_fn(anim_u), dtype=float)
    q_maps_path = diagnostics.get("q_maps_path", [])
    if q_maps_path:
        dense_u = np.linspace(0.0, 1.0, len(q_maps_path), dtype=float)
        tcp_xyz_dense = np.asarray(diagnostics["tcp_xyz_path"], dtype=float)
        tcp_yaw_dense = np.asarray(diagnostics["tcp_yaw_path_rad"], dtype=float).reshape(-1)
        arm_clear_dense = np.asarray([arm_model.clearance(q_map, scenario.scene, ignore_ids=["table"]) for q_map in q_maps_path], dtype=float)
        payload_clear_dense = np.asarray([arm_model.payload_clearance(tcp_xyz_dense[i], float(tcp_yaw_dense[i]), scenario.moving_block_size, scenario.scene) for i in range(len(q_maps_path))], dtype=float)
        combined_clear_dense = np.minimum(arm_clear_dense, payload_clear_dense)
        arm_clear_anim = np.interp(anim_u, dense_u, arm_clear_dense)
        combined_clear_anim = np.interp(anim_u, dense_u, combined_clear_dense)
    else:
        arm_clear_anim = np.full(anim_u.shape, np.nan, dtype=float)
        combined_clear_anim = np.full(anim_u.shape, np.nan, dtype=float)
    anim_dists = np.asarray([scenario.scene.signed_distance_block(size=scenario.moving_block_size, position=point, quat=yaw_deg_to_quat(float(anim_yaw[i]))) for i, point in enumerate(anim_pts)], dtype=float)
    goal = np.asarray(scenario.goal, dtype=float)
    anim_dists_eval = evaluated_clearance_subset(P=anim_pts, d=anim_dists, goal=goal, contact_window_fraction=float(planner_cfg.get("contact_window_fraction", 0.1)), goal_contact_radius=0.08)
    goal_dist_anim = np.linalg.norm(anim_pts - goal.reshape(1, 3), axis=1)
    eval_mask = (anim_u < (1.0 - float(planner_cfg.get("contact_window_fraction", 0.1)))) & (goal_dist_anim > 0.08)
    anim_dists_eval_plot = np.full_like(anim_dists, np.nan, dtype=float)
    anim_dists_eval_plot[eval_mask] = anim_dists[eval_mask]
    return AnimationData(anim_u, anim_pts, anim_yaw, anim_dists, anim_dists_eval, anim_dists_eval_plot, arm_clear_anim, combined_clear_anim), q_maps_path


def add_arm_visuals(ax, q_maps_path: list[Any], arm_model) -> None:
    if not q_maps_path:
        return
    draw_capsule_segments(ax, arm_model.capsule_segments(q_maps_path[0]), color="royalblue", alpha=0.30, label="crane capsules (start)")
    draw_capsule_segments(ax, arm_model.capsule_segments(q_maps_path[-1]), color="forestgreen", alpha=0.30, label="crane capsules (goal)")
    draw_frame_axes(ax, tool_frame_with_z_down(arm_model._frame_tf(q_maps_path[0], "K8_tool_center_point")), 0.18, label="tool frame (start, z-down)")
    if len(q_maps_path) > 1:
        draw_frame_axes(ax, tool_frame_with_z_down(arm_model._frame_tf(q_maps_path[-1], "K8_tool_center_point")), 0.18, label="tool frame (goal, z-down)")


def set_demo_bounds(ax, scene, curve: np.ndarray, animation: AnimationData, start: np.ndarray, goal: np.ndarray, vias_opt: np.ndarray, q_maps_path: list[Any], arm_model) -> None:
    bounds_pts = [np.asarray(curve, dtype=float), np.asarray(animation.pts, dtype=float), np.asarray([start, goal], dtype=float), np.asarray(vias_opt, dtype=float)]
    if q_maps_path:
        capsule_pts = [p for q_map in [q_maps_path[0], q_maps_path[-1]] for seg in arm_model.capsule_segments(q_map) for p in (seg.p1, seg.p2)]
        if capsule_pts:
            bounds_pts.append(np.asarray(capsule_pts, dtype=float))
    set_demo_axes(ax, scene, *bounds_pts)


def plot_clearance_axis(ax_clear, animation: AnimationData, info: dict[str, Any]):
    ax_clear.plot(animation.u, animation.payload_clearance, "b-", lw=1.5, label="payload clearance (raw)")
    ax_clear.plot(animation.u, animation.payload_clearance_eval_plot, "k--", lw=1.5, label="payload clearance (evaluated)")
    if np.any(np.isfinite(animation.arm_clearance)):
        ax_clear.plot(animation.u, animation.arm_clearance, color="purple", lw=2.0, label="arm capsule clearance")
        ax_clear.plot(animation.u, animation.combined_clearance, color="green", lw=2.0, label="combined clearance")
    ax_clear.axhline(0.0, color="r", lw=1, ls="--", label="collision boundary")
    ax_clear.axhline(info["preferred_clearance"], color="orange", lw=1, ls="--", label="preferred clearance")
    if info.get("approach_only_clearance") is not None:
        ax_clear.axhline(info["approach_only_clearance"], color="green", lw=1, ls="--", label="approach clearance")
    marker_y = animation.combined_clearance[0] if np.isfinite(animation.combined_clearance[0]) else animation.payload_clearance[0]
    clear_marker, = ax_clear.plot([animation.u[0]], [marker_y], "ko", ms=6)
    ax_clear.set_xlabel("path parameter u")
    ax_clear.set_ylabel("signed distance [m]")
    ax_clear.set_title("Clearance Along Path")
    ax_clear.grid(True, alpha=0.3)
    ax_clear.legend(loc="best")
    return clear_marker


def attach_animation(ax, fig, animation: AnimationData, scenario, clear_marker) -> FuncAnimation:
    moving_poly = Poly3DCollection(box_faces(box_vertices(animation.pts[0], scenario.moving_block_size, float(animation.yaw_deg[0]))), alpha=0.25, facecolor="limegreen", edgecolor="k", linewidths=0.8)
    ax.add_collection3d(moving_poly)
    moving_center = ax.scatter([animation.pts[0, 0]], [animation.pts[0, 1]], [animation.pts[0, 2]], s=40, c="k", label="moving block")
    dist_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def update(frame_idx: int):
        point = animation.pts[frame_idx]
        payload_dist = float(animation.payload_clearance[frame_idx])
        arm_dist = float(animation.arm_clearance[frame_idx]) if np.isfinite(animation.arm_clearance[frame_idx]) else float("nan")
        combined_dist = float(animation.combined_clearance[frame_idx]) if np.isfinite(animation.combined_clearance[frame_idx]) else payload_dist
        moving_poly.set_verts(box_faces(box_vertices(point, scenario.moving_block_size, float(animation.yaw_deg[frame_idx]))))
        moving_poly.set_facecolor("crimson" if combined_dist < 0.0 else "darkorange" if combined_dist < 0.03 else "limegreen")
        moving_center._offsets3d = ([point[0]], [point[1]], [point[2]])
        dist_text.set_text(f"payload: {payload_dist:+.3f} m | arm: {arm_dist:+.3f} m | combined: {combined_dist:+.3f} m")
        clear_marker.set_data([animation.u[frame_idx]], [combined_dist])
        return moving_poly, moving_center, dist_text, clear_marker

    return FuncAnimation(fig=fig, func=update, frames=len(animation.pts), interval=50, blit=False, repeat=True)
