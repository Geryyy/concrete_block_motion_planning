from __future__ import annotations

from typing import Iterable

import numpy as np

from motion_planning.geometry.vis import plot_scene

from .types import SolverComparisonResult, StandalonePlanResult


def _compute_speeds(
    tcp_xyz: np.ndarray, time_s: np.ndarray | None
) -> np.ndarray | None:
    """Return per-point speed values, or None if no timing is available."""
    n = len(tcp_xyz)
    if time_s is None or len(time_s) != n or n < 2:
        return None
    dt = np.diff(time_s)
    ds = np.linalg.norm(np.diff(tcp_xyz, axis=0), axis=1)
    seg_speed = np.where(dt > 1e-9, ds / dt, 0.0)
    return np.concatenate(
        [[seg_speed[0]], (seg_speed[:-1] + seg_speed[1:]) / 2, [seg_speed[-1]]]
    )


def _add_approach_arrows(ax, xyz: np.ndarray, yaw_rad: np.ndarray, *, is_3d: bool) -> None:
    """Draw short approach-direction arrows at start and goal."""
    length = 0.25
    for idx, color in [(0, "tab:blue"), (-1, "tab:red")]:
        p = xyz[idx]
        yaw = float(yaw_rad[idx])
        dx, dy = length * np.cos(yaw), length * np.sin(yaw)
        if is_3d:
            ax.quiver(
                p[0], p[1], p[2], dx, dy, 0.0,
                color=color, arrow_length_ratio=0.4, linewidth=1.5,
            )
        else:
            ax.annotate(
                "",
                xy=(p[0] + dx, p[1] + dy),
                xytext=(p[0], p[1]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            )


def _colored_path_3d(ax, xyz: np.ndarray, speeds: np.ndarray, cmap: str = "plasma") -> object:
    """Scatter the 3D TCP path coloured by speed/arc-position."""
    return ax.scatter(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        c=speeds, cmap=cmap, s=14, zorder=5,
        vmin=float(speeds.min()), vmax=float(speeds.max()),
    )


def _colored_path_2d(ax, x: np.ndarray, y: np.ndarray, speeds: np.ndarray, cmap: str = "plasma") -> object:
    """LineCollection along (x, y) coloured by speed/arc-position."""
    from matplotlib.collections import LineCollection

    pts = np.stack([x, y], axis=1).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    # mid-point colour for each segment
    seg_c = (speeds[:-1] + speeds[1:]) / 2.0
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=float(speeds.min()), vmax=float(speeds.max()))
    lc = LineCollection(segs, cmap=cmap, norm=norm, lw=2.0)
    lc.set_array(seg_c)
    ax.add_collection(lc)
    return lc


def plot_solver_results(results: Iterable[SolverComparisonResult]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"matplotlib unavailable: {exc}") from exc

    items = list(results)
    fig, ax = plt.subplots(figsize=(7, 5))
    for item in items:
        label = f"{item.name} | err={item.position_error_m:.3f}m"
        ax.scatter(item.fk_xyz[0], item.fk_xyz[1], label=label)
    ax.set_title("Solver Comparison (TCP XY)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_plan_result(
    result: StandalonePlanResult,
    *,
    scene=None,
    scene_name: str | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"matplotlib unavailable: {exc}") from exc

    tcp_xyz = np.asarray(result.tcp_xyz, dtype=float).reshape(-1, 3)
    ref_xyz = np.asarray(result.reference_xyz, dtype=float).reshape(-1, 3)
    tcp_yaw = np.asarray(result.tcp_yaw_rad, dtype=float).ravel()
    q_waypoints = np.asarray(result.q_waypoints, dtype=float)

    speeds = _compute_speeds(tcp_xyz, result.time_s)
    cmap = "plasma"

    if scene is None:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        ax_scene = None
        xy_ax, xz_ax, q_ax = axes
    else:
        fig = plt.figure(figsize=(18, 9))
        ax_scene = fig.add_subplot(2, 2, 1, projection="3d")
        xy_ax = fig.add_subplot(2, 2, 2)
        xz_ax = fig.add_subplot(2, 2, 3)
        q_ax = fig.add_subplot(2, 2, 4)

    # ---- 3-D scene subplot ------------------------------------------------
    if ax_scene is not None:
        plot_scene(
            scene,
            ax=ax_scene,
            start=ref_xyz[0],
            goal=ref_xyz[-1],
            show_legend=False,
        )
        ax_scene.plot(
            ref_xyz[:, 0], ref_xyz[:, 1], ref_xyz[:, 2],
            "k--", lw=1.0, alpha=0.6, label="reference",
        )
        ax_scene.plot(
            tcp_xyz[:, 0], tcp_xyz[:, 1], tcp_xyz[:, 2],
            color="tab:orange", lw=2.0, label=result.stack_name,
        )
        if speeds is not None:
            sc3d = _colored_path_3d(ax_scene, tcp_xyz, speeds, cmap)
            cb = plt.colorbar(sc3d, ax=ax_scene, shrink=0.6, pad=0.02)
            cb.set_label("TCP speed [m/s]", fontsize=8)
        _add_approach_arrows(ax_scene, tcp_xyz, tcp_yaw, is_3d=True)

        title = "Scene + TCP Path" if scene_name is None else f"Scene: {scene_name}"
        ax_scene.set_title(title)
        ax_scene.legend(loc="upper right", fontsize=7)

    # ---- XY projection ----------------------------------------------------
    xy_ax.plot(ref_xyz[:, 0], ref_xyz[:, 1], "k--", lw=1.0, alpha=0.6, label="reference")
    if speeds is not None:
        lc_xy = _colored_path_2d(xy_ax, tcp_xyz[:, 0], tcp_xyz[:, 1], speeds, cmap)
        xy_ax.autoscale()
        cb2 = plt.colorbar(lc_xy, ax=xy_ax)
        cb2.set_label("TCP speed [m/s]", fontsize=8)
    else:
        xy_ax.plot(tcp_xyz[:, 0], tcp_xyz[:, 1], color="tab:orange", lw=2.0, label=result.stack_name)
    _add_approach_arrows(xy_ax, tcp_xyz, tcp_yaw, is_3d=False)
    xy_ax.autoscale()
    xy_ax.set_title(f"TCP XY  —  {result.stack_name}")
    xy_ax.set_xlabel("x [m]")
    xy_ax.set_ylabel("y [m]")
    xy_ax.set_aspect("equal", adjustable="datalim")
    xy_ax.legend(fontsize=7)

    # ---- XZ projection ----------------------------------------------------
    xz_ax.plot(ref_xyz[:, 0], ref_xyz[:, 2], "k--", lw=1.0, alpha=0.6, label="reference")
    if speeds is not None:
        lc_xz = _colored_path_2d(xz_ax, tcp_xyz[:, 0], tcp_xyz[:, 2], speeds, cmap)
        xz_ax.autoscale()
        cb3 = plt.colorbar(lc_xz, ax=xz_ax)
        cb3.set_label("TCP speed [m/s]", fontsize=8)
    else:
        xz_ax.plot(tcp_xyz[:, 0], tcp_xyz[:, 2], color="tab:orange", lw=2.0, label=result.stack_name)
    xz_ax.autoscale()
    xz_ax.set_title("TCP XZ")
    xz_ax.set_xlabel("x [m]")
    xz_ax.set_ylabel("z [m]")
    xz_ax.legend(fontsize=7)

    # ---- Joint path -------------------------------------------------------
    time_axis = result.time_s if result.time_s is not None else np.arange(q_waypoints.shape[0])
    xlabel = "time [s]" if result.time_s is not None else "sample"
    for j in range(q_waypoints.shape[1]):
        q_ax.plot(time_axis, q_waypoints[:, j], label=f"q{j + 1}")
    q_ax.set_title("Joint Path")
    q_ax.set_xlabel(xlabel)
    q_ax.set_ylabel("joint value [rad / m]")
    q_ax.legend(fontsize=7)

    # ---- Eval annotation ---------------------------------------------------
    if result.evaluation is not None:
        ev = result.evaluation
        info = (
            f"final pos err: {ev.final_position_error_m * 100:.1f} cm  "
            f"| final yaw err: {ev.final_yaw_error_deg:.1f}°  "
            f"| path len: {ev.path_length_m:.2f} m"
        )
        fig.suptitle(info, fontsize=9, y=0.02)

    plt.tight_layout()
    plt.show()
