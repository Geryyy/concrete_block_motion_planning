from __future__ import annotations

from typing import Iterable

import numpy as np

from motion_planning.geometry.vis import plot_scene

from .types import SolverComparisonResult, StandalonePlanResult


def plot_solver_results(results: Iterable[SolverComparisonResult]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting
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


def plot_plan_result(result: StandalonePlanResult, *, scene=None, scene_name: str | None = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting
        raise RuntimeError(f"matplotlib unavailable: {exc}") from exc

    tcp_xyz = np.asarray(result.tcp_xyz, dtype=float).reshape(-1, 3)
    ref_xyz = np.asarray(result.reference_xyz, dtype=float).reshape(-1, 3)
    q_waypoints = np.asarray(result.q_waypoints, dtype=float)

    if scene is None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        axes_list = list(axes)
        ax_scene = None
    else:
        fig = plt.figure(figsize=(17, 9))
        ax_scene = fig.add_subplot(2, 2, 1, projection="3d")
        axes_list = [
            ax_scene,
            fig.add_subplot(2, 2, 2),
            fig.add_subplot(2, 2, 3),
            fig.add_subplot(2, 2, 4),
        ]

    if ax_scene is not None:
        plot_scene(
            scene,
            ax=ax_scene,
            start=ref_xyz[0, :],
            goal=ref_xyz[-1, :],
            show_legend=True,
        )
        ax_scene.plot(
            ref_xyz[:, 0],
            ref_xyz[:, 1],
            ref_xyz[:, 2],
            "k--",
            lw=1.5,
            label="reference path",
        )
        ax_scene.plot(
            tcp_xyz[:, 0],
            tcp_xyz[:, 1],
            tcp_xyz[:, 2],
            color="tab:orange",
            lw=2.0,
            label=result.stack_name,
        )
        title = "Scene + TCP Path" if scene_name is None else f"Scene + TCP Path: {scene_name}"
        ax_scene.set_title(title)
        ax_scene.legend(loc="upper right")

    xy_ax = axes_list[0] if ax_scene is None else axes_list[1]
    xz_ax = axes_list[1] if ax_scene is None else axes_list[2]
    q_ax = axes_list[2] if ax_scene is None else axes_list[3]

    xy_ax.plot(ref_xyz[:, 0], ref_xyz[:, 1], "k--", label="reference")
    xy_ax.plot(tcp_xyz[:, 0], tcp_xyz[:, 1], label=result.stack_name)
    xy_ax.set_title("TCP XY")
    xy_ax.set_xlabel("x [m]")
    xy_ax.set_ylabel("y [m]")
    xy_ax.axis("equal")
    xy_ax.legend()

    xz_ax.plot(ref_xyz[:, 0], ref_xyz[:, 2], "k--", label="reference")
    xz_ax.plot(tcp_xyz[:, 0], tcp_xyz[:, 2], label=result.stack_name)
    xz_ax.set_title("TCP XZ")
    xz_ax.set_xlabel("x [m]")
    xz_ax.set_ylabel("z [m]")
    xz_ax.legend()

    for j in range(q_waypoints.shape[1]):
        q_ax.plot(q_waypoints[:, j], label=f"q{j + 1}")
    q_ax.set_title("Joint Path")
    q_ax.set_xlabel("sample")
    q_ax.set_ylabel("joint value")
    q_ax.legend()

    plt.tight_layout()
    plt.show()
