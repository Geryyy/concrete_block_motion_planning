from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import StandaloneScenario


@dataclass(frozen=True)
class ReferencePath:
    xyz: np.ndarray
    yaw_rad: np.ndarray
    diagnostics: dict[str, float | str]


def build_linear_reference_path(
    scenario: StandaloneScenario,
    *,
    waypoint_count: int,
    start_xyz: tuple[float, float, float] | np.ndarray | None = None,
    start_yaw_rad: float | None = None,
) -> ReferencePath:
    n_wp = max(2, int(waypoint_count))
    start_xyz_arr = np.asarray(
        scenario.start_world_xyz if start_xyz is None else start_xyz,
        dtype=float,
    ).reshape(3)
    goal_xyz_arr = np.asarray(scenario.goal_world_xyz, dtype=float).reshape(3)
    xyz = np.vstack(
        [
            np.linspace(start_xyz_arr[i], goal_xyz_arr[i], n_wp, dtype=float)
            for i in range(3)
        ]
    ).T
    yaw = np.linspace(
        float(scenario.start_yaw_rad if start_yaw_rad is None else start_yaw_rad),
        float(scenario.goal_yaw_rad),
        n_wp,
        dtype=float,
    )
    return ReferencePath(
        xyz=xyz,
        yaw_rad=yaw,
        diagnostics={
            "reference_path_backend": "linear",
            "reference_path_fallback_used": 0.0,
            "reference_waypoint_count": float(n_wp),
        },
    )
