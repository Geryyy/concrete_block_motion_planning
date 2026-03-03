from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from motion_planning.geometry.spline_opt import optimize_bspline_path
from motion_planning.core.spline import BSplinePath
from motion_planning.core.types import PlannerRequest, PlannerResult


@dataclass
class SplineOptimizerPlanner:
    """Adapter for spline-based methods in geom.spline_opt."""

    method: str

    def plan(self, req: PlannerRequest) -> PlannerResult:
        sc = req.scenario
        cfg = dict(req.config)
        options = dict(req.options)
        moving_block_size = cfg.pop("moving_block_size", tuple(sc.moving_block_size))

        S_xyz, _, info = optimize_bspline_path(
            scene=sc.scene,
            start=np.asarray(sc.start, dtype=float),
            goal=np.asarray(sc.goal, dtype=float),
            moving_block_size=None if moving_block_size is None else tuple(moving_block_size),
            start_yaw_deg=float(sc.start_yaw_deg),
            goal_yaw_deg=float(sc.goal_yaw_deg),
            goal_approach_normals=np.asarray(sc.goal_normals, dtype=float),
            method=self.method,
            options=options,
            **cfg,
        )

        yaw_fn = info.get("yaw_fn") if callable(info.get("yaw_fn")) else None
        path = BSplinePath(xyz_fn=S_xyz, yaw_fn=yaw_fn)
        metrics = _extract_scalar_metrics(info)
        diagnostics: Dict[str, Any] = {
            "nit": int(info.get("nit", 0)),
            "solver_method": self.method,
        }
        ctrl_pts = info.get("xyz_ctrl_pts", None)
        if ctrl_pts is not None:
            diagnostics["ctrl_pts_xyz"] = np.asarray(ctrl_pts, dtype=float)
        if "xyz_spline_degree" in info:
            diagnostics["spline_degree"] = int(info["xyz_spline_degree"])
        return PlannerResult(
            success=bool(info.get("success", False)),
            message=str(info.get("message", "")),
            path=path,
            metrics=metrics,
            diagnostics=diagnostics,
        )


def _extract_scalar_metrics(info: Dict[str, Any]) -> Dict[str, float]:
    keys = [
        "fun",
        "length",
        "curvature_cost",
        "safety_cost",
        "approach_collision_cost",
        "goal_approach_normal_cost",
        "min_clearance",
        "mean_clearance",
        "turn_angle_mean_deg",
    ]
    out: Dict[str, float] = {}
    for k in keys:
        if k in info:
            out[k] = float(info[k])
    return out
