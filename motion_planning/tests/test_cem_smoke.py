#!/usr/bin/env python3
"""Simple smoke tests for the CEM optimizer backend.

Run:
  python -m motion_planning.tests.test_cem_smoke
"""

from __future__ import annotations

import numpy as np

from motion_planning.geometry import Scene
from motion_planning.geometry.spline_opt import _simple_cem_optimize, optimize_bspline_path


def test_cem_quadratic() -> None:
    x_star = np.array([1.5, -2.0, 0.75, 3.0], dtype=float)

    def objective(X: np.ndarray) -> np.ndarray:
        D = X - x_star[None, :]
        return np.sum(D * D, axis=1)

    res = _simple_cem_optimize(
        objective=objective,
        x0=np.zeros_like(x_star),
        sigma0=np.ones_like(x_star),
        population_size=80,
        elite_frac=0.2,
        max_iter=70,
        sample_method="Gaussian",
        seed=7,
    )

    x = np.asarray(res["x"], dtype=float)
    err = float(np.linalg.norm(x - x_star))
    assert err < 0.6, f"quadratic test failed: ||x-x*||={err:.4f}"


def test_cem_path_planning() -> None:
    scene = Scene()
    start = np.array([0.0, 0.0, 0.5], dtype=float)
    goal = np.array([1.2, 0.8, 0.5], dtype=float)

    _, _, info = optimize_bspline_path(
        scene=scene,
        start=start,
        goal=goal,
        n_vias=2,
        n_yaw_vias=2,
        combined_4d=True,
        n_samples_curve=81,
        moving_block_size=(0.3, 0.2, 0.2),
        start_yaw_deg=0.0,
        goal_yaw_deg=0.0,
        method="CEM",
        options={
            "population_size": 64,
            "elite_frac": 0.2,
            "max_iter": 45,
            "sample_method": "Gaussian",
            "seed": 11,
        },
        cost_mode="simple",
    )

    assert bool(info.get("success", False)), f"planning did not report success: {info.get('message', '')}"
    assert np.isfinite(float(info.get("fun", np.inf))), "planning objective is not finite"
    assert float(info.get("length", 0.0)) > 0.0, "path length must be positive"


if __name__ == "__main__":
    test_cem_quadratic()
    test_cem_path_planning()
    print("CEM smoke tests passed")
