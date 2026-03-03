#!/usr/bin/env python3
"""Regression tests for benchmark clearance handling.

Run:
  python -m motion_planning.tests.test_benchmark_clearance_regression
"""

from __future__ import annotations

import numpy as np

from motion_planning_tools.benchmark.metrics import evaluate_path_metrics, make_eval_context


class _TerminalContactScene:
    """Scene stub: positive clearance along approach, negative only at terminal contact."""

    def signed_distance_block(self, size, position, quat):
        x = float(np.asarray(position, dtype=float)[0])
        return -1.0 if x >= 0.9 else 0.05


def test_metrics_ignore_terminal_goal_contact() -> None:
    scene = _TerminalContactScene()
    P = np.column_stack(
        [
            np.linspace(0.0, 1.0, 21),
            np.zeros(21, dtype=float),
            np.zeros(21, dtype=float),
        ]
    )
    ctx = make_eval_context(
        scene=scene,
        goal=np.array([1.0, 0.0, 0.0], dtype=float),
        moving_block_size=(0.1, 0.1, 0.1),
        start_yaw_deg=0.0,
        goal_yaw_deg=0.0,
        goal_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        config={"contact_window_fraction": 0.1, "w_len": 1.0, "w_curv": 0.1, "w_safe": 1.0},
    )
    info = evaluate_path_metrics(ctx=ctx, P=P, message="test", nit=0, solver_success=True)
    assert info["min_clearance"] > 0.0, f"expected positive approach clearance, got {info['min_clearance']}"
    assert info["min_clearance_raw"] < 0.0, "raw clearance should retain terminal contact negativity"
    assert bool(info["success"]), "solver + approach clearance should count as success"


def test_metrics_with_custom_yaw_samples_keeps_terminal_contact_out_of_eval() -> None:
    """Regression proxy for spline methods that provide custom yaw profiles."""
    scene = _TerminalContactScene()
    P = np.column_stack(
        [
            np.linspace(0.0, 1.0, 21),
            np.zeros(21, dtype=float),
            np.zeros(21, dtype=float),
        ]
    )
    yaw = np.linspace(-45.0, 45.0, 21)
    ctx = make_eval_context(
        scene=scene,
        goal=np.array([1.0, 0.0, 0.0], dtype=float),
        moving_block_size=(0.1, 0.1, 0.1),
        start_yaw_deg=0.0,
        goal_yaw_deg=0.0,
        goal_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        config={"contact_window_fraction": 0.1, "w_len": 1.0, "w_curv": 0.1, "w_safe": 1.0},
    )
    info = evaluate_path_metrics(
        ctx=ctx,
        P=P,
        message="test",
        nit=0,
        yaw_samples_deg=yaw,
        solver_success=True,
    )
    assert info["min_clearance"] > 0.0, f"expected positive approach clearance, got {info['min_clearance']}"
    assert info["min_clearance_raw"] < 0.0, "raw clearance should keep terminal negative contact"
    assert bool(info["success"]), "solver + approach clearance should count as success"


if __name__ == "__main__":
    test_metrics_ignore_terminal_goal_contact()
    test_metrics_with_custom_yaw_samples_keeps_terminal_contact_out_of_eval()
    print("benchmark clearance regression tests passed")
