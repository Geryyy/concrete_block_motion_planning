from __future__ import annotations

from motion_planning.standalone.compare_solvers import compare_solver_suite
from motion_planning.standalone.scenarios import make_default_scenarios
from motion_planning.standalone.stacks import STACK_REGISTRY


def test_default_scenarios_have_expected_entries() -> None:
    scenarios = make_default_scenarios()
    assert "single_block_transfer" in scenarios
    assert "short_reachable_move" in scenarios


def test_solver_compare_suite_returns_start_and_goal() -> None:
    scenario = make_default_scenarios()["single_block_transfer"]
    results = compare_solver_suite(scenario)
    assert len(results) == 2
    assert results[0].name == "concrete_start"
    assert results[1].name == "concrete_goal"
    assert results[0].ik_backend != ""


def test_joint_goal_interpolation_stack_returns_evaluation() -> None:
    scenario = make_default_scenarios()["short_reachable_move"]
    result = STACK_REGISTRY["joint_goal_interpolation"](scenario)
    assert result.success
    assert result.evaluation is not None
    assert result.q_waypoints.shape[0] >= 10


def test_anchor_joint_spline_stack_runs() -> None:
    scenario = make_default_scenarios()["short_reachable_move"]
    result = STACK_REGISTRY["cartesian_anchor_joint_spline"](scenario)
    assert result.success
    assert "anchor_count" in result.diagnostics


def test_simple_time_scaling_adds_duration() -> None:
    from motion_planning.standalone.stacks import apply_simple_time_scaling

    scenario = make_default_scenarios()["short_reachable_move"]
    result = STACK_REGISTRY["joint_goal_interpolation"](scenario)
    assert result.success
    timed = apply_simple_time_scaling(result)
    assert timed.time_s is not None
    assert timed.dq_waypoints is not None
    assert timed.diagnostics["timing_backend"] == "simple_limits"
    assert timed.diagnostics["duration_s"] > 0.0
