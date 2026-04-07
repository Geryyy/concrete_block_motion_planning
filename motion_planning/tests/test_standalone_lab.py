from __future__ import annotations

from motion_planning import STACK_REGISTRY, make_default_scenarios


def test_default_scenarios_have_expected_entries() -> None:
    scenarios = make_default_scenarios()
    assert "single_block_transfer" in scenarios
    assert "short_reachable_move" in scenarios


def test_joint_space_global_path_stack_runs() -> None:
    scenario = make_default_scenarios()["short_reachable_move"]
    result = STACK_REGISTRY["joint_space_global_path"](scenario)
    assert result.success
    assert result.evaluation is not None
    assert result.diagnostics["via_point_count"] == 2.0
