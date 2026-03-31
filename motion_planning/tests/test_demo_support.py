from __future__ import annotations

from pathlib import Path

from motion_planning.standalone.demo_support import is_cbs_stack, planner_entry


def test_is_cbs_stack_matches_supported_standalone_names() -> None:
    assert is_cbs_stack("vpsto_path_planning")
    assert is_cbs_stack("VPSTO-ILQR")
    assert not is_cbs_stack("Powell")


def test_planner_entry_resolves_canonical_method() -> None:
    params_file = Path(__file__).resolve().parents[1] / "data" / "optimized_params.yaml"
    method, cfg, opts = planner_entry("powell", params_file)
    assert method == "Powell"
    assert isinstance(cfg, dict)
    assert isinstance(opts, dict)
