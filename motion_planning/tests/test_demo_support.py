from __future__ import annotations

from motion_planning.standalone import is_cbs_stack


def test_is_cbs_stack_matches_only_supported_standalone_name() -> None:
    assert is_cbs_stack("joint_space_global_path")
    assert is_cbs_stack("joint-space-global-path")
    assert not is_cbs_stack("vpsto_path_planning")
    assert not is_cbs_stack("Powell")
