from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from joint_state_timber_compat import CompatRangeMap, map_source_gripper_position


def test_map_source_gripper_position_maps_pzs100_range_to_timber_range() -> None:
    range_map = CompatRangeMap(
        source_min=0.0,
        source_max=0.538,
        compat_min=0.8472,
        compat_max=3.0357,
    )

    assert map_source_gripper_position(0.0, range_map) == pytest.approx(0.8472)
    assert map_source_gripper_position(0.538, range_map) == pytest.approx(3.0357)
    midpoint = map_source_gripper_position(0.269, range_map)
    assert midpoint == pytest.approx((0.8472 + 3.0357) / 2.0)


def test_map_source_gripper_position_clamps_outside_range() -> None:
    range_map = CompatRangeMap(
        source_min=0.0,
        source_max=0.538,
        compat_min=0.8472,
        compat_max=3.0357,
    )

    assert map_source_gripper_position(-1.0, range_map) == pytest.approx(0.8472)
    assert map_source_gripper_position(99.0, range_map) == pytest.approx(3.0357)
