from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest
import yaml

# Allow importing motion_planning package in test environments without python-fcl.
if "fcl" not in sys.modules:
    fcl = types.ModuleType("fcl")

    class _Dummy:  # pragma: no cover - test-only shim
        def __init__(self, *args, **kwargs) -> None:
            pass

    fcl.CollisionObject = _Dummy
    fcl.Box = _Dummy
    fcl.Transform = _Dummy
    sys.modules["fcl"] = fcl

from motion_planning.scenarios import DEFAULT_WALL_PLANS_FILE, WallPlanLibrary


def test_wall_plan_library_lists_default_plan() -> None:
    lib = WallPlanLibrary()
    names = lib.list_plans()
    assert "basic_interlocking_3_2" in names


def test_basic_interlocking_plan_resolves_relative_positions() -> None:
    lib = WallPlanLibrary()
    plan = lib.build_plan("basic_interlocking_3_2")
    assert len(plan.placements) == 5

    p = {entry.block_id: entry for entry in plan.placements}

    assert p["B0"].reference_block_id is None
    assert p["B0"].absolute_position == pytest.approx((0.0, 0.0, 0.3))

    assert p["B1"].reference_block_id == "B0"
    assert p["B1"].absolute_position == pytest.approx((0.62, 0.0, 0.3))
    assert p["B2"].reference_block_id == "B1"
    assert p["B2"].absolute_position == pytest.approx((1.24, 0.0, 0.3))

    assert p["T0"].reference_block_id == "B0"
    assert p["T0"].absolute_position == pytest.approx((0.31, 0.0, 0.9))
    assert p["T1"].reference_block_id == "T0"
    assert p["T1"].absolute_position == pytest.approx((0.93, 0.0, 0.9))


def test_wall_plan_rejects_unknown_reference(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad_wall_plan.yaml"
    payload = {
        "defaults": {"block_size": [0.6, 0.9, 0.6]},
        "wall_plans": {
            "bad": {
                "sequence": [
                    {"id": "B0", "absolute_position": [0.0, 0.0, 0.3]},
                    {"id": "B1", "relative_to": "UNKNOWN", "offset": [0.5, 0.0, 0.0]},
                ]
            }
        },
    }
    bad_file.write_text(yaml.safe_dump(payload), encoding="utf-8")

    lib = WallPlanLibrary(plans_file=bad_file)
    with pytest.raises(ValueError, match="references unknown block"):
        lib.build_plan("bad")


def test_default_wall_plan_file_exists() -> None:
    assert Path(DEFAULT_WALL_PLANS_FILE).exists()
