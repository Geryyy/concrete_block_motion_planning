from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cbmp.execution import ExecutionAdapter


def test_measured_motion_metrics_reports_static_feedback() -> None:
    max_abs, norm, covered = ExecutionAdapter._measured_motion_metrics(
        {"theta1_slewing_joint": 0.5, "q4_big_telescope": 2.0},
        {"theta1_slewing_joint": 0.5, "q4_big_telescope": 2.0},
    )

    assert covered == ["theta1_slewing_joint", "q4_big_telescope"]
    assert max_abs == pytest.approx(0.0)
    assert norm == pytest.approx(0.0)


def test_measured_motion_metrics_reports_joint_progress() -> None:
    max_abs, norm, covered = ExecutionAdapter._measured_motion_metrics(
        {"theta1_slewing_joint": 0.5, "q4_big_telescope": 2.0},
        {"theta1_slewing_joint": 0.7, "q4_big_telescope": 1.8},
    )

    assert covered == ["theta1_slewing_joint", "q4_big_telescope"]
    assert max_abs == pytest.approx(0.2)
    assert norm > 0.28


class _DummyNode:
    _execution_motion_check_timeout_s = 0.0
    _execution_motion_min_delta = 0.02


def test_validate_motion_uses_command_joint_subset() -> None:
    adapter = ExecutionAdapter(_DummyNode())

    class _Trajectory:
        joint_names = [
            "theta1_slewing_joint",
            "theta2_boom_joint",
            "theta3_arm_joint",
            "q4_big_telescope",
            "theta6_tip_joint",
            "theta7_tilt_joint",
            "theta8_rotator_joint",
            "q9_left_rail_joint",
        ]

    ok, message = adapter._validate_motion_after_execution(
        _Trajectory(),
        {
            "theta1_slewing_joint": 0.1,
            "theta2_boom_joint": -1.0,
            "theta3_arm_joint": -0.4,
            "q4_big_telescope": 2.2,
            "theta8_rotator_joint": -0.2,
            "q9_left_rail_joint": 0.05,
        },
    )
    assert not ok
    assert "theta6_tip_joint" not in message
    assert "theta7_tilt_joint" not in message
