from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from geometry_msgs.msg import PoseStamped


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cbmp.services import ServiceHandlersMixin
from cbmp.types import WallPlanTask


class _FakeNode(ServiceHandlersMixin):
    def __init__(self) -> None:
        self._default_wall_plan_name = "demo"
        self._wall_plans = {
            "demo": [
                WallPlanTask(
                    task_id="demo_01_A",
                    target_block_id="A",
                    reference_block_id="",
                    target_pose=PoseStamped(),
                    reference_pose=PoseStamped(),
                ),
                WallPlanTask(
                    task_id="demo_02_B",
                    target_block_id="B",
                    reference_block_id="A",
                    target_pose=PoseStamped(),
                    reference_pose=PoseStamped(),
                ),
            ]
        }
        self._wall_plan_progress = {"demo": 0}


def _req(name: str = "", reset: bool = False) -> SimpleNamespace:
    return SimpleNamespace(wall_plan_name=name, reset_plan=reset)


def _res() -> SimpleNamespace:
    return SimpleNamespace()


def test_wall_plan_progression_and_completion() -> None:
    node = _FakeNode()

    first = node._handle_get_next_assembly_task(_req(), _res())
    assert first.success is True
    assert first.has_task is True
    assert first.task_id == "demo_01_A"

    second = node._handle_get_next_assembly_task(_req(), _res())
    assert second.success is True
    assert second.has_task is True
    assert second.task_id == "demo_02_B"

    done = node._handle_get_next_assembly_task(_req(), _res())
    assert done.success is True
    assert done.has_task is False
    assert "completed" in done.message


def test_wall_plan_reset_restarts_sequence() -> None:
    node = _FakeNode()

    _ = node._handle_get_next_assembly_task(_req(), _res())
    restarted = node._handle_get_next_assembly_task(_req(reset=True), _res())

    assert restarted.success is True
    assert restarted.has_task is True
    assert restarted.task_id == "demo_01_A"


def test_unknown_wall_plan_returns_failure() -> None:
    node = _FakeNode()

    response = node._handle_get_next_assembly_task(_req(name="unknown"), _res())

    assert response.success is False
    assert response.has_task is False
    assert "Unknown wall plan" in response.message
