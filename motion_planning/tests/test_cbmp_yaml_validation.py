from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cbmp.runtime import RuntimeHelpersMixin


class _FakeLogger:
    def __init__(self) -> None:
        self.warns: list[str] = []
        self.errors: list[str] = []
        self.infos: list[str] = []

    def warn(self, msg: str) -> None:
        self.warns.append(msg)

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def info(self, msg: str) -> None:
        self.infos.append(msg)


class _FakeNode(RuntimeHelpersMixin):
    def __init__(self) -> None:
        self._default_named_joint_names = ["joint_1", "joint_2"]
        self._named_cfg_default_duration_s = 4.0
        self._named_configurations = {}

        self._wall_plan_frame_id = "world"
        self._wall_plans = {}
        self._wall_plan_progress = {}

        self._logger = _FakeLogger()

    def get_logger(self) -> _FakeLogger:
        return self._logger


def test_named_config_loader_rejects_invalid_entries(tmp_path: Path) -> None:
    cfg_file = tmp_path / "named.yaml"
    cfg_file.write_text(
        """
joint_names: [joint_1, joint_2]
configurations:
  valid_home:
    positions: [0.0, 1.0]
    duration_s: 2.5
  bad_non_numeric:
    positions: [0.0, nope]
  bad_duration:
    positions: [1.0, 2.0]
    duration_s: -1.0
  bad_duplicates:
    positions: [1.0, 2.0]
    joint_names: [joint_1, joint_1]
""",
        encoding="utf-8",
    )

    node = _FakeNode()
    node._load_named_configurations_from_file(str(cfg_file))

    assert sorted(node._named_configurations.keys()) == ["valid_home"]
    assert len(node.get_logger().warns) >= 2


def test_wall_plan_loader_rejects_duplicate_block_ids(tmp_path: Path) -> None:
    wall_file = tmp_path / "walls.yaml"
    wall_file.write_text(
        """
wall_plans:
  demo:
    sequence:
      - id: A
        absolute_position: [0.0, 0.0, 0.3]
      - id: A
        absolute_position: [1.0, 0.0, 0.3]
""",
        encoding="utf-8",
    )

    node = _FakeNode()
    node._load_wall_plans_from_file(str(wall_file))

    assert node._wall_plans == {}
    assert any("duplicates block id" in msg for msg in node.get_logger().warns)
