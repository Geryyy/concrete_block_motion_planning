from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cbmp.config import declare_and_load_config


class _Parameter:
    def __init__(self, value):
        self.value = value


class _FakeNode:
    def __init__(self, overrides: dict[str, object] | None = None):
        self._params: dict[str, object] = dict(overrides or {})

    def declare_parameter(self, name: str, default_value):
        if name not in self._params:
            self._params[name] = default_value
        return _Parameter(self._params[name])

    def get_parameter(self, name: str):
        return _Parameter(self._params[name])


def test_declare_and_load_config_defaults() -> None:
    node = _FakeNode()

    cfg = declare_and_load_config(node)

    assert cfg.default_geometric_method == "POWELL"
    assert cfg.default_trajectory_method == "ACADOS_PATH_FOLLOWING"
    assert cfg.path_interpolation_points == 81
    assert cfg.moving_block_size == (0.6, 0.9, 0.6)
    assert cfg.execution_enabled is False
    assert cfg.default_named_joint_names == []
    assert cfg.default_wall_plan_name == "basic_interlocking_3_2"
    assert cfg.wall_plan_frame_id == "world"


def test_declare_and_load_config_respects_overrides() -> None:
    node = _FakeNode(
        {
            "default_geometric_method": "SLSQP",
            "path_interpolation_points": 2,
            "moving_block_size": [1.0, 2.0, 3.0],
            "trajectory.ctrl_points_min": 6,
            "execution.enabled": True,
            "default_named_configuration_joint_names": ["j1", "j2"],
            "wall_plan_file": "/tmp/custom_wall_plan.yaml",
            "default_wall_plan_name": "demo_plan",
            "wall_plan_frame_id": "map",
        }
    )

    cfg = declare_and_load_config(node)

    assert cfg.default_geometric_method == "SLSQP"
    assert cfg.path_interpolation_points == 2
    assert cfg.moving_block_size == (1.0, 2.0, 3.0)
    assert cfg.traj_ctrl_pts_min == 6
    assert cfg.execution_enabled is True
    assert cfg.default_named_joint_names == ["j1", "j2"]
    assert cfg.wall_plan_file == "/tmp/custom_wall_plan.yaml"
    assert cfg.default_wall_plan_name == "demo_plan"
    assert cfg.wall_plan_frame_id == "map"
