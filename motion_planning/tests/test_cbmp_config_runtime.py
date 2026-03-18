from __future__ import annotations

import sys
from pathlib import Path

import rclpy


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cbmp.config import declare_and_load_config


def test_declare_and_load_config_reads_current_runtime_shape() -> None:
    rclpy.init()
    try:
        node = rclpy.create_node(
            "cbmp_config_runtime_test",
            parameter_overrides=[
                rclpy.parameter.Parameter(
                    "default_trajectory_method",
                    value="ACADOS_PATH_FOLLOWING_STABLE",
                ),
                rclpy.parameter.Parameter(
                    "execution.enabled",
                    value=True,
                ),
                rclpy.parameter.Parameter(
                    "execution.backend",
                    value="action",
                ),
                rclpy.parameter.Parameter(
                    "named_configurations_file",
                    value="/tmp/named_configurations.yaml",
                ),
                rclpy.parameter.Parameter(
                    "wall_plan_file",
                    value="/tmp/wall_plans.yaml",
                ),
            ],
        )
        cfg = declare_and_load_config(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    assert cfg.default_geometric_method == "POWELL"
    assert cfg.default_trajectory_method == "ACADOS_PATH_FOLLOWING_STABLE"
    assert cfg.execution_enabled is True
    assert cfg.execution_backend == "action"
    assert cfg.named_configurations_file == "/tmp/named_configurations.yaml"
    assert cfg.wall_plan_file == "/tmp/wall_plans.yaml"
    assert cfg.moving_block_size == (0.6, 0.9, 0.6)
