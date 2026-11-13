"""What the grip server refuses to plan from."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import rclpy
from rclpy.time import Time
from sensor_msgs.msg import JointState

from timber_crane_planning_interfaces.srv import CalcGripMovement

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "scripts"))
sys.path.insert(0, str(PKG))

from grip_traj_server_simple import (  # noqa: E402
    CONTROLLED_JOINT_NAMES,
    MAX_INPUT_AGE_S,
    GripTrajServerSimple,
)


@pytest.fixture
def server():
    rclpy.init()
    node = GripTrajServerSimple()
    yield node
    node.destroy_node()
    rclpy.shutdown()


def joint_state(node, age_s):
    message = JointState()
    stamp_ns = node.get_clock().now().nanoseconds - int(age_s * 1e9)
    message.header.stamp = Time(nanoseconds=stamp_ns).to_msg()
    message.name = list(CONTROLLED_JOINT_NAMES)
    message.position = [0.0] * len(message.name)
    return message


def test_a_stale_joint_state_is_not_planned_from(server):
    """A pose from before the crane moved plans a descent into wherever it was."""
    errors = []
    server.get_logger().error = errors.append
    server._on_joint_state(joint_state(server, MAX_INPUT_AGE_S + 0.1))

    response = server._handle_request(
        CalcGripMovement.Request(), CalcGripMovement.Response()
    )

    assert response.success == 0
    assert errors and "s old" in errors[-1]
