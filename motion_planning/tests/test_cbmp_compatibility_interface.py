from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest
import rclpy
from geometry_msgs.msg import Quaternion
from rclpy.parameter import Parameter
from trajectory_msgs.msg import JointTrajectory


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cbmp.compatibility import PZS100_A2B_JOINT_ORDER, translate_calc_movement_request
from cbmp.node import ConcreteBlockMotionPlanningNode
from timber_crane_planning_interfaces.srv import CalcMovement


def test_translate_calc_movement_request_detects_zero_q0() -> None:
    req = CalcMovement.Request()
    req.y_n.x = 1.0
    req.y_n.y = 2.0
    req.y_n.z = 3.0
    req.phi_tool_n = 0.5
    req.q0 = [0.0] * 8
    req.slow_down = 0.8
    req.t_end = 5.0

    translated = translate_calc_movement_request(
        req,
        frame_id="K0_mounting_base",
        yaw_to_quaternion=lambda yaw: Quaternion(x=0.0, y=0.0, z=yaw, w=1.0),
    )

    assert translated.use_current_state is True
    assert translated.start_joint_positions is None
    assert translated.goal_pose.header.frame_id == "K0_mounting_base"
    assert translated.goal_pose.pose.position.x == pytest.approx(1.0)
    assert translated.goal_pose.pose.position.y == pytest.approx(2.0)
    assert translated.goal_pose.pose.position.z == pytest.approx(3.0)
    assert translated.slow_down == pytest.approx(0.8)
    assert translated.t_end == pytest.approx(5.0)


def test_translate_calc_movement_request_keeps_explicit_q0() -> None:
    req = CalcMovement.Request()
    req.q0 = [0.1] + [0.0] * 7

    translated = translate_calc_movement_request(
        req,
        frame_id="K0_mounting_base",
        yaw_to_quaternion=lambda yaw: Quaternion(x=0.0, y=0.0, z=yaw, w=1.0),
    )

    assert translated.use_current_state is False
    assert translated.start_joint_positions == pytest.approx(tuple(req.q0))


@pytest.fixture(scope="module")
def ros_runtime():
    rclpy.init()

    server_node = ConcreteBlockMotionPlanningNode(
        parameter_overrides=[
            Parameter("planner.backend", value="concrete"),
            Parameter("compatibility.a2b_service_enabled", value=True),
            Parameter("compatibility.a2b_service_name", value="a2b_movement"),
        ]
    )
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    executor.add_node(server_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    client_node = rclpy.create_node("cbmp_compatibility_test_client")

    try:
        yield server_node, client_node
    finally:
        executor.remove_node(server_node)
        server_node.destroy_node()
        client_node.destroy_node()
        executor.shutdown(timeout_sec=2.0)
        spin_thread.join(timeout=2.0)
        rclpy.shutdown()


def _call_service(client_node, client, request, timeout_sec: float = 5.0):
    future = client.call_async(request)
    rclpy.spin_until_future_complete(client_node, future, timeout_sec=timeout_sec)
    assert future.done(), "Service call timed out"
    return future.result()


def test_compatibility_a2b_service_is_available(ros_runtime) -> None:
    _, client_node = ros_runtime
    client = client_node.create_client(CalcMovement, "/a2b_movement")
    assert client.wait_for_service(timeout_sec=3.0)


def test_concrete_compatibility_a2b_returns_explicit_stub_failure(ros_runtime) -> None:
    _, client_node = ros_runtime
    client = client_node.create_client(CalcMovement, "/a2b_movement")
    assert client.wait_for_service(timeout_sec=3.0)

    req = CalcMovement.Request()
    req.carries_log = False
    res = _call_service(client_node, client, req)
    assert res is not None
    assert res.success is False
    assert isinstance(res.trajectory, JointTrajectory)
    assert list(res.trajectory.joint_names) == list(PZS100_A2B_JOINT_ORDER)
    assert list(res.tcp_path) == []


def test_concrete_compatibility_a2b_rejects_payload_requests(ros_runtime) -> None:
    _, client_node = ros_runtime
    client = client_node.create_client(CalcMovement, "/a2b_movement")
    assert client.wait_for_service(timeout_sec=3.0)

    req = CalcMovement.Request()
    req.carries_log = True
    res = _call_service(client_node, client, req)
    assert res is not None
    assert res.success is False
    assert list(res.trajectory.joint_names) == list(PZS100_A2B_JOINT_ORDER)
