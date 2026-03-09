from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cbmp.node import ConcreteBlockMotionPlanningNode
from concrete_block_motion_planning.srv import (
    ExecuteNamedConfiguration,
    ExecuteTrajectory,
    GetNextAssemblyTask,
    PlanGeometricPath,
)


@pytest.fixture(scope="module")
def ros_runtime():
    rclpy.init()

    server_node = ConcreteBlockMotionPlanningNode()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    executor.add_node(server_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    client_node = rclpy.create_node("cbmp_integration_test_client")

    try:
        yield server_node, client_node
    finally:
        executor.remove_node(server_node)
        server_node.destroy_node()
        client_node.destroy_node()
        executor.shutdown(timeout_sec=2.0)
        spin_thread.join(timeout=2.0)
        rclpy.shutdown()


def _service_name(node_name: str, suffix: str) -> str:
    return f"/{node_name}/{suffix}"


def _call_service(client_node, client, request, timeout_sec: float = 5.0):
    future = client.call_async(request)
    rclpy.spin_until_future_complete(client_node, future, timeout_sec=timeout_sec)
    assert future.done(), "Service call timed out"
    return future.result()


def test_clean_service_endpoints_are_available(ros_runtime) -> None:
    server_node, client_node = ros_runtime
    node_name = server_node.get_name()

    service_specs = [
        (PlanGeometricPath, "plan_geometric_path"),
        (ExecuteTrajectory, "execute_trajectory"),
        (ExecuteNamedConfiguration, "execute_named_configuration"),
        (GetNextAssemblyTask, "get_next_assembly_task"),
    ]

    for srv_type, suffix in service_specs:
        client = client_node.create_client(srv_type, _service_name(node_name, suffix))
        assert client.wait_for_service(timeout_sec=3.0), f"Service '{suffix}' not available"


def test_execute_trajectory_unknown_id_returns_failure(ros_runtime) -> None:
    server_node, client_node = ros_runtime
    client = client_node.create_client(
        ExecuteTrajectory,
        _service_name(server_node.get_name(), "execute_trajectory"),
    )
    assert client.wait_for_service(timeout_sec=3.0)

    req = ExecuteTrajectory.Request()
    req.trajectory_id = "traj_does_not_exist"
    req.dry_run = True

    res = _call_service(client_node, client, req)
    assert res is not None
    assert res.success is False
    assert "Unknown trajectory_id" in res.message


def test_named_configuration_and_wall_task_error_paths(ros_runtime) -> None:
    server_node, client_node = ros_runtime

    named_client = client_node.create_client(
        ExecuteNamedConfiguration,
        _service_name(server_node.get_name(), "execute_named_configuration"),
    )
    wall_client = client_node.create_client(
        GetNextAssemblyTask,
        _service_name(server_node.get_name(), "get_next_assembly_task"),
    )

    assert named_client.wait_for_service(timeout_sec=3.0)
    assert wall_client.wait_for_service(timeout_sec=3.0)

    named_req = ExecuteNamedConfiguration.Request()
    named_req.configuration_name = "unknown_cfg"
    named_req.dry_run = True
    named_res = _call_service(client_node, named_client, named_req)
    assert named_res is not None
    assert named_res.success is False
    assert "Unknown named configuration" in named_res.message

    wall_req = GetNextAssemblyTask.Request()
    wall_req.wall_plan_name = "unknown_plan"
    wall_req.reset_plan = False
    wall_res = _call_service(client_node, wall_client, wall_req)
    assert wall_res is not None
    assert wall_res.success is False
    assert wall_res.has_task is False
    assert "Unknown wall plan" in wall_res.message


def test_plan_geometric_service_round_trip(ros_runtime) -> None:
    server_node, client_node = ros_runtime
    plan_client = client_node.create_client(
        PlanGeometricPath,
        _service_name(server_node.get_name(), "plan_geometric_path"),
    )
    assert plan_client.wait_for_service(timeout_sec=3.0)

    req = PlanGeometricPath.Request()
    req.start_pose = PoseStamped()
    req.goal_pose = PoseStamped()
    req.start_pose.header.frame_id = "world"
    req.goal_pose.header.frame_id = "world"
    req.start_pose.pose.orientation.w = 1.0
    req.goal_pose.pose.orientation.w = 1.0

    res = _call_service(client_node, plan_client, req, timeout_sec=10.0)
    assert res is not None
    assert res.geometric_plan_id.startswith("geo_")
    assert isinstance(res.message, str)


def test_execute_named_configuration_happy_path_when_seeded(ros_runtime) -> None:
    server_node, client_node = ros_runtime

    # Seed one named configuration directly into runtime state for deterministic integration test.
    traj = JointTrajectory()
    traj.joint_names = ["joint_1", "joint_2"]
    point = JointTrajectoryPoint()
    point.positions = [0.0, 0.0]
    point.velocities = [0.0, 0.0]
    traj.points = [point]
    server_node._named_configurations["home"] = traj

    named_client = client_node.create_client(
        ExecuteNamedConfiguration,
        _service_name(server_node.get_name(), "execute_named_configuration"),
    )
    assert named_client.wait_for_service(timeout_sec=3.0)

    req = ExecuteNamedConfiguration.Request()
    req.configuration_name = "home"
    req.dry_run = True

    res = _call_service(client_node, named_client, req)
    assert res is not None
    assert res.success is True
    assert res.trajectory_id.startswith("named_home_")
