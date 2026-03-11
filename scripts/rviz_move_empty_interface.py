#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as NavPath
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.time import Time
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformException, TransformListener

from concrete_block_motion_planning.srv import (
    ComputeTrajectory,
    ExecuteTrajectory,
    PlanAndComputeTrajectory,
    PlanGeometricPath,
)


class RvizMoveEmptyInterface(Node):
    def __init__(self) -> None:
        super().__init__("rviz_move_empty_interface")

        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("tool_frame", "K8_tool_center_point")
        self.declare_parameter(
            "plan_service",
            "/concrete_block_motion_planning_node/plan_geometric_path",
        )
        self.declare_parameter(
            "compute_service",
            "/concrete_block_motion_planning_node/compute_trajectory",
        )
        self.declare_parameter(
            "combined_service",
            "/concrete_block_motion_planning_node/plan_and_compute_trajectory",
        )
        self.declare_parameter(
            "execute_service",
            "/concrete_block_motion_planning_node/execute_trajectory",
        )
        self.declare_parameter("use_combined_service", True)
        self.declare_parameter("geometric_method", "")
        self.declare_parameter("trajectory_method", "")
        self.declare_parameter("dry_run", False)
        self.declare_parameter("enable_topic", "/cb_move_empty/enable")
        self.declare_parameter("require_enable", True)
        self.declare_parameter("fallback_direct_path_on_geometric_failure", True)

        self._goal_topic = str(self.get_parameter("goal_topic").value)
        self._world_frame = str(self.get_parameter("world_frame").value)
        self._tool_frame = str(self.get_parameter("tool_frame").value)
        self._geometric_method = str(self.get_parameter("geometric_method").value)
        self._trajectory_method = str(self.get_parameter("trajectory_method").value)
        self._use_combined_service = bool(self.get_parameter("use_combined_service").value)
        self._dry_run = bool(self.get_parameter("dry_run").value)
        self._enable_topic = str(self.get_parameter("enable_topic").value)
        self._require_enable = bool(self.get_parameter("require_enable").value)
        self._fallback_direct_path_on_geometric_failure = bool(
            self.get_parameter("fallback_direct_path_on_geometric_failure").value
        )
        self._enabled = not self._require_enable

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._busy = False
        self._pending_start_pose: Optional[PoseStamped] = None
        self._pending_goal_pose: Optional[PoseStamped] = None

        self._plan_client = self.create_client(
            PlanGeometricPath,
            str(self.get_parameter("plan_service").value),
        )
        self._compute_client = self.create_client(
            ComputeTrajectory,
            str(self.get_parameter("compute_service").value),
        )
        self._combined_client = self.create_client(
            PlanAndComputeTrajectory,
            str(self.get_parameter("combined_service").value),
        )
        self._execute_client = self.create_client(
            ExecuteTrajectory,
            str(self.get_parameter("execute_service").value),
        )

        qos = QoSProfile(depth=10)
        self._goal_sub = self.create_subscription(
            PoseStamped,
            self._goal_topic,
            self._on_goal,
            qos,
        )
        self._enable_sub = self.create_subscription(
            Bool,
            self._enable_topic,
            self._on_enable,
            qos,
        )

        self.get_logger().info(
            "RViz move-empty interface ready | "
            f"goal_topic={self._goal_topic} | world_frame={self._world_frame} | "
            f"tool_frame={self._tool_frame} | dry_run={self._dry_run} | "
            f"use_combined_service={self._use_combined_service} | "
            f"require_enable={self._require_enable} | enable_topic={self._enable_topic} | "
            f"fallback_direct_path={self._fallback_direct_path_on_geometric_failure}"
        )

    def _on_enable(self, msg: Bool) -> None:
        self._enabled = bool(msg.data)
        state = "ENABLED" if self._enabled else "DISABLED"
        self.get_logger().info(f"Move-empty interface {state}.")

    def _lookup_tool_pose(self) -> Optional[PoseStamped]:
        try:
            tf = self._tf_buffer.lookup_transform(
                self._world_frame,
                self._tool_frame,
                Time(),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"Failed to lookup start pose transform {self._world_frame}->{self._tool_frame}: {exc}"
            )
            return None

        pose = PoseStamped()
        pose.header.frame_id = self._world_frame
        pose.pose.position.x = float(tf.transform.translation.x)
        pose.pose.position.y = float(tf.transform.translation.y)
        pose.pose.position.z = float(tf.transform.translation.z)
        pose.pose.orientation = tf.transform.rotation
        return pose

    def _on_goal(self, msg: PoseStamped) -> None:
        if self._require_enable and not self._enabled:
            self.get_logger().debug("Ignoring goal because move-empty interface is disabled.")
            return
        if self._busy:
            self.get_logger().warn("Ignoring goal while a move-empty request is still running.")
            return

        start_pose = self._lookup_tool_pose()
        if start_pose is None:
            return

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = self._world_frame
        goal_pose.pose = msg.pose

        if self._use_combined_service:
            if not self._combined_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn("Combined plan+compute service unavailable.")
                return
        else:
            if not self._plan_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn("Plan service unavailable.")
                return
            if not self._compute_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn("Compute trajectory service unavailable.")
                return
        if not self._execute_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Execute trajectory service unavailable.")
            return

        self._pending_start_pose = start_pose
        self._pending_goal_pose = goal_pose
        self._busy = True
        self.get_logger().info(
            f"Move-empty request: start=({start_pose.pose.position.x:.2f},"
            f"{start_pose.pose.position.y:.2f},{start_pose.pose.position.z:.2f}) "
            f"goal=({goal_pose.pose.position.x:.2f},{goal_pose.pose.position.y:.2f},"
            f"{goal_pose.pose.position.z:.2f})"
        )
        if self._use_combined_service:
            req = PlanAndComputeTrajectory.Request()
            req.start_pose = start_pose
            req.goal_pose = goal_pose
            req.use_world_model = True
            req.validate_dynamics = True
            req.geometric_timeout_s = 5.0
            req.trajectory_timeout_s = 10.0
            if self._geometric_method:
                req.geometric_method = self._geometric_method
            if self._trajectory_method:
                req.trajectory_method = self._trajectory_method
            future = self._combined_client.call_async(req)
            future.add_done_callback(self._on_combined_done)
            return

        req = PlanGeometricPath.Request()
        req.start_pose = start_pose
        req.goal_pose = goal_pose
        if self._geometric_method:
            req.method = self._geometric_method
        future = self._plan_client.call_async(req)
        future.add_done_callback(self._on_plan_done)

    def _on_combined_done(self, future) -> None:
        try:
            res = future.result()
        except Exception as exc:  # pragma: no cover - runtime safety
            self._busy = False
            self.get_logger().error(f"Combined plan+compute call failed: {exc}")
            return
        if res is None or not res.success:
            msg = "<no response>" if res is None else res.message
            if self._should_fallback_to_direct_path(msg):
                self.get_logger().warn(
                    f"Combined service failed ({msg}); falling back to direct-path trajectory compute."
                )
                self._call_compute_direct_path()
                return
            self._busy = False
            self.get_logger().warn(f"Combined service failed: {msg}")
            return

        req = ExecuteTrajectory.Request()
        req.trajectory_id = res.trajectory_id
        req.dry_run = self._dry_run
        future3 = self._execute_client.call_async(req)
        future3.add_done_callback(self._on_execute_done)

    def _on_plan_done(self, future) -> None:
        try:
            res = future.result()
        except Exception as exc:  # pragma: no cover - runtime safety
            self._busy = False
            self.get_logger().error(f"Plan call failed: {exc}")
            return
        if res is None or not res.success:
            msg = "<no response>" if res is None else res.message
            if self._should_fallback_to_direct_path(msg):
                self.get_logger().warn(
                    f"Planning failed ({msg}); falling back to direct-path trajectory compute."
                )
                self._call_compute_direct_path()
                return
            self._busy = False
            self.get_logger().warn(f"Planning failed: {msg}")
            return

        req = ComputeTrajectory.Request()
        req.geometric_plan_id = res.geometric_plan_id
        if self._trajectory_method:
            req.method = self._trajectory_method
        req.validate_dynamics = True

        self._call_compute(req)

    def _call_compute_direct_path(self) -> None:
        if self._pending_start_pose is None or self._pending_goal_pose is None:
            self._busy = False
            self.get_logger().warn("Cannot fallback to direct path: missing pending poses.")
            return

        path = NavPath()
        path.header.frame_id = self._world_frame
        path.poses = [self._pending_start_pose, self._pending_goal_pose]

        req = ComputeTrajectory.Request()
        req.use_direct_path = True
        req.direct_path = path
        if self._trajectory_method:
            req.method = self._trajectory_method
        req.validate_dynamics = True

        self._call_compute(req)

    def _call_compute(self, req: ComputeTrajectory.Request) -> None:
        future2 = self._compute_client.call_async(req)
        future2.add_done_callback(self._on_compute_done)

    def _on_compute_done(self, future) -> None:
        try:
            res = future.result()
        except Exception as exc:  # pragma: no cover - runtime safety
            self._busy = False
            self.get_logger().error(f"Compute trajectory call failed: {exc}")
            return
        if res is None or not res.success:
            self._busy = False
            msg = "<no response>" if res is None else res.message
            self.get_logger().warn(f"Trajectory compute failed: {msg}")
            return

        req = ExecuteTrajectory.Request()
        req.trajectory_id = res.trajectory_id
        req.dry_run = self._dry_run
        future3 = self._execute_client.call_async(req)
        future3.add_done_callback(self._on_execute_done)

    def _on_execute_done(self, future) -> None:
        self._busy = False
        try:
            res = future.result()
        except Exception as exc:  # pragma: no cover - runtime safety
            self.get_logger().error(f"Execute trajectory call failed: {exc}")
            return
        if res is None:
            self.get_logger().warn("Execute trajectory returned no response.")
            return
        if res.success:
            self.get_logger().info(f"Move-empty execution started: {res.message}")
            return
        self.get_logger().warn(f"Move-empty execution failed: {res.message}")

    def _should_fallback_to_direct_path(self, message: str) -> bool:
        if not self._fallback_direct_path_on_geometric_failure:
            return False
        msg = message.lower()
        return ("geometric backend unavailable" in msg) or ("missing modules: fcl" in msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RvizMoveEmptyInterface()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
