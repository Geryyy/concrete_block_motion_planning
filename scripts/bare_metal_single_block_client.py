#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
import time

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener
from tf_transformations import quaternion_from_euler

from concrete_block_motion_planning.srv import (
    ExecuteTrajectory,
    GetNextAssemblyTask,
    PlanAndComputeTrajectory,
)


class BareMetalSingleBlockClient(Node):
    def __init__(self) -> None:
        super().__init__("bare_metal_single_block_client")

        self.declare_parameter(
            "task_service", "/concrete_block_motion_planning_node/get_next_assembly_task"
        )
        self.declare_parameter(
            "plan_service", "/concrete_block_motion_planning_node/plan_and_compute_trajectory"
        )
        self.declare_parameter(
            "execute_service", "/concrete_block_motion_planning_node/execute_trajectory"
        )
        self.declare_parameter("wall_plan_name", "basic_interlocking_3_2")
        self.declare_parameter("reset_plan", True)
        self.declare_parameter("source_frame", "K8_tool_center_point")
        self.declare_parameter("target_frame", "world")
        self.declare_parameter("use_tf_start_pose", True)
        self.declare_parameter("start_pose_xyz", [-10.98, -3.71, 2.15])
        self.declare_parameter("start_pose_yaw_deg", -180.0)
        self.declare_parameter("approach_offset_m", 2.0)
        self.declare_parameter("use_world_model", True)
        self.declare_parameter("geometric_method", "POWELL")
        self.declare_parameter("trajectory_method", "TOPPRA_PATH_FOLLOWING")
        self.declare_parameter("execute", True)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("startup_delay_s", 2.0)

        self._task_client = self.create_client(
            GetNextAssemblyTask, str(self.get_parameter("task_service").value)
        )
        self._plan_client = self.create_client(
            PlanAndComputeTrajectory, str(self.get_parameter("plan_service").value)
        )
        self._execute_client = self.create_client(
            ExecuteTrajectory, str(self.get_parameter("execute_service").value)
        )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)

    def _wait_for_service(self, client, timeout_s: float, label: str) -> bool:
        if client.wait_for_service(timeout_sec=timeout_s):
            return True
        self.get_logger().error(f"Timed out waiting for service '{label}'.")
        return False

    def _call(self, client, request, timeout_s: float, label: str):
        future = client.call_async(request)
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() <= deadline:
            if future.done():
                return future.result()
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().error(f"Timed out waiting for service response from '{label}'.")
        return None

    def _lookup_start_pose(self, source_frame: str, target_frame: str) -> PoseStamped | None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() <= deadline:
            try:
                tf = self._tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    rclpy.time.Time(),
                )
                pose = PoseStamped()
                pose.header = tf.header
                pose.pose.position.x = tf.transform.translation.x
                pose.pose.position.y = tf.transform.translation.y
                pose.pose.position.z = tf.transform.translation.z
                pose.pose.orientation = tf.transform.rotation
                return pose
            except TransformException:
                rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().error(
            f"Timed out looking up transform {target_frame} <- {source_frame}."
        )
        return None

    @staticmethod
    def _apply_approach_offset(
        start_pose: PoseStamped,
        goal_pose: PoseStamped,
        approach_offset_m: float,
    ) -> PoseStamped:
        if approach_offset_m <= 1e-6:
            return goal_pose
        dx = goal_pose.pose.position.x - start_pose.pose.position.x
        dy = goal_pose.pose.position.y - start_pose.pose.position.y
        dz = goal_pose.pose.position.z - start_pose.pose.position.z
        norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        if norm <= 1e-6:
            return goal_pose
        adjusted = PoseStamped()
        adjusted.header = goal_pose.header
        adjusted.pose = goal_pose.pose
        adjusted.pose.position.x -= approach_offset_m * (dx / norm)
        adjusted.pose.position.y -= approach_offset_m * (dy / norm)
        adjusted.pose.position.z -= approach_offset_m * (dz / norm)
        return adjusted

    def _configured_start_pose(self, target_frame: str) -> PoseStamped:
        raw_xyz = self.get_parameter("start_pose_xyz").value
        if isinstance(raw_xyz, str):
            xyz = list(yaml.safe_load(raw_xyz))
        else:
            xyz = list(raw_xyz)
        yaw_deg = float(self.get_parameter("start_pose_yaw_deg").value)
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, math.radians(yaw_deg))
        pose = PoseStamped()
        pose.header.frame_id = target_frame
        pose.pose.position.x = float(xyz[0])
        pose.pose.position.y = float(xyz[1])
        pose.pose.position.z = float(xyz[2])
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)
        return pose

    def run(self) -> int:
        startup_delay_s = float(self.get_parameter("startup_delay_s").value)
        if startup_delay_s > 0.0:
            self.get_logger().info(
                f"Waiting {startup_delay_s:.1f}s before bare-metal service call."
            )
            end = time.monotonic() + startup_delay_s
            while time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=0.1)

        if not self._wait_for_service(self._task_client, 10.0, "get_next_assembly_task"):
            return 2
        if not self._wait_for_service(self._plan_client, 10.0, "plan_and_compute_trajectory"):
            return 3
        if bool(self.get_parameter("execute").value):
            if not self._wait_for_service(self._execute_client, 10.0, "execute_trajectory"):
                return 4

        task_req = GetNextAssemblyTask.Request()
        task_req.wall_plan_name = str(self.get_parameter("wall_plan_name").value)
        task_req.reset_plan = bool(self.get_parameter("reset_plan").value)
        task_res = self._call(self._task_client, task_req, 10.0, "get_next_assembly_task")
        if task_res is None:
            return 5
        if not task_res.success or not task_res.has_task:
            self.get_logger().error(
                f"GetNextAssemblyTask failed | success={task_res.success} has_task={task_res.has_task} message={task_res.message}"
            )
            return 6

        target_frame = str(self.get_parameter("target_frame").value)
        source_frame = str(self.get_parameter("source_frame").value)
        if bool(self.get_parameter("use_tf_start_pose").value):
            start_pose = self._lookup_start_pose(source_frame, target_frame)
            if start_pose is None:
                return 7
        else:
            start_pose = self._configured_start_pose(target_frame)
            self.get_logger().info(
                "Using configured start pose | "
                f"frame={target_frame} "
                f"xyz=({start_pose.pose.position.x:.2f},{start_pose.pose.position.y:.2f},{start_pose.pose.position.z:.2f})"
            )

        goal_pose = self._apply_approach_offset(
            start_pose,
            task_res.pickup_pose,
            float(self.get_parameter("approach_offset_m").value),
        )

        plan_req = PlanAndComputeTrajectory.Request()
        plan_req.start_pose = start_pose
        plan_req.goal_pose = goal_pose
        plan_req.target_block_id = task_res.target_block_id
        plan_req.reference_block_id = task_res.reference_block_id
        plan_req.use_world_model = bool(self.get_parameter("use_world_model").value)
        plan_req.geometric_method = str(self.get_parameter("geometric_method").value)
        plan_req.trajectory_method = str(self.get_parameter("trajectory_method").value)
        plan_req.validate_dynamics = False
        plan_res = self._call(self._plan_client, plan_req, 60.0, "plan_and_compute_trajectory")
        if plan_res is None:
            return 8

        self.get_logger().info(
            "PlanAndComputeTrajectory | "
            f"success={plan_res.success} trajectory_id={plan_res.trajectory_id} message={plan_res.message}"
        )
        if not plan_res.success:
            return 9

        if not bool(self.get_parameter("execute").value):
            return 0

        exec_req = ExecuteTrajectory.Request()
        exec_req.trajectory_id = plan_res.trajectory_id
        exec_req.dry_run = bool(self.get_parameter("dry_run").value)
        exec_res = self._call(self._execute_client, exec_req, 60.0, "execute_trajectory")
        if exec_res is None:
            return 10
        self.get_logger().info(
            f"ExecuteTrajectory | success={exec_res.success} message={exec_res.message}"
        )
        return 0 if exec_res.success else 11


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BareMetalSingleBlockClient()
    try:
        code = node.run()
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main(sys.argv)
