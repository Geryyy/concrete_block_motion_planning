#!/usr/bin/env python3
"""Lightweight wall plan server.

Serves GetNextAssemblyTask and SetBlockTaskStatus without the heavy
concrete_block_motion_planning dependencies (acados, motion_planning, etc.).
Reads wall_plans.yaml and tracks progress through the plan sequence.
"""

import math
import yaml

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from concrete_block_motion_planning.srv import GetNextAssemblyTask
from concrete_block_perception.srv import SetBlockTaskStatus
from geometry_msgs.msg import PoseStamped


def yaw_to_quaternion(yaw_rad: float):
    """Convert yaw angle to quaternion (x, y, z, w)."""
    return (0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0))


class WallPlanServer(Node):
    def __init__(self):
        super().__init__("concrete_block_motion_planning_node")

        # Z offset from block center to crane tip grip target.
        # The grip planner expects the crane tip position, not the block CoG.
        # For a 0.6m tall block: tip is ~0.5m above block center.
        self.declare_parameter("grip_z_offset", 0.5)
        self._grip_z_offset = self.get_parameter("grip_z_offset").value
        self.get_logger().info(f"Grip Z offset: {self._grip_z_offset}m")

        # Load wall plans
        pkg_share = get_package_share_directory("concrete_block_motion_planning")
        wall_plan_path = f"{pkg_share}/motion_planning/data/wall_plans.yaml"
        self.get_logger().info(f"Loading wall plans from {wall_plan_path}")

        with open(wall_plan_path) as f:
            data = yaml.safe_load(f)

        self._wall_plans = {}
        self._progress = {}
        for plan_name, plan_data in data.get("wall_plans", {}).items():
            tasks = self._resolve_plan(plan_name, plan_data)
            self._wall_plans[plan_name] = tasks
            self._progress[plan_name] = 0
            self.get_logger().info(
                f"Loaded wall plan '{plan_name}' with {len(tasks)} tasks"
            )

        # Services
        self.create_service(
            GetNextAssemblyTask,
            "~/get_next_assembly_task",
            self._handle_get_next_task,
        )
        self.create_service(
            SetBlockTaskStatus,
            "~/set_block_task_status",
            self._handle_set_status,
        )
        self.get_logger().info("Wall plan server ready")

    def _resolve_plan(self, plan_name, plan_data):
        """Resolve plan sequence into list of (block_id, x, y, z, yaw_rad)."""
        resolved = {}
        tasks = []
        for idx, item in enumerate(plan_data.get("sequence", [])):
            block_id = item["id"]
            yaw_rad = math.radians(item.get("yaw_deg", 0.0))

            if "absolute_position" in item:
                pos = item["absolute_position"]
            elif "relative_to" in item:
                ref = item["relative_to"]
                if ref not in resolved:
                    self.get_logger().error(
                        f"Plan '{plan_name}': block '{block_id}' references "
                        f"unknown block '{ref}'"
                    )
                    continue
                ref_pos = resolved[ref]
                offset = item.get("offset", [0, 0, 0])
                pos = [ref_pos[i] + offset[i] for i in range(3)]
            else:
                self.get_logger().error(
                    f"Plan '{plan_name}': block '{block_id}' has no position"
                )
                continue

            resolved[block_id] = pos
            task_id = f"{plan_name}_{idx + 1:02d}_{block_id}"
            tasks.append((task_id, block_id, pos[0], pos[1], pos[2], yaw_rad))
        return tasks

    def _make_pose(self, x, y, z, yaw_rad, frame="K0_mounting_base"):
        pose = PoseStamped()
        pose.header.frame_id = frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        qx, qy, qz, qw = yaw_to_quaternion(yaw_rad)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    def _handle_get_next_task(self, request, response):
        plan_name = request.wall_plan_name.lower()

        if request.reset_plan:
            self._progress[plan_name] = 0
            self.get_logger().info(f"Reset progress for plan '{plan_name}'")

        if plan_name not in self._wall_plans:
            response.success = False
            response.has_task = False
            response.message = f"Unknown wall plan: '{plan_name}'"
            self.get_logger().error(response.message)
            return response

        tasks = self._wall_plans[plan_name]
        idx = self._progress.get(plan_name, 0)

        if idx >= len(tasks):
            response.success = True
            response.has_task = False
            response.message = f"Plan '{plan_name}' complete ({len(tasks)} tasks done)"
            self.get_logger().info(response.message)
            return response

        task_id, block_id, x, y, z, yaw = tasks[idx]
        self._progress[plan_name] = idx + 1

        response.success = True
        response.has_task = True
        response.task_id = task_id
        response.target_block_id = block_id
        response.reference_block_id = ""
        # pickup_pose: crane tip target = block CoG + grip_z_offset
        grip_z = z + self._grip_z_offset
        response.pickup_pose = self._make_pose(x, y, grip_z, yaw)
        response.target_pose = self._make_pose(x, y, grip_z, yaw)
        response.reference_pose = PoseStamped()
        response.message = f"Task {idx + 1}/{len(tasks)}: {block_id}"

        self.get_logger().info(
            f"GetNextAssemblyTask | plan={plan_name} task={task_id} "
            f"block={block_id} pos=({x:.2f}, {y:.2f}, {z:.2f}) yaw={math.degrees(yaw):.1f}deg"
        )
        return response

    def _handle_set_status(self, request, response):
        self.get_logger().info(
            f"SetBlockTaskStatus | block={request.block_id} status={request.task_status}"
        )
        response.success = True
        response.message = "OK"
        return response


def main():
    rclpy.init()
    node = WallPlanServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
