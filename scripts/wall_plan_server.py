#!/usr/bin/env python3
"""Lightweight wall plan server.

Serves GetNextAssemblyTask and SetBlockTaskStatus.
Reads wall_plans.yaml for target (place) positions and plan sequence.
Queries the world model for actual block pickup poses at runtime;
falls back to YAML positions if the world model is unavailable.
"""

import math
import yaml

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from concrete_block_motion_planning.srv import GetNextAssemblyTask
from concrete_block_perception.srv import GetCoarseBlocks, SetBlockTaskStatus
from geometry_msgs.msg import PoseStamped


def yaw_to_quaternion(yaw_rad: float):
    """Convert yaw angle to quaternion (x, y, z, w)."""
    return (0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0))


def quaternion_to_yaw(q) -> float:
    """Extract yaw from quaternion (geometry_msgs/Quaternion)."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class WallPlanServer(Node):
    def __init__(self):
        super().__init__("concrete_block_motion_planning_node")

        # NOTE: No Z offset applied here. The wall plan sends raw block CoG
        # positions. The grip trajectory server handles the full offset from
        # block center to K8_tool_center_point via its tcp_z_offset parameter.

        # Lateral offset for angled placement approach (meters).
        # The approach position is shifted along the block's yaw direction
        # by this amount, so the descend slides in at an angle rather than
        # dropping straight down. Helps contact with existing blocks.
        # Positive = approach from the block's +Y (lengthwise) side.
        # Set to 0.0 for pure vertical placement.
        self.declare_parameter("place_approach_offset", 0.3)
        self._place_approach_offset = self.get_parameter("place_approach_offset").value

        self.declare_parameter(
            "world_model_service",
            "/block_world_model_node/get_coarse_blocks",
        )
        self.declare_parameter("world_model_timeout_s", 2.0)

        self._wm_service_name = self.get_parameter("world_model_service").value
        self._wm_timeout = self.get_parameter("world_model_timeout_s").value

        # World model client (for live block poses)
        self._wm_client = self.create_client(
            GetCoarseBlocks, self._wm_service_name
        )

        # Load wall plans
        self.declare_parameter("wall_plans_file", "")
        wall_plan_path = self.get_parameter("wall_plans_file").value
        if not wall_plan_path:
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

    def _query_block_pose(self, block_id: str):
        """Query world model for a block's pose. Returns (x, y, z, yaw) or None."""
        if not self._wm_client.service_is_ready():
            self.get_logger().warn(
                f"World model service '{self._wm_service_name}' not available, "
                "using YAML fallback"
            )
            return None

        req = GetCoarseBlocks.Request()
        req.force_refresh = False
        req.timeout_s = float(self._wm_timeout)
        future = self._wm_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self._wm_timeout + 1.0)

        if not future.done() or future.result() is None:
            self.get_logger().warn("World model query timed out, using YAML fallback")
            return None

        result = future.result()
        if not result.success:
            self.get_logger().warn(
                f"World model query failed: {result.message}, using YAML fallback"
            )
            return None

        for block in result.blocks.blocks:
            if block.id == block_id:
                p = block.pose.position
                yaw = quaternion_to_yaw(block.pose.orientation)
                self.get_logger().info(
                    f"World model pose for '{block_id}': "
                    f"({p.x:.2f}, {p.y:.2f}, {p.z:.2f}) yaw={math.degrees(yaw):.1f}deg"
                )
                return (p.x, p.y, p.z, yaw)

        self.get_logger().warn(
            f"Block '{block_id}' not found in world model, using YAML fallback"
        )
        return None

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

        # Query world model for actual pickup pose; fall back to YAML
        wm_pose = self._query_block_pose(block_id)
        if wm_pose is not None:
            px, py, pz, pyaw = wm_pose
            pose_source = "world_model"
        else:
            px, py, pz, pyaw = x, y, z, yaw
            pose_source = "yaml"

        # Compute place approach position: offset along block yaw direction
        d = self._place_approach_offset
        ax = x + d * math.cos(yaw)
        ay = y + d * math.sin(yaw)

        response.success = True
        response.has_task = True
        response.task_id = task_id
        response.target_block_id = block_id
        response.reference_block_id = ""
        # pickup_pose from world model (or YAML fallback), target_pose always from YAML
        # reference_pose = place approach position (laterally offset for angled descent)
        # All poses are raw block CoG — grip server handles TCP offset
        response.pickup_pose = self._make_pose(px, py, pz, pyaw)
        response.target_pose = self._make_pose(x, y, z, yaw)
        response.reference_pose = self._make_pose(ax, ay, z, yaw)
        response.message = f"Task {idx + 1}/{len(tasks)}: {block_id}"

        self.get_logger().info(
            f"GetNextAssemblyTask | plan={plan_name} task={task_id} "
            f"block={block_id} pickup=({px:.2f}, {py:.2f}, {pz:.2f}) [{pose_source}] "
            f"target=({x:.2f}, {y:.2f}, {z:.2f}) approach=({ax:.2f}, {ay:.2f}) "
            f"yaw={math.degrees(yaw):.1f}deg"
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
