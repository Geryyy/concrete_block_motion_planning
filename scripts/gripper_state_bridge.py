#!/usr/bin/env python3

from __future__ import annotations

from typing import Dict

import rclpy
from epsilon_crane_control_interfaces.msg import GripperState
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from wood_log_msgs.msg import LogShape


class GripperStateBridge(Node):
    def __init__(self) -> None:
        super().__init__("gripper_state_bridge")
        self.declare_parameter("command_topic", "/concrete_block/gripper_command")
        self.declare_parameter("state_topic", "/gripper_state")
        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("default_mass_kg", 800.0)
        self.declare_parameter("default_length_m", 0.9)
        self.declare_parameter("default_radius_m", 0.3)
        self.declare_parameter("default_grippoint_xyz", [0.0, 0.0, 0.0])

        command_topic = str(self.get_parameter("command_topic").value)
        state_topic = str(self.get_parameter("state_topic").value)
        publish_rate_hz = max(float(self.get_parameter("publish_rate_hz").value), 0.1)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._publisher = self.create_publisher(GripperState, state_topic, qos)
        self._subscription = self.create_subscription(
            String,
            command_topic,
            self._on_command,
            10,
        )

        self._attached_block_id = ""
        self._msg = GripperState()
        self._reset_to_empty()
        self.create_timer(1.0 / publish_rate_hz, self._publish)
        self.get_logger().info(
            f"Publishing gripper_state on {state_topic} and listening for commands on {command_topic}"
        )

    def _default_grip_point(self) -> Point:
        values = list(self.get_parameter("default_grippoint_xyz").value)
        point = Point()
        if len(values) >= 3:
            point.x = float(values[0])
            point.y = float(values[1])
            point.z = float(values[2])
        return point

    def _default_shape(self) -> LogShape:
        shape = LogShape()
        shape.length = float(self.get_parameter("default_length_m").value)
        radius = float(self.get_parameter("default_radius_m").value)
        shape.radius_top = radius
        shape.radius_bottom = radius
        return shape

    def _reset_to_empty(self) -> None:
        self._attached_block_id = ""
        self._msg.carries_log = False
        self._msg.mass = 0.0
        self._msg.log = LogShape()
        self._msg.s_log_grippoint = Point()

    def _on_command(self, msg: String) -> None:
        tokens = msg.data.strip().split()
        if not tokens:
            return
        command = tokens[0].upper()
        values: Dict[str, str] = {}
        for token in tokens[1:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            values[key.strip()] = value.strip()

        if command == "OPEN":
            self._reset_to_empty()
            self.get_logger().info("Received OPEN command; cleared payload state")
            self._publish()
            return

        if command != "CLOSE":
            self.get_logger().warn(f"Ignoring unsupported gripper command: {msg.data}")
            return

        self._attached_block_id = values.get("block_id", "")
        self._msg.carries_log = True
        self._msg.mass = float(values.get("mass_kg", self.get_parameter("default_mass_kg").value))
        self._msg.log = self._default_shape()
        self._msg.log.length = float(values.get("length_m", self._msg.log.length))
        radius = float(values.get("radius_m", self._msg.log.radius_top))
        self._msg.log.radius_top = radius
        self._msg.log.radius_bottom = radius
        self._msg.s_log_grippoint = self._default_grip_point()
        self._msg.s_log_grippoint.x = float(values.get("grip_point_x", self._msg.s_log_grippoint.x))
        self._msg.s_log_grippoint.y = float(values.get("grip_point_y", self._msg.s_log_grippoint.y))
        self._msg.s_log_grippoint.z = float(values.get("grip_point_z", self._msg.s_log_grippoint.z))

        self.get_logger().info(
            "Received CLOSE command; attached payload "
            f"block_id={self._attached_block_id or '<unspecified>'} "
            f"mass={self._msg.mass:.1f}kg length={self._msg.log.length:.3f}m radius={radius:.3f}m"
        )
        self._publish()

    def _publish(self) -> None:
        self._msg.header.stamp = self.get_clock().now().to_msg()
        self._publisher.publish(self._msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GripperStateBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
