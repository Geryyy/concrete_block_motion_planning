#!/usr/bin/env python3

from __future__ import annotations

import rclpy
from epsilon_crane_control_interfaces.msg import GripperState
from rclpy.node import Node


class DummyGripperStatePublisher(Node):
    def __init__(self) -> None:
        super().__init__("dummy_gripper_state_publisher")
        self.declare_parameter("topic", "/gripper_state")
        self.declare_parameter("publish_rate_hz", 5.0)

        topic = str(self.get_parameter("topic").value)
        publish_rate_hz = max(float(self.get_parameter("publish_rate_hz").value), 0.1)

        self._publisher = self.create_publisher(GripperState, topic, 10)
        self._msg = GripperState()
        self._msg.carries_log = False
        self._msg.mass = 0.0
        self._msg.s_log_grippoint.x = 0.0
        self._msg.s_log_grippoint.y = 0.0
        self._msg.s_log_grippoint.z = 0.0

        self.create_timer(1.0 / publish_rate_hz, self._publish)
        self.get_logger().info(f"Publishing dummy empty gripper_state on {topic}")

    def _publish(self) -> None:
        self._msg.header.stamp = self.get_clock().now().to_msg()
        self._publisher.publish(self._msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DummyGripperStatePublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
