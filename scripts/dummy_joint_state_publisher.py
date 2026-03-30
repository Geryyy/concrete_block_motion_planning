#!/usr/bin/env python3

from __future__ import annotations

from typing import List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class DummyJointStatePublisher(Node):
    def __init__(self) -> None:
        super().__init__("dummy_joint_state_publisher")

        self.declare_parameter("topic", "/joint_states")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter(
            "joint_names",
            [
                "theta1_slewing_joint",
                "theta2_boom_joint",
                "theta3_arm_joint",
                "q4_big_telescope",
                "theta6_tip_joint",
                "theta7_tilt_joint",
                "theta8_rotator_joint",
                "q9_left_rail_joint",
                "q11_right_rail_joint",
                "pincer_cylinder_piston_in_barrel_linear_joint",
            ],
        )
        self.declare_parameter(
            "joint_positions",
            [0.0, 0.5, 0.5, 0.2, 0.0, 1.57, 0.0, 0.1, 0.1, 0.65],
        )

        topic = str(self.get_parameter("topic").value)
        publish_rate_hz = max(float(self.get_parameter("publish_rate_hz").value), 0.1)
        joint_names = [str(name) for name in self.get_parameter("joint_names").value]
        joint_positions = [float(value) for value in self.get_parameter("joint_positions").value]
        if len(joint_names) != len(joint_positions):
            raise ValueError("joint_names and joint_positions must have identical length")

        self._publisher = self.create_publisher(JointState, topic, 10)
        self._joint_names: List[str] = joint_names
        self._joint_positions: List[float] = joint_positions
        self.create_timer(1.0 / publish_rate_hz, self._publish)

        self.get_logger().info(
            f"Publishing fixed dummy joint states on {topic} for {len(self._joint_names)} joints"
        )

    def _publish(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self._joint_names)
        msg.position = list(self._joint_positions)
        msg.velocity = [0.0] * len(self._joint_names)
        msg.effort = [0.0] * len(self._joint_names)
        self._publisher.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DummyJointStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
