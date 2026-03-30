#!/usr/bin/env python3

from __future__ import annotations

import copy

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStateTimberCompat(Node):
    def __init__(self) -> None:
        super().__init__("joint_state_timber_compat")

        self.declare_parameter("source_topic", "/joint_states")
        self.declare_parameter("target_topic", "/joint_states_timber_compat")
        self.declare_parameter("source_gripper_joint", "q9_left_rail_joint")
        self.declare_parameter("compat_gripper_joint", "theta10_outer_jaw_joint")
        self.declare_parameter("default_compat_position", 1.5708)
        self.declare_parameter("compat_position_min", 0.8472)
        self.declare_parameter("compat_position_max", 3.0357)

        source_topic = str(self.get_parameter("source_topic").value)
        target_topic = str(self.get_parameter("target_topic").value)
        self._source_gripper_joint = str(
            self.get_parameter("source_gripper_joint").value
        )
        self._compat_gripper_joint = str(
            self.get_parameter("compat_gripper_joint").value
        )
        self._default_compat_position = float(
            self.get_parameter("default_compat_position").value
        )
        self._compat_position_min = float(
            self.get_parameter("compat_position_min").value
        )
        self._compat_position_max = float(
            self.get_parameter("compat_position_max").value
        )
        self._warned_out_of_range = False

        self.create_subscription(JointState, source_topic, self._on_joint_state, 10)
        self._publisher = self.create_publisher(JointState, target_topic, 10)

        self.get_logger().info(
            "Publishing timber-compatible joint states | "
            f"{source_topic} -> {target_topic} | "
            f"{self._source_gripper_joint} -> {self._compat_gripper_joint}"
        )

    def _on_joint_state(self, msg: JointState) -> None:
        compat_msg = copy.deepcopy(msg)
        if self._compat_gripper_joint in compat_msg.name:
            self._publisher.publish(compat_msg)
            return

        compat_pos = self._default_compat_position
        compat_vel = 0.0
        compat_effort = 0.0

        if self._source_gripper_joint in compat_msg.name:
            idx = compat_msg.name.index(self._source_gripper_joint)
            if idx < len(compat_msg.position):
                source_pos = float(compat_msg.position[idx])
                if self._compat_position_min <= source_pos <= self._compat_position_max:
                    compat_pos = source_pos
                else:
                    compat_pos = self._default_compat_position
                    if not self._warned_out_of_range:
                        self.get_logger().warn(
                            "Source gripper joint %.4f is outside timber-compatible range "
                            "[%.4f, %.4f]; using default compat position %.4f"
                            % (
                                source_pos,
                                self._compat_position_min,
                                self._compat_position_max,
                                self._default_compat_position,
                            )
                        )
                        self._warned_out_of_range = True
            if idx < len(compat_msg.velocity):
                compat_vel = float(compat_msg.velocity[idx])
            if idx < len(compat_msg.effort):
                compat_effort = float(compat_msg.effort[idx])

        compat_msg.name.append(self._compat_gripper_joint)
        compat_msg.position.append(compat_pos)
        if compat_msg.velocity:
            compat_msg.velocity.append(compat_vel)
        if compat_msg.effort:
            compat_msg.effort.append(compat_effort)

        self._publisher.publish(compat_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JointStateTimberCompat()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
