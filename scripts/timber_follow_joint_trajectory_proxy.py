#!/usr/bin/env python3

from __future__ import annotations

import time
from typing import Dict

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class TimberFollowJointTrajectoryProxy(Node):
    def __init__(self) -> None:
        super().__init__("timber_follow_joint_trajectory_proxy")

        self.declare_parameter(
            "proxy_action_name", "/trajectory_controller_a2b/follow_joint_trajectory"
        )
        self.declare_parameter(
            "target_commands_topic", "/trajectory_controllers/commands"
        )
        self.declare_parameter("joint_states_topic", "/joint_states")

        proxy_action_name = str(self.get_parameter("proxy_action_name").value)
        target_commands_topic = str(self.get_parameter("target_commands_topic").value)
        joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self._latest_joint_state: Dict[str, float] = {}

        self.create_subscription(JointState, joint_states_topic, self._on_joint_state, 10)
        self._commands_pub = self.create_publisher(
            JointTrajectory, target_commands_topic, 10
        )
        self._server = ActionServer(
            self,
            FollowJointTrajectory,
            proxy_action_name,
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
        )

        self.get_logger().info(
            "Started timber FollowJointTrajectory proxy | "
            f"{proxy_action_name} -> topic:{target_commands_topic}"
        )

    def destroy_node(self) -> bool:
        self._server.destroy()
        return super().destroy_node()

    def _on_joint_state(self, msg: JointState) -> None:
        for idx, name in enumerate(msg.name):
            if idx < len(msg.position):
                self._latest_joint_state[name] = float(msg.position[idx])

    def _goal_callback(self, _goal_request) -> GoalResponse:
        return GoalResponse.ACCEPT

    def _cancel_callback(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    async def _execute_callback(self, goal_handle):
        trajectory = goal_handle.request.trajectory
        result = FollowJointTrajectory.Result()

        if not trajectory.points:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = "Trajectory is empty."
            return result

        expanded_trajectory = self._expand_trajectory(trajectory)
        first_point = expanded_trajectory.points[0]
        if len(first_point.velocities) != 8:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = (
                "TrajectoryForwardController expects 8 velocity commands per point; "
                f"received {len(first_point.velocities)}."
            )
            self.get_logger().error(result.error_string)
            return result

        self.get_logger().info(
            "Executing proxied trajectory with "
            f"{len(expanded_trajectory.points)} points, joints={list(expanded_trajectory.joint_names)}"
        )

        previous_time_s = 0.0
        for index, point in enumerate(expanded_trajectory.points):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                result.error_string = "Trajectory execution cancelled."
                self._publish_stop_command(expanded_trajectory.joint_names)
                return result

            point_time_s = self._duration_to_seconds(point.time_from_start)
            sleep_s = max(0.0, point_time_s - previous_time_s)
            if sleep_s > 0.0:
                time.sleep(sleep_s)

            sample = JointTrajectory()
            sample.header.stamp = self.get_clock().now().to_msg()
            sample.joint_names = list(expanded_trajectory.joint_names)
            sample.points = [point]
            self._commands_pub.publish(sample)
            previous_time_s = point_time_s

            if index == 0 or index == len(expanded_trajectory.points) - 1:
                self.get_logger().info(
                    "Published trajectory sample "
                    f"{index + 1}/{len(expanded_trajectory.points)} at t={point_time_s:.2f}s"
                )

        time.sleep(0.1)
        self._publish_stop_command(expanded_trajectory.joint_names)
        goal_handle.succeed()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        result.error_string = "Trajectory streamed to /trajectory_controllers/commands."
        return result

    def _publish_stop_command(self, joint_names: list[str]) -> None:
        stop = JointTrajectory()
        stop.header.stamp = self.get_clock().now().to_msg()
        stop.joint_names = list(joint_names)
        point = JointTrajectoryPoint()
        point.velocities = [0.0] * 8
        stop.points = [point]
        self._commands_pub.publish(stop)

    def _expand_trajectory(self, trajectory: JointTrajectory) -> JointTrajectory:
        if not trajectory.points:
            return trajectory
        if trajectory.points[0].velocities and len(trajectory.points[0].velocities) == 8:
            return trajectory

        output = JointTrajectory()
        output.header = trajectory.header
        output.joint_names = [
            "theta1_slewing_joint",
            "theta2_boom_joint",
            "theta3_arm_joint",
            "q4_big_telescope",
            "theta6_tip_joint",
            "theta7_tilt_joint",
            "theta8_rotator_joint",
            "theta10_outer_jaw_joint",
        ]

        current_outer_jaw = self._latest_joint_state.get("theta10_outer_jaw_joint", 1.5708)
        current_tip = self._latest_joint_state.get("theta6_tip_joint", 0.0)
        current_tilt = self._latest_joint_state.get("theta7_tilt_joint", 0.0)

        for point in trajectory.points:
            expanded = JointTrajectoryPoint()
            expanded.time_from_start = point.time_from_start

            pos_map = self._to_map(trajectory.joint_names, point.positions)
            vel_map = self._to_map(trajectory.joint_names, point.velocities)
            acc_map = self._to_map(trajectory.joint_names, point.accelerations)

            expanded.positions = [
                pos_map.get("theta1_slewing_joint", 0.0),
                pos_map.get("theta2_boom_joint", 0.0),
                pos_map.get("theta3_arm_joint", 0.0),
                pos_map.get("q4_big_telescope", 0.0),
                pos_map.get("theta6_tip_joint", current_tip),
                pos_map.get("theta7_tilt_joint", current_tilt),
                pos_map.get("theta8_rotator_joint", 0.0),
                pos_map.get("theta10_outer_jaw_joint", current_outer_jaw),
            ]
            if point.velocities:
                expanded.velocities = [
                    vel_map.get("theta1_slewing_joint", 0.0),
                    vel_map.get("theta2_boom_joint", 0.0),
                    vel_map.get("theta3_arm_joint", 0.0),
                    vel_map.get("q4_big_telescope", 0.0),
                    vel_map.get("theta6_tip_joint", 0.0),
                    vel_map.get("theta7_tilt_joint", 0.0),
                    vel_map.get("theta8_rotator_joint", 0.0),
                    vel_map.get("theta10_outer_jaw_joint", 0.0),
                ]
            if point.accelerations:
                expanded.accelerations = [
                    acc_map.get("theta1_slewing_joint", 0.0),
                    acc_map.get("theta2_boom_joint", 0.0),
                    acc_map.get("theta3_arm_joint", 0.0),
                    acc_map.get("q4_big_telescope", 0.0),
                    acc_map.get("theta6_tip_joint", 0.0),
                    acc_map.get("theta7_tilt_joint", 0.0),
                    acc_map.get("theta8_rotator_joint", 0.0),
                    acc_map.get("theta10_outer_jaw_joint", 0.0),
                ]
            output.points.append(expanded)
        return output

    @staticmethod
    def _to_map(names: list[str], values) -> Dict[str, float]:
        if not values:
            return {}
        return {
            name: float(values[idx])
            for idx, name in enumerate(names)
            if idx < len(values)
        }

    @staticmethod
    def _duration_to_seconds(duration_msg) -> float:
        return float(duration_msg.sec) + float(duration_msg.nanosec) / 1e9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TimberFollowJointTrajectoryProxy()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
