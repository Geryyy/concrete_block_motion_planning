#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from cbmp.path_setup import ensure_motion_planning_on_path

ensure_motion_planning_on_path()

from control_msgs.action import FollowJointTrajectory
from motion_planning.adapters import profile_command_joint_names
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class HeadlessTrajectorySmoke(Node):
    def __init__(self) -> None:
        super().__init__("headless_trajectory_smoke")

        default_profiles = (
            Path(get_package_share_directory("concrete_block_motion_planning"))
            / "config"
            / "headless_trajectory_profiles.yaml"
        )
        self.declare_parameter(
            "action_name", "/trajectory_controller_a2b/follow_joint_trajectory"
        )
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("profiles_file", str(default_profiles))
        self.declare_parameter("profile", "slew_out_and_back")
        self.declare_parameter("motion_timeout_s", 2.0)
        self.declare_parameter("motion_min_delta", 0.02)

        self._action_name = str(self.get_parameter("action_name").value)
        self._joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self._profiles_file = Path(str(self.get_parameter("profiles_file").value))
        self._profile_name = str(self.get_parameter("profile").value)
        self._motion_timeout_s = float(self.get_parameter("motion_timeout_s").value)
        self._motion_min_delta = float(self.get_parameter("motion_min_delta").value)

        self._latest_joint_positions: Dict[str, float] = {}
        self.create_subscription(JointState, self._joint_states_topic, self._on_joint_states, 10)
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            self._action_name,
        )

    def _on_joint_states(self, msg: JointState) -> None:
        for idx, name in enumerate(msg.name):
            if idx < len(msg.position):
                self._latest_joint_positions[str(name)] = float(msg.position[idx])

    def _wait_for_joint_state(self, joint_names: Sequence[str], timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() <= deadline:
            if all(name in self._latest_joint_positions for name in joint_names):
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def _load_profile(self) -> tuple[List[str], List[dict]]:
        data = yaml.safe_load(self._profiles_file.read_text(encoding="utf-8")) or {}
        profiles = data.get("profiles", {})
        if self._profile_name not in profiles:
            available = ", ".join(sorted(profiles.keys()))
            raise ValueError(
                f"Unknown profile '{self._profile_name}'. Available: [{available}]"
            )
        profile = profiles[self._profile_name]
        joint_names = [str(name) for name in profile["joint_names"]]
        waypoints = list(profile["waypoints"])
        if len(waypoints) < 2:
            raise ValueError("Profile must contain at least two waypoints.")
        return joint_names, waypoints

    def _build_trajectory(
        self,
        joint_names: Sequence[str],
        waypoints: Sequence[dict],
    ) -> JointTrajectory:
        current = [float(self._latest_joint_positions.get(name, 0.0)) for name in joint_names]
        trajectory = JointTrajectory()
        trajectory.joint_names = list(joint_names)

        positions: List[List[float]] = []
        times: List[float] = []
        for waypoint in waypoints:
            deltas = [float(v) for v in waypoint["deltas"]]
            if len(deltas) != len(joint_names):
                raise ValueError("Waypoint delta dimension does not match joint names.")
            positions.append([current[i] + deltas[i] for i in range(len(joint_names))])
            times.append(float(waypoint["time_from_start"]))

        velocities: List[List[float]] = []
        for idx in range(len(positions)):
            if idx == len(positions) - 1:
                velocities.append([0.0] * len(joint_names))
                continue
            dt = max(1e-3, times[idx + 1] - times[idx])
            velocities.append(
                [
                    (positions[idx + 1][j] - positions[idx][j]) / dt
                    for j in range(len(joint_names))
                ]
            )

        for idx, pos in enumerate(positions):
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in pos]
            point.velocities = [float(v) for v in velocities[idx]]
            secs = float(times[idx])
            point.time_from_start.sec = int(secs)
            point.time_from_start.nanosec = int((secs - int(secs)) * 1e9)
            trajectory.points.append(point)
        return trajectory

    def _motion_metrics(
        self,
        joint_names: Sequence[str],
        start_positions: Dict[str, float],
    ) -> tuple[float, float, Dict[str, float]]:
        deltas = {}
        for name in joint_names:
            if name not in self._latest_joint_positions or name not in start_positions:
                continue
            deltas[name] = float(self._latest_joint_positions[name]) - float(start_positions[name])
        if not deltas:
            return 0.0, 0.0, deltas
        max_abs = max(abs(v) for v in deltas.values())
        norm = math.sqrt(sum(v * v for v in deltas.values()))
        return float(max_abs), float(norm), deltas

    def _update_peak_metrics(
        self,
        joint_names: Sequence[str],
        start_positions: Dict[str, float],
        peak_abs: float,
        peak_norm: float,
        peak_deltas: Dict[str, float],
    ) -> tuple[float, float, Dict[str, float], float, float, Dict[str, float]]:
        max_abs, norm, deltas = self._motion_metrics(joint_names, start_positions)
        if max_abs >= peak_abs:
            return max_abs, norm, dict(deltas), max_abs, norm, deltas
        return peak_abs, peak_norm, peak_deltas, max_abs, norm, deltas

    def run(self) -> int:
        joint_names, waypoints = self._load_profile()
        measured_joint_names = list(profile_command_joint_names(joint_names))

        self.get_logger().info(
            f"Waiting for joint states on '{self._joint_states_topic}' for commanded joints={measured_joint_names}"
        )
        if not self._wait_for_joint_state(measured_joint_names):
            missing = [
                name for name in measured_joint_names if name not in self._latest_joint_positions
            ]
            self.get_logger().error(
                "Timed out waiting for required joint_states. "
                f"missing_command_joints={missing}"
            )
            return 2

        missing_profile_joints = [
            name for name in joint_names if name not in self._latest_joint_positions
        ]
        if missing_profile_joints:
            self.get_logger().warn(
                "Profile joints missing from /joint_states; defaulting their initial positions to 0.0: "
                f"{missing_profile_joints}"
            )

        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                f"Action server '{self._action_name}' unavailable."
            )
            return 3

        start_positions = {
            name: float(self._latest_joint_positions[name]) for name in measured_joint_names
        }
        trajectory = self._build_trajectory(joint_names, waypoints)
        self.get_logger().info(
            f"Sending profile '{self._profile_name}' with {len(trajectory.points)} points to "
            f"'{self._action_name}'"
        )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        send_future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=5.0)
        if not send_future.done():
            self.get_logger().error("Timed out sending goal.")
            return 4
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Goal was rejected.")
            return 5

        result_future = goal_handle.get_result_async()
        result_deadline = time.monotonic() + 120.0
        peak_abs = 0.0
        peak_norm = 0.0
        peak_deltas: Dict[str, float] = {}
        final_abs = 0.0
        final_norm = 0.0
        final_deltas: Dict[str, float] = {}
        while not result_future.done():
            if time.monotonic() > result_deadline:
                self.get_logger().error("Timed out waiting for trajectory result.")
                return 6
            rclpy.spin_once(self, timeout_sec=0.1)
            (
                peak_abs,
                peak_norm,
                peak_deltas,
                final_abs,
                final_norm,
                final_deltas,
            ) = self._update_peak_metrics(
                measured_joint_names,
                start_positions,
                peak_abs,
                peak_norm,
                peak_deltas,
            )

        wrapped = result_future.result()
        if wrapped is None:
            self.get_logger().error("Trajectory action returned no result.")
            return 7

        deadline = time.monotonic() + self._motion_timeout_s
        while time.monotonic() <= deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            (
                peak_abs,
                peak_norm,
                peak_deltas,
                final_abs,
                final_norm,
                final_deltas,
            ) = self._update_peak_metrics(
                measured_joint_names,
                start_positions,
                peak_abs,
                peak_norm,
                peak_deltas,
            )

        result = wrapped.result
        self.get_logger().info(
            "Action result | "
            f"status={wrapped.status} error_code={int(result.error_code)} "
            f"error='{result.error_string}'"
        )
        self.get_logger().info(
            "Measured motion | "
            f"peak_abs={peak_abs:.4f} peak_norm={peak_norm:.4f} peak_deltas={peak_deltas} "
            f"final_abs={final_abs:.4f} final_norm={final_norm:.4f} final_deltas={final_deltas}"
        )
        if peak_abs < self._motion_min_delta:
            self.get_logger().error(
                "Measured joint motion stayed below threshold during and after action success."
            )
            return 8
        return 0


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node = HeadlessTrajectorySmoke()
    try:
        return node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
