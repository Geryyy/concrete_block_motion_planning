from __future__ import annotations

import math
import time
from typing import Tuple

import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import SwitchController
from .path_setup import ensure_motion_planning_on_path

ensure_motion_planning_on_path()

from motion_planning.adapters import profile_command_joint_names, project_positions_to_command_joints


class ExecutionAdapter:
    def __init__(self, node) -> None:
        self._node = node

    @staticmethod
    def _wait_for_future(future, timeout_s: float, poll_s: float = 0.01) -> bool:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() <= deadline:
            if future.done():
                return True
            time.sleep(max(0.001, float(poll_s)))
        return future.done()

    @staticmethod
    def _measured_motion_metrics(
        start_positions: dict[str, float],
        current_positions: dict[str, float],
    ) -> tuple[float, float, list[str]]:
        deltas = []
        covered = []
        for name, start in start_positions.items():
            if name not in current_positions:
                continue
            delta = float(current_positions[name]) - float(start)
            deltas.append(delta)
            covered.append(name)
        if not deltas:
            return 0.0, 0.0, covered
        max_abs = max(abs(v) for v in deltas)
        norm = math.sqrt(sum(v * v for v in deltas))
        return float(max_abs), float(norm), covered

    def _validate_motion_after_execution(
        self,
        trajectory,
        start_positions: dict[str, float],
    ) -> tuple[bool, str]:
        joint_names = [str(name) for name in profile_command_joint_names(trajectory.joint_names)]
        if not joint_names:
            return False, "Execution completed but trajectory has no joint names for motion validation."

        if not start_positions:
            return (
                False,
                "Execution completed but no measured joint-state feedback was available for "
                f"{joint_names}.",
            )

        deadline = time.monotonic() + float(self._node._execution_motion_check_timeout_s)
        latest_positions = start_positions
        while time.monotonic() <= deadline:
            latest_positions = self._node._snapshot_joint_positions(list(start_positions.keys()))
            max_abs, norm, covered = self._measured_motion_metrics(
                start_positions,
                latest_positions,
            )
            if covered and max_abs >= float(self._node._execution_motion_min_delta):
                return (
                    True,
                    f"measured_motion=max_abs:{max_abs:.4f}, norm:{norm:.4f}, joints={covered}",
                )
            time.sleep(0.05)

        max_abs, norm, covered = self._measured_motion_metrics(start_positions, latest_positions)
        return (
            False,
            "Trajectory action reported success, but measured motion stayed below threshold "
            f"(max_abs={max_abs:.4f}, norm={norm:.4f}, min_delta={float(self._node._execution_motion_min_delta):.4f}, "
            f"joints={covered}).",
        )

    def switch_execution_controller(
        self,
        activate: bool,
        timeout_s: float = 2.0,
    ) -> Tuple[bool, str]:
        if not self._node._execution_switch_controller:
            return True, "controller switching disabled"
        if self._node._switch_controller_client is None:
            return False, "controller switch client is not initialized"
        if not self._node._execution_activate_controller:
            return False, "execution.activate_controller is empty"
        if not self._node._switch_controller_client.wait_for_service(timeout_sec=timeout_s):
            return (
                False,
                "controller switch service "
                f"'{self._node._execution_switch_service}' unavailable",
            )

        req = SwitchController.Request()
        req.strictness = SwitchController.Request.BEST_EFFORT
        req.activate_asap = True
        timeout_s_f = max(0.0, float(timeout_s))
        timeout_sec = int(timeout_s_f)
        timeout_nsec = int((timeout_s_f - timeout_sec) * 1e9)
        req.timeout.sec = timeout_sec
        req.timeout.nanosec = timeout_nsec
        if activate:
            req.activate_controllers = [self._node._execution_activate_controller]
            req.deactivate_controllers = []
        else:
            req.activate_controllers = []
            req.deactivate_controllers = [self._node._execution_activate_controller]

        future = self._node._switch_controller_client.call_async(req)
        if not self._wait_for_future(future, timeout_s=timeout_s):
            return False, "controller switch call timed out"
        res = future.result()
        if res is None:
            return False, "controller switch call returned no response"
        if not bool(res.ok):
            return False, "controller switch request rejected"

        action = "activated" if activate else "deactivated"
        return True, f"controller '{self._node._execution_activate_controller}' {action}"

    def dispatch_trajectory_topic(self, trajectory, trajectory_id: str) -> tuple[bool, str]:
        if self._node._trajectory_cmd_pub is None:
            return False, "Execution publisher is not initialized."
        if self._node._trajectory_cmd_pub.get_subscription_count() <= 0:
            return (
                False,
                "No subscribers on execution topic "
                f"'{self._node._execution_trajectory_topic}'.",
            )
        self._node._trajectory_cmd_pub.publish(trajectory)
        return (
            True,
            "Dispatched trajectory "
            f"'{trajectory_id}' to '{self._node._execution_trajectory_topic}'.",
        )

    def dispatch_trajectory_action(self, trajectory, trajectory_id: str) -> tuple[bool, str]:
        if self._node._trajectory_action_client is None:
            return False, "Execution action client is not initialized."
        if not self._node._trajectory_action_client.wait_for_server(timeout_sec=2.0):
            return (
                False,
                "Execution action server "
                f"'{self._node._execution_action_name}' unavailable.",
            )

        switched, switch_msg = self.switch_execution_controller(activate=True)
        if not switched:
            return False, f"Failed to activate execution controller: {switch_msg}"

        start_positions = project_positions_to_command_joints(
            trajectory.joint_names,
            self._node._snapshot_joint_positions([str(name) for name in trajectory.joint_names]),
        )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        send_future = self._node._trajectory_action_client.send_goal_async(goal)
        if not self._wait_for_future(send_future, timeout_s=5.0):
            return False, "Timed out while sending trajectory action goal."
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return (
                False,
                "Execution action goal rejected by "
                f"'{self._node._execution_action_name}'.",
            )

        result_future = goal_handle.get_result_async()
        if not self._wait_for_future(
            result_future,
            timeout_s=float(self._node._execution_result_timeout_s),
        ):
            return (
                False,
                "Timed out waiting for trajectory action result "
                f"(timeout={self._node._execution_result_timeout_s:.1f}s).",
            )

        wrapped = result_future.result()
        if wrapped is None:
            return False, "Trajectory action completed with empty result."
        result = wrapped.result
        status_ok = wrapped.status == GoalStatus.STATUS_SUCCEEDED
        code_ok = int(result.error_code) == int(result.SUCCESSFUL)
        ok = bool(status_ok and code_ok)
        if self._node._execution_deactivate_after_execution:
            switched_down, switch_down_msg = self.switch_execution_controller(activate=False)
            if not switched_down:
                self._node.get_logger().warn(
                    f"Controller deactivation failed after execution: {switch_down_msg}"
                )

        if not ok:
            return (
                False,
                "Trajectory action failed: "
                f"status={wrapped.status}, error_code={int(result.error_code)}, "
                f"error='{result.error_string}'.",
            )
        if self._node._execution_motion_check_timeout_s > 0.0:
            motion_ok, motion_msg = self._validate_motion_after_execution(
                trajectory,
                start_positions,
            )
            if not motion_ok:
                return False, motion_msg
        success_msg = (
            f"Trajectory '{trajectory_id}' executed via action "
            f"'{self._node._execution_action_name}'."
        )
        if self._node._execution_motion_check_timeout_s > 0.0:
            success_msg = f"{success_msg} {motion_msg}"
        return True, success_msg

    def dispatch_trajectory(self, trajectory, trajectory_id: str) -> tuple[bool, str]:
        if not self._node._execution_enabled:
            return False, "Execution disabled (execution.enabled=false)."
        if not trajectory.points:
            return False, f"Trajectory '{trajectory_id}' has no points."
        if self._node._execution_backend == "action":
            return self.dispatch_trajectory_action(trajectory, trajectory_id)
        if self._node._execution_backend != "topic":
            return (
                False,
                f"Unsupported execution backend '{self._node._execution_backend}'.",
            )
        return self.dispatch_trajectory_topic(trajectory, trajectory_id)
