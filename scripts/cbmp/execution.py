from __future__ import annotations

from typing import Tuple

import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import SwitchController


class ExecutionAdapter:
    def __init__(self, node) -> None:
        self._node = node

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
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=timeout_s)
        if not future.done():
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

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        send_future = self._node._trajectory_action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self._node, send_future, timeout_sec=5.0)
        if not send_future.done():
            return False, "Timed out while sending trajectory action goal."
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return (
                False,
                "Execution action goal rejected by "
                f"'{self._node._execution_action_name}'.",
            )

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self._node,
            result_future,
            timeout_sec=float(self._node._execution_result_timeout_s),
        )
        if not result_future.done():
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
        return (
            True,
            "Trajectory "
            f"'{trajectory_id}' executed via action '{self._node._execution_action_name}'.",
        )

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

