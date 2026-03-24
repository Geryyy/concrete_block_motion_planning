from __future__ import annotations

from typing import Dict

import tf2_geometry_msgs
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path as NavPath
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.time import Time
from tf2_ros import TransformException
from trajectory_msgs.msg import JointTrajectory

from timber_crane_planning_interfaces.srv import CalcMovement

from ..results import BackendPlanResult, PlannerCapabilities
from .base import PlannerBackend


class TimberPlannerBackend(PlannerBackend):
    def __init__(self, node) -> None:
        self._node = node
        self._client_cb_group = ReentrantCallbackGroup()
        self._client = node.create_client(
            CalcMovement,
            node._timber_a2b_service,
            callback_group=self._client_cb_group,
        )

    @property
    def backend_name(self) -> str:
        return "timber"

    @property
    def capabilities(self) -> PlannerCapabilities:
        return PlannerCapabilities(
            supports_move_empty=True,
            supports_named_configurations=True,
            supports_world_model_obstacles=False,
            supports_pick_place=False,
            supports_geometric_stage=False,
        )

    def plan_move_empty(
        self,
        *,
        start_pose: PoseStamped,
        goal_pose: PoseStamped,
        geometric_method: str,
        geometric_timeout_s: float,
        trajectory_method: str,
        trajectory_timeout_s: float,
        validate_dynamics: bool,
        planning_context: Dict[str, object],
    ) -> BackendPlanResult:
        del start_pose, geometric_method, geometric_timeout_s, trajectory_method, validate_dynamics

        if not self._client.wait_for_service(timeout_sec=2.0):
            return BackendPlanResult(
                success=False,
                message=f"Timber backend service '{self._node._timber_a2b_service}' unavailable.",
                trajectory=JointTrajectory(),
                cartesian_path=NavPath(),
            )

        transformed_goal, transform_msg = self._transform_goal(goal_pose)
        if transformed_goal is None:
            return BackendPlanResult(
                success=False,
                message=transform_msg,
                trajectory=JointTrajectory(),
                cartesian_path=NavPath(),
            )

        req = CalcMovement.Request()
        req.y_n = Point(
            x=float(transformed_goal.pose.position.x),
            y=float(transformed_goal.pose.position.y),
            z=float(transformed_goal.pose.position.z),
        )
        req.phi_tool_n = float(
            self._node._quaternion_to_yaw(transformed_goal.pose.orientation)
        )
        req.slow_down = 1.0
        req.t_end = 0.0
        req.publish_path = True

        timeout_s = max(5.0, float(trajectory_timeout_s))
        self._client.service_is_ready()
        try:
            res = self._client.call(req, timeout_sec=timeout_s)
        except TypeError:
            # Older rclpy versions expose call() without timeout support.
            res = self._client.call(req)
        except Exception as exc:
            return BackendPlanResult(
                success=False,
                message=f"Timed out waiting for timber backend trajectory (timeout={timeout_s:.1f}s): {exc}",
                trajectory=JointTrajectory(),
                cartesian_path=NavPath(),
            )
        if res is None:
            return BackendPlanResult(
                success=False,
                message="Timber backend returned no response.",
                trajectory=JointTrajectory(),
                cartesian_path=NavPath(),
            )

        path = NavPath()
        if res.tcp_path:
            path.header = res.tcp_path[0].header
            path.poses = list(res.tcp_path)

        context_msg = ""
        if planning_context.get("use_world_model"):
            context_msg = " | timber backend ignores world model obstacles"
        return BackendPlanResult(
            success=bool(res.success),
            message=(
                "Timber move-empty plan+compute "
                f"{'success' if res.success else 'failure'} via '{self._node._timber_a2b_service}'"
                f"{context_msg}"
            ),
            trajectory=res.trajectory,
            cartesian_path=path,
            geometric_plan_id="timber_direct",
        )

    def _transform_goal(self, goal_pose: PoseStamped) -> tuple[PoseStamped | None, str]:
        pose = PoseStamped()
        pose.header = goal_pose.header
        if pose.header.stamp == TimeMsg():
            pose.header.stamp = self._node.get_clock().now().to_msg()
        pose.pose = goal_pose.pose

        if pose.header.frame_id == self._node._timber_goal_frame:
            return self._apply_target_z(pose), ""

        try:
            transform = self._node._tf_buffer.lookup_transform(
                self._node._timber_goal_frame,
                pose.header.frame_id,
                Time(),
            )
        except TransformException as exc:
            return (
                None,
                "Failed to transform timber target from "
                f"'{pose.header.frame_id}' to '{self._node._timber_goal_frame}': {exc}",
            )

        transformed = PoseStamped()
        transformed.header.frame_id = self._node._timber_goal_frame
        transformed.header.stamp = transform.header.stamp
        transformed.pose = tf2_geometry_msgs.do_transform_pose(
            pose.pose,
            transform,
        )
        return self._apply_target_z(transformed), ""

    def _apply_target_z(self, pose: PoseStamped) -> PoseStamped:
        adjusted = PoseStamped()
        adjusted.header = pose.header
        adjusted.pose = pose.pose
        adjusted.pose.position.z = float(self._node._timber_move_empty_target_z)
        return adjusted
