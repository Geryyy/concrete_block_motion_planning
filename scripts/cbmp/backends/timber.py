from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import tf2_geometry_msgs
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path as NavPath
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.time import Time
from tf2_ros import TransformException
from trajectory_msgs.msg import JointTrajectory
from wood_log_msgs.msg import LogShape

from timber_crane_planning_interfaces.srv import CalcGripMovement, CalcMovement

from ..results import BackendPlanResult, PlannerCapabilities
from .base import PlannerBackend


@dataclass(frozen=True)
class _PayloadProxy:
    mass_kg: float
    shape: LogShape
    s_log_grippoint: Point
    source: str


class TimberPlannerBackend(PlannerBackend):
    _TIMBER_DIRECT_METHODS = {"", "TIMBER_MOVE_EMPTY", "TIMBER_A2B", "MOVE_EMPTY"}
    _TIMBER_LOADED_METHODS = {"TIMBER_MOVE_LOADED", "TIMBER_CARRY", "MOVE_LOADED"}
    _TIMBER_GRIP_PHASES = {
        "TIMBER_LAYDOWN": 0,
        "TIMBER_GRIP": 1,
        "TIMBER_LIFT": 2,
        "TIMBER_RELEASE": 3,
        "TIMBER_REGRASP": 4,
    }

    def __init__(self, node) -> None:
        self._node = node
        self._client_cb_group = ReentrantCallbackGroup()
        self._move_client = node.create_client(
            CalcMovement,
            node._timber_a2b_service,
            callback_group=self._client_cb_group,
        )
        self._grip_client = node.create_client(
            CalcGripMovement,
            node._timber_grip_service,
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
            supports_pick_place=True,
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
        del start_pose, geometric_method, geometric_timeout_s, validate_dynamics

        method_norm = trajectory_method.strip().upper()
        if method_norm in self._TIMBER_GRIP_PHASES:
            return self._plan_grip_phase(
                method_norm=method_norm,
                goal_pose=goal_pose,
                trajectory_timeout_s=trajectory_timeout_s,
                planning_context=planning_context,
            )
        loaded = method_norm in self._TIMBER_LOADED_METHODS
        return self._plan_a2b(
            method_norm=method_norm,
            goal_pose=goal_pose,
            trajectory_timeout_s=trajectory_timeout_s,
            planning_context=planning_context,
            carries_payload=loaded,
        )

    def _plan_a2b(
        self,
        *,
        method_norm: str,
        goal_pose: PoseStamped,
        trajectory_timeout_s: float,
        planning_context: Dict[str, object],
        carries_payload: bool,
    ) -> BackendPlanResult:
        if not self._move_client.wait_for_service(timeout_sec=2.0):
            return self._failure(
                f"Timber backend service '{self._node._timber_a2b_service}' unavailable."
            )

        transformed_goal, transform_msg = self._transform_goal(
            goal_pose,
            override_target_z=not carries_payload,
        )
        if transformed_goal is None:
            return self._failure(transform_msg)

        payload = self._build_payload_proxy(planning_context)
        req = CalcMovement.Request()
        req.y_n = Point(
            x=float(transformed_goal.pose.position.x),
            y=float(transformed_goal.pose.position.y),
            z=float(transformed_goal.pose.position.z),
        )
        req.phi_tool_n = float(
            self._node._quaternion_to_yaw(transformed_goal.pose.orientation)
        )
        req.carries_log = bool(carries_payload)
        if carries_payload:
            req.log_carrying = payload.shape
            req.mass_log = float(payload.mass_kg)
            req.s_log_grippoint = payload.s_log_grippoint
        req.slow_down = 1.0
        req.t_end = 0.0
        req.publish_path = True

        timeout_s = max(5.0, float(trajectory_timeout_s))
        try:
            res = self._move_client.call(req, timeout_sec=timeout_s)
        except TypeError:
            res = self._move_client.call(req)
        except Exception as exc:
            return self._failure(
                f"Timed out waiting for timber backend trajectory (timeout={timeout_s:.1f}s): {exc}"
            )
        if res is None:
            return self._failure("Timber backend returned no response.")

        path = self._path_from_tcp_path(res.tcp_path)
        context_msg = self._limitations_message(planning_context)
        payload_msg = (
            f" | payload={payload.source} mass={payload.mass_kg:.1f}kg"
            if carries_payload
            else ""
        )
        move_kind = "loaded" if carries_payload else "empty"
        return BackendPlanResult(
            success=bool(res.success),
            message=(
                f"Timber {move_kind} A2B "
                f"{'success' if res.success else 'failure'} via '{self._node._timber_a2b_service}'"
                f"{payload_msg}{context_msg}"
            ),
            trajectory=res.trajectory,
            cartesian_path=path,
            geometric_plan_id=f"timber_{method_norm.lower() or 'move_empty'}",
        )

    def _plan_grip_phase(
        self,
        *,
        method_norm: str,
        goal_pose: PoseStamped,
        trajectory_timeout_s: float,
        planning_context: Dict[str, object],
    ) -> BackendPlanResult:
        if not self._grip_client.wait_for_service(timeout_sec=2.0):
            return self._failure(
                f"Timber grip service '{self._node._timber_grip_service}' unavailable."
            )

        transformed_goal, transform_msg = self._transform_goal(goal_pose, override_target_z=False)
        if transformed_goal is None:
            return self._failure(transform_msg)

        payload = self._build_payload_proxy(planning_context)
        req = CalcGripMovement.Request()
        req.y_n = Point(
            x=float(transformed_goal.pose.position.x),
            y=float(transformed_goal.pose.position.y),
            z=float(transformed_goal.pose.position.z),
        )
        req.xy_n = [math.nan, math.nan]
        req.phi_tool_n = float(
            self._node._quaternion_to_yaw(transformed_goal.pose.orientation)
        )
        req.carries_log = bool(
            method_norm in {"TIMBER_LAYDOWN", "TIMBER_LIFT", "TIMBER_RELEASE"}
        )
        req.log = payload.shape
        req.slow_down = 1.0
        req.select_phases = int(self._TIMBER_GRIP_PHASES[method_norm])
        req.s_log_grippoint = payload.s_log_grippoint

        timeout_s = max(5.0, float(trajectory_timeout_s))
        try:
            res = self._grip_client.call(req, timeout_sec=timeout_s)
        except TypeError:
            res = self._grip_client.call(req)
        except Exception as exc:
            return self._failure(
                f"Timed out waiting for timber grip trajectory (timeout={timeout_s:.1f}s): {exc}"
            )
        if res is None:
            return self._failure("Timber grip backend returned no response.")

        success = int(res.success) > 0
        path = self._path_from_tcp_path(res.tcp_path)
        context_msg = self._limitations_message(planning_context)
        return BackendPlanResult(
            success=success,
            message=(
                f"Timber manipulation phase '{method_norm}' "
                f"{'success' if success else f'failure(code={int(res.success)})'} via "
                f"'{self._node._timber_grip_service}' | payload={payload.source} "
                f"mass={payload.mass_kg:.1f}kg{context_msg}"
            ),
            trajectory=res.trajectory,
            cartesian_path=path,
            geometric_plan_id=f"timber_{method_norm.lower()}",
        )

    def _build_payload_proxy(self, planning_context: Dict[str, object]) -> _PayloadProxy:
        target_id = str(planning_context.get("target_block_id", "")).strip()
        dims = None
        source = "defaults"

        for obj in planning_context.get("planning_scene_objects", []):
            if str(obj.get("id", "")).strip() == target_id:
                dims = tuple(float(v) for v in obj.get("dimensions", ()))
                source = f"planning_scene:{target_id}"
                break

        if not dims or len(dims) != 3 or max(dims) <= 0.0:
            dims = tuple(float(v) for v in self._node._moving_block_size)
            if target_id:
                source = f"default_block_size:{target_id}"

        dims_sorted = sorted(float(v) for v in dims if float(v) > 0.0)
        if len(dims_sorted) != 3:
            dims_sorted = sorted(float(v) for v in self._node._moving_block_size)
            source = "fallback_defaults"

        length = dims_sorted[-1]
        radius = 0.5 * max(dims_sorted[0], dims_sorted[1])
        volume = dims_sorted[0] * dims_sorted[1] * dims_sorted[2]
        mass = max(1.0, volume * float(self._node._timber_payload_density_kg_m3))

        shape = LogShape()
        shape.length = float(length)
        shape.radius_top = float(radius)
        shape.radius_bottom = float(radius)

        grip_point = Point()
        grip_point.x = float(self._node._timber_payload_grippoint_xyz[0])
        grip_point.y = float(self._node._timber_payload_grippoint_xyz[1])
        grip_point.z = float(self._node._timber_payload_grippoint_xyz[2])
        return _PayloadProxy(
            mass_kg=mass,
            shape=shape,
            s_log_grippoint=grip_point,
            source=source,
        )

    def _transform_goal(
        self,
        goal_pose: PoseStamped,
        *,
        override_target_z: bool,
    ) -> tuple[PoseStamped | None, str]:
        pose = PoseStamped()
        pose.header = goal_pose.header
        if pose.header.stamp == TimeMsg():
            pose.header.stamp = self._node.get_clock().now().to_msg()
        pose.pose = goal_pose.pose

        if pose.header.frame_id == self._node._timber_goal_frame:
            return self._apply_target_z(pose) if override_target_z else pose, ""

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
        return self._apply_target_z(transformed) if override_target_z else transformed, ""

    def _apply_target_z(self, pose: PoseStamped) -> PoseStamped:
        adjusted = PoseStamped()
        adjusted.header = pose.header
        adjusted.pose = pose.pose
        adjusted.pose.position.z = float(self._node._timber_move_empty_target_z)
        return adjusted

    def _path_from_tcp_path(self, tcp_path) -> NavPath:
        path = NavPath()
        if tcp_path:
            path.header = tcp_path[0].header
            path.poses = list(tcp_path)
        return path

    def _limitations_message(self, planning_context: Dict[str, object]) -> str:
        limitations = []
        if planning_context.get("use_world_model"):
            limitations.append("timber backend ignores CBS planning-scene obstacles")
        if planning_context.get("reference_block_id"):
            limitations.append("reference_block_id is not consumed by timber services")
        if limitations:
            return " | " + "; ".join(limitations)
        return ""

    def _failure(self, message: str) -> BackendPlanResult:
        return BackendPlanResult(
            success=False,
            message=message,
            trajectory=JointTrajectory(),
            cartesian_path=NavPath(),
        )
