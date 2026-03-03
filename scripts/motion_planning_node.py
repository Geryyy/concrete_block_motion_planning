#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as NavPath
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from concrete_block_motion_planning.srv import (
    ComputeTrajectory,
    ExecutePlannedMotion,  # deprecated
    ExecuteTrajectory,
    PlanBlockMotion,  # deprecated
    PlanGeometricPath,
)


@dataclass
class StoredGeometricPlan:
    geometric_plan_id: str
    path: NavPath
    success: bool
    message: str
    method: str


@dataclass
class StoredTrajectory:
    trajectory_id: str
    trajectory: JointTrajectory
    success: bool
    message: str
    method: str
    geometric_plan_id: str


class ConcreteBlockMotionPlanningNode(Node):
    def __init__(self) -> None:
        super().__init__("concrete_block_motion_planning_node")

        self._geometric_plans: Dict[str, StoredGeometricPlan] = {}
        self._trajectories: Dict[str, StoredTrajectory] = {}
        self._latest_trajectory_for_geometric: Dict[str, str] = {}

        self.declare_parameter("default_geometric_method", "POWELL")
        self.declare_parameter("default_trajectory_method", "ACADOS_PATH_FOLLOWING")
        self.declare_parameter("path_interpolation_points", 20)
        self.declare_parameter("fallback_segment_dt_s", 0.2)

        self._default_geometric_method = str(self.get_parameter("default_geometric_method").value)
        self._default_trajectory_method = str(self.get_parameter("default_trajectory_method").value)
        self._n_points = max(2, int(self.get_parameter("path_interpolation_points").value))
        self._segment_dt_s = float(self.get_parameter("fallback_segment_dt_s").value)

        self._status_pub = self.create_publisher(String, "~/trajectory_backend_status", 10)
        self._acados_available, self._acados_reason = self._check_acados_runtime()
        self._publish_backend_status(
            "AVAILABLE" if self._acados_available else "UNAVAILABLE",
            self._acados_reason,
        )

        # New stage-split services
        self._plan_geo_srv = self.create_service(
            PlanGeometricPath,
            "~/plan_geometric_path",
            self._handle_plan_geometric,
        )
        self._compute_traj_srv = self.create_service(
            ComputeTrajectory,
            "~/compute_trajectory",
            self._handle_compute_trajectory,
        )
        self._execute_traj_srv = self.create_service(
            ExecuteTrajectory,
            "~/execute_trajectory",
            self._handle_execute_trajectory,
        )

        # Deprecated compatibility services
        self._plan_legacy_srv = self.create_service(
            PlanBlockMotion,
            "~/plan_block_motion",
            self._handle_plan_block_motion_legacy,
        )
        self._execute_legacy_srv = self.create_service(
            ExecutePlannedMotion,
            "~/execute_planned_motion",
            self._handle_execute_planned_motion_legacy,
        )

        self.get_logger().info(
            "ConcreteBlockMotionPlanningNode ready | "
            f"default_geometric_method={self._default_geometric_method} | "
            f"default_trajectory_method={self._default_trajectory_method} | "
            f"acados_available={self._acados_available}"
        )

    def _handle_plan_geometric(
        self,
        request: PlanGeometricPath.Request,
        response: PlanGeometricPath.Response,
    ) -> PlanGeometricPath.Response:
        method = request.method.strip() or self._default_geometric_method
        geometric_plan_id = f"geo_{uuid.uuid4().hex[:8]}"

        result = self._build_geometric_plan(request.start_pose, request.goal_pose, method)
        result.geometric_plan_id = geometric_plan_id
        self._geometric_plans[geometric_plan_id] = result

        response.success = result.success
        response.geometric_plan_id = geometric_plan_id
        response.cartesian_path = result.path
        response.message = result.message
        return response

    def _handle_compute_trajectory(
        self,
        request: ComputeTrajectory.Request,
        response: ComputeTrajectory.Response,
    ) -> ComputeTrajectory.Response:
        method = request.method.strip() or self._default_trajectory_method

        path, geometric_plan_id, path_error = self._resolve_path_for_trajectory(request)
        if path_error is not None:
            response.success = False
            response.message = path_error
            return response

        if not self._acados_available:
            response.success = False
            response.message = (
                "ACADOS trajectory backend unavailable: "
                f"{self._acados_reason}. Install acados/casadi/pinocchio runtime."
            )
            self._publish_backend_status("UNAVAILABLE", response.message)
            return response

        trajectory = self._build_placeholder_trajectory(path)
        trajectory_id = f"traj_{uuid.uuid4().hex[:8]}"
        stored = StoredTrajectory(
            trajectory_id=trajectory_id,
            trajectory=trajectory,
            success=True,
            message=(
                f"Trajectory computed using method={method} (acados runtime detected; "
                "solver integration placeholder currently active)."
            ),
            method=method,
            geometric_plan_id=geometric_plan_id,
        )
        self._trajectories[trajectory_id] = stored
        if geometric_plan_id:
            self._latest_trajectory_for_geometric[geometric_plan_id] = trajectory_id

        response.success = True
        response.trajectory_id = trajectory_id
        response.trajectory = trajectory
        response.message = stored.message
        self._publish_backend_status("AVAILABLE", response.message)
        return response

    def _handle_execute_trajectory(
        self,
        request: ExecuteTrajectory.Request,
        response: ExecuteTrajectory.Response,
    ) -> ExecuteTrajectory.Response:
        stored = self._trajectories.get(request.trajectory_id)
        if stored is None:
            response.success = False
            response.message = f"Unknown trajectory_id '{request.trajectory_id}'."
            return response

        if not stored.success:
            response.success = False
            response.message = f"Trajectory '{request.trajectory_id}' is invalid: {stored.message}"
            return response

        if request.dry_run:
            response.success = True
            response.message = f"Dry-run execution accepted for '{request.trajectory_id}'."
            return response

        response.success = True
        response.message = (
            f"Execution accepted for '{request.trajectory_id}' "
            "(controller integration placeholder)."
        )
        return response

    def _handle_plan_block_motion_legacy(
        self,
        request: PlanBlockMotion.Request,
        response: PlanBlockMotion.Response,
    ) -> PlanBlockMotion.Response:
        self.get_logger().warn(
            "Deprecated service ~/plan_block_motion called. "
            "Please migrate to ~/plan_geometric_path + ~/compute_trajectory."
        )

        geo_req = PlanGeometricPath.Request()
        geo_req.start_pose = request.start_pose
        geo_req.goal_pose = request.goal_pose
        geo_req.target_block_id = request.target_block_id
        geo_req.reference_block_id = request.reference_block_id
        geo_req.method = request.planner_method
        geo_req.timeout_s = request.timeout_s
        geo_req.use_world_model = request.use_world_model

        geo_res = PlanGeometricPath.Response()
        geo_res = self._handle_plan_geometric(geo_req, geo_res)

        response.success = geo_res.success
        response.plan_id = geo_res.geometric_plan_id
        response.cartesian_path = geo_res.cartesian_path
        response.message = f"[DEPRECATED] {geo_res.message}"
        return response

    def _handle_execute_planned_motion_legacy(
        self,
        request: ExecutePlannedMotion.Request,
        response: ExecutePlannedMotion.Response,
    ) -> ExecutePlannedMotion.Response:
        self.get_logger().warn(
            "Deprecated service ~/execute_planned_motion called. "
            "Please migrate to ~/execute_trajectory."
        )

        trajectory_id = request.plan_id
        if trajectory_id not in self._trajectories:
            trajectory_id = self._latest_trajectory_for_geometric.get(request.plan_id, "")

        if not trajectory_id:
            response.success = False
            response.message = (
                f"[DEPRECATED] No trajectory found for id '{request.plan_id}'. "
                "Call ~/compute_trajectory first."
            )
            return response

        exec_req = ExecuteTrajectory.Request()
        exec_req.trajectory_id = trajectory_id
        exec_req.dry_run = request.dry_run

        exec_res = ExecuteTrajectory.Response()
        exec_res = self._handle_execute_trajectory(exec_req, exec_res)
        response.success = exec_res.success
        response.message = f"[DEPRECATED] {exec_res.message}"
        return response

    def _build_geometric_plan(
        self,
        start_pose: PoseStamped,
        goal_pose: PoseStamped,
        method: str,
    ) -> StoredGeometricPlan:
        # Deterministic geometric interpolation fallback for stage-1 contract stability.
        path = self._linear_path(start_pose, goal_pose, self._n_points)
        return StoredGeometricPlan(
            geometric_plan_id="",
            path=path,
            success=True,
            message=f"Geometric path computed with method={method}.",
            method=method,
        )

    def _resolve_path_for_trajectory(
        self,
        request: ComputeTrajectory.Request,
    ) -> Tuple[Optional[NavPath], str, Optional[str]]:
        if request.use_direct_path:
            if not request.direct_path.poses:
                return None, "", "Requested direct_path is empty."
            return request.direct_path, "", None

        if not request.geometric_plan_id:
            return None, "", "geometric_plan_id is required when use_direct_path=false."

        stored = self._geometric_plans.get(request.geometric_plan_id)
        if stored is None:
            return None, request.geometric_plan_id, f"Unknown geometric_plan_id '{request.geometric_plan_id}'."
        if not stored.success:
            return (
                None,
                request.geometric_plan_id,
                f"Geometric plan '{request.geometric_plan_id}' is invalid: {stored.message}",
            )
        return stored.path, request.geometric_plan_id, None

    def _build_placeholder_trajectory(self, path: NavPath) -> JointTrajectory:
        traj = JointTrajectory()
        traj.header = path.header
        traj.joint_names = ["virtual_progress_joint"]

        if not path.poses:
            return traj

        n = len(path.poses)
        for i in range(n):
            p = JointTrajectoryPoint()
            s = 0.0 if n == 1 else float(i) / float(n - 1)
            p.positions = [s]
            p.velocities = [0.0 if i == 0 else 1.0 / max(self._segment_dt_s, 1e-3)]
            total = i * self._segment_dt_s
            p.time_from_start = Duration(
                sec=int(total),
                nanosec=int((total - int(total)) * 1e9),
            )
            traj.points.append(p)

        return traj

    @staticmethod
    def _check_acados_runtime() -> Tuple[bool, str]:
        missing = []
        for module in ("acados_template", "casadi", "pinocchio"):
            try:
                __import__(module)
            except Exception:
                missing.append(module)

        if missing:
            return False, f"missing modules: {', '.join(missing)}"
        return True, "acados runtime modules are available"

    def _publish_backend_status(self, state: str, detail: str) -> None:
        msg = String()
        msg.data = f"trajectory_backend={state}; detail={detail}"
        self._status_pub.publish(msg)

    @staticmethod
    def _linear_path(start_pose: PoseStamped, goal_pose: PoseStamped, n_points: int) -> NavPath:
        out = NavPath()
        out.header = goal_pose.header
        out.header.frame_id = goal_pose.header.frame_id or start_pose.header.frame_id

        sx = float(start_pose.pose.position.x)
        sy = float(start_pose.pose.position.y)
        sz = float(start_pose.pose.position.z)

        gx = float(goal_pose.pose.position.x)
        gy = float(goal_pose.pose.position.y)
        gz = float(goal_pose.pose.position.z)

        for i in range(n_points):
            t = 0.0 if n_points == 1 else float(i) / float(n_points - 1)
            p = PoseStamped()
            p.header = out.header
            p.pose.position.x = (1.0 - t) * sx + t * gx
            p.pose.position.y = (1.0 - t) * sy + t * gy
            p.pose.position.z = (1.0 - t) * sz + t * gz
            p.pose.orientation = start_pose.pose.orientation if t < 0.5 else goal_pose.pose.orientation
            if math.isnan(p.pose.position.x) or math.isnan(p.pose.position.y) or math.isnan(p.pose.position.z):
                raise ValueError("Generated NaN in interpolation path.")
            out.poses.append(p)

        return out


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ConcreteBlockMotionPlanningNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
