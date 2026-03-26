from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from concrete_block_perception.msg import PlanningSceneObject
from concrete_block_perception.srv import GetCoarseBlocks, GetPlanningScene

from concrete_block_motion_planning.srv import (
    ComputeTrajectory,
    ExecuteNamedConfiguration,
    ExecuteTrajectory,
    GetNextAssemblyTask,
    PlanAndComputeTrajectory,
    PlanGeometricPath,
)

from .ids import make_geometric_plan_id, make_trajectory_id
from .types import StoredTrajectory


class ServiceHandlersMixin:
    def _resolve_planning_context(
        self,
        target_block_id: str,
        reference_block_id: str,
        use_world_model: bool,
        timeout_s: float,
    ) -> tuple[Dict[str, Any], str]:
        planning_context: Dict[str, Any] = {
            "target_block_id": target_block_id.strip(),
            "reference_block_id": reference_block_id.strip(),
            "use_world_model": bool(use_world_model),
            "world_model_blocks": [],
            "planning_scene_objects": [],
        }
        if not use_world_model:
            return planning_context, "world model disabled by request"

        if self._get_planning_scene_client is None:
            return planning_context, "planning-scene service client not initialized; fallback to manual planning"
        if not self._get_planning_scene_client.wait_for_service(timeout_sec=1.0):
            return planning_context, (
                f"planning-scene service '{self._world_model_get_planning_scene_service}' unavailable; "
                "fallback to manual planning"
            )

        req = GetPlanningScene.Request()
        req.force_refresh = False
        req.timeout_s = max(0.0, float(timeout_s))
        req.query_stamp = self.get_clock().now().to_msg()
        try:
            res = self._get_planning_scene_client.call(req)
        except Exception as exc:
            return planning_context, f"world model request timed out; fallback to manual planning ({exc})"
        if res is None:
            return planning_context, "world model request returned no response; fallback to manual planning"
        if not res.success:
            return planning_context, f"world model request failed: {res.message}; fallback to manual planning"

        scene_payload = []
        block_payload = []
        for obj in res.scene.objects:
            payload = {
                "id": str(obj.id),
                "frame_id": str(obj.frame_id),
                "pose": obj.pose,
                "shape_type": int(obj.shape_type),
                "source_type": int(obj.source_type),
                "dimensions": (
                    float(obj.dimensions.x),
                    float(obj.dimensions.y),
                    float(obj.dimensions.z),
                ),
                "task_status": int(obj.task_status),
                "pose_status": int(obj.pose_status),
                "confidence": float(obj.confidence),
            }
            scene_payload.append(payload)
            if int(obj.source_type) == int(PlanningSceneObject.SOURCE_BLOCK):
                block_payload.append(payload)
        planning_context["planning_scene_objects"] = scene_payload
        planning_context["world_model_blocks"] = block_payload
        return planning_context, (
            f"world model ok ({len(scene_payload)} objects, {len(block_payload)} blocks)"
        )

    def _handle_plan_geometric(
        self,
        request: PlanGeometricPath.Request,
        response: PlanGeometricPath.Response,
    ) -> PlanGeometricPath.Response:
        if self._planner_backend.backend_name != "concrete":
            response.success = False
            response.message = (
                "Geometric stage is unavailable for planner backend "
                f"'{self._planner_backend.backend_name}'. Use plan_and_compute_trajectory instead."
            )
            return response
        method = request.method.strip() or self._default_geometric_method
        geometric_plan_id = make_geometric_plan_id()
        planning_context, context_msg = self._resolve_planning_context(
            target_block_id=request.target_block_id,
            reference_block_id=request.reference_block_id,
            use_world_model=bool(request.use_world_model),
            timeout_s=float(request.timeout_s),
        )

        result = self._build_geometric_plan(
            request.start_pose,
            request.goal_pose,
            method,
            planning_context=planning_context,
        )
        if context_msg:
            result.message = f"{result.message} | planning_context={context_msg}"
        result.geometric_plan_id = geometric_plan_id
        self._geometric_plans[geometric_plan_id] = result

        response.success = result.success
        response.geometric_plan_id = geometric_plan_id
        response.cartesian_path = result.path
        response.message = result.message
        if result.success and result.path is not None:
            self._planned_path_pub.publish(result.path)
        return response

    def _handle_plan_and_compute_trajectory(
        self,
        request: PlanAndComputeTrajectory.Request,
        response: PlanAndComputeTrajectory.Response,
    ) -> PlanAndComputeTrajectory.Response:
        geometric_method = request.geometric_method.strip() or self._default_geometric_method
        trajectory_method = request.trajectory_method.strip() or self._default_trajectory_method
        planning_context, context_msg = self._resolve_planning_context(
            target_block_id=request.target_block_id,
            reference_block_id=request.reference_block_id,
            use_world_model=bool(request.use_world_model),
            timeout_s=float(request.geometric_timeout_s),
        )
        result = self._planner_backend.plan_move_empty(
            start_pose=request.start_pose,
            goal_pose=request.goal_pose,
            geometric_method=geometric_method,
            geometric_timeout_s=float(request.geometric_timeout_s),
            trajectory_method=trajectory_method,
            trajectory_timeout_s=float(request.trajectory_timeout_s),
            validate_dynamics=bool(request.validate_dynamics),
            planning_context={
                **planning_context,
                "use_world_model": bool(request.use_world_model),
            },
        )
        response.geometric_plan_id = result.geometric_plan_id
        response.cartesian_path = result.cartesian_path
        response.success = bool(result.success)
        response.trajectory = result.trajectory
        response.trajectory_id = ""
        if result.success:
            trajectory_id = make_trajectory_id()
            self._trajectories[trajectory_id] = StoredTrajectory(
                trajectory_id=trajectory_id,
                trajectory=result.trajectory,
                success=True,
                message=result.message,
                method=self._planner_backend.backend_name.upper(),
                geometric_plan_id=result.geometric_plan_id,
                cartesian_path=result.cartesian_path,
            )
            response.trajectory_id = trajectory_id
            response.cartesian_path = result.cartesian_path
        response.message = (
            f"{result.message} | planning_context={context_msg}"
            if context_msg
            else result.message
        )
        if not result.success:
            self.get_logger().warn(
                "PlanAndComputeTrajectory aborted | "
                f"backend={self._planner_backend.backend_name} "
                f"geometric_method={geometric_method} "
                f"trajectory_method={trajectory_method} "
                f"target={request.target_block_id or '<none>'} "
                f"reference={request.reference_block_id or '<none>'} "
                f"message={response.message}"
            )
        if result.success and result.cartesian_path is not None:
            self._planned_path_pub.publish(result.cartesian_path)
            if result.cartesian_path.poses:
                self._planned_trajectory_path_pub.publish(result.cartesian_path)
        return response

    def _handle_compute_trajectory(
        self,
        request: ComputeTrajectory.Request,
        response: ComputeTrajectory.Response,
    ) -> ComputeTrajectory.Response:
        if self._planner_backend.backend_name != "concrete":
            response.success = False
            response.message = (
                "compute_trajectory is unavailable for planner backend "
                f"'{self._planner_backend.backend_name}'."
            )
            return response
        method = request.method.strip() or self._default_trajectory_method
        method_upper = method.strip().upper()

        if not self._trajectory_runtime_available:
            response.success = False
            response.message = (
                "Trajectory backend unavailable: "
                f"{self._trajectory_runtime_reason}."
            )
            self._publish_backend_status("UNAVAILABLE", response.message)
            return response

        path, geometric_plan_id, path_error = self._resolve_path_for_trajectory(request)
        if path_error is not None:
            response.success = False
            response.message = path_error
            return response

        if not self._planning_runtime_ready:
            response.success = False
            response.message = (
                "Planning runtime unavailable: "
                f"{self._planning_runtime_reason}."
            )
            self._publish_backend_status("UNAVAILABLE", response.message)
            return response

        if path is None or not path.poses:
            response.success = False
            response.message = "Cannot compute trajectory from empty path."
            return response
        self._planned_path_pub.publish(path)

        start_pose = path.poses[0]
        goal_pose = path.poses[-1]

        if method_upper in (
            "TOPPRA_PATH_FOLLOWING",
            "TOPPRA",
            "TIME_PARAMETERIZED_PATH_FOLLOWING",
        ):
            start_ok, q_start, start_msg = self._solve_canonical_q_from_pose(start_pose)
            if not start_ok:
                response.success = False
                response.message = f"Failed to solve canonical start IK/steady-state: {start_msg}"
                return response

            goal_ok, q_goal, goal_msg = self._solve_canonical_q_from_pose(goal_pose)
            if not goal_ok:
                response.success = False
                response.message = f"Failed to solve canonical goal IK/steady-state: {goal_msg}"
                return response
            try:
                stored = self._build_joint_space_cartesian_trajectory(
                    path=path,
                    geometric_plan_id=geometric_plan_id,
                    method=method_upper,
                    q_start=q_start,
                    q_goal=q_goal,
                )
            except Exception as exc:
                response.success = False
                response.message = (
                    "TOPP-RA trajectory generation failed in concrete joint-space stage: "
                    f"{exc} | start_q={np.array2string(np.asarray(q_start, dtype=float), precision=3, suppress_small=True)} "
                    f"| goal_q={np.array2string(np.asarray(q_goal, dtype=float), precision=3, suppress_small=True)} "
                    f"| start_q_named={self._format_named_joint_vector(self._canonical_online_joint_names(), q_start)} "
                    f"| goal_q_named={self._format_named_joint_vector(self._canonical_online_joint_names(), q_goal)} "
                    f"| path_points={len(path.poses)}"
                )
                self._publish_backend_status("UNAVAILABLE", response.message)
                return response
            try:
                self._trajectories[stored.trajectory_id] = stored
                response.success = True
                response.trajectory_id = stored.trajectory_id
                response.trajectory = stored.trajectory
                response.message = stored.message
                if stored.cartesian_path.poses:
                    self._planned_path_pub.publish(stored.cartesian_path)
                    self._planned_trajectory_path_pub.publish(stored.cartesian_path)
                self._publish_backend_status("AVAILABLE", stored.message)
                return response
            except Exception as exc:
                response.success = False
                response.message = f"TOPP-RA trajectory staging failed: {exc}"
                self._publish_backend_status("UNAVAILABLE", response.message)
                return response

        start_ok, q_start, start_msg = self._solve_reduced_q_from_pose(start_pose)
        if not start_ok:
            response.success = False
            response.message = f"Failed to solve start IK/steady-state: {start_msg}"
            return response

        goal_ok, q_goal, goal_msg = self._solve_reduced_q_from_pose(goal_pose)
        if not goal_ok:
            response.success = False
            response.message = f"Failed to solve goal IK/steady-state: {goal_msg}"
            return response

        if method_upper in (
            "FIXED_TIME_INTERPOLATION",
            "FIXED_JOINT_INTERPOLATION",
            "JOINT_INTERPOLATION",
            "LINEAR_JOINT_INTERPOLATION",
        ):
            stored = self._build_fixed_time_interpolation_trajectory(
                q_start=q_start,
                q_goal=q_goal,
                duration_s=self._traj_fixed_duration_s,
                num_points=self._traj_fixed_num_points,
                method=method_upper,
                geometric_plan_id=geometric_plan_id,
                path=path,
            )
            self._trajectories[stored.trajectory_id] = stored
            response.success = True
            response.trajectory_id = stored.trajectory_id
            response.trajectory = stored.trajectory
            response.message = stored.message
            self._publish_backend_status("AVAILABLE", stored.message)
            return response

        response.success = False
        response.message = (
            "Unsupported trajectory method for online concrete backend: "
            f"'{method_upper}'. Use TOPPRA_PATH_FOLLOWING or FIXED_TIME_INTERPOLATION."
        )
        self._publish_backend_status("UNAVAILABLE", response.message)
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
            response.message = f"Dry-run accepted for '{request.trajectory_id}'."
            return response

        if (
            self._planner_backend.backend_name == "concrete"
            and (
                "JOINT_LINEAR_INTERPOLATION" in str(stored.message)
                or "joint_space_stage_fallback=legacy_path_ik" in str(stored.message)
                or "TOPP-RA path fallback via JOINT_GOAL_LINEARIZATION" in str(stored.message)
            )
        ):
            response.success = False
            response.message = (
                "Refusing to execute concrete fallback trajectory "
                f"'{request.trajectory_id}': planner produced a fallback-only concrete path "
                "(legacy path IK or joint linearization), which is acceptable for plan "
                "inspection but not for live execution."
            )
            return response

        response.success, response.message = self._execution_adapter.dispatch_trajectory(
            stored.trajectory,
            request.trajectory_id,
        )
        return response

    def _handle_execute_named_configuration(
        self,
        request: ExecuteNamedConfiguration.Request,
        response: ExecuteNamedConfiguration.Response,
    ) -> ExecuteNamedConfiguration.Response:
        resolution = self._named_configuration_resolver.resolve(
            request.configuration_name
        )
        response.trajectory_id = resolution.trajectory_id
        if not resolution.success or resolution.trajectory is None:
            response.success = False
            response.message = resolution.message
            return response

        if request.dry_run:
            response.success = True
            response.message = (
                f"{resolution.message} Dry-run accepted."
            )
            return response

        response.success, dispatch_msg = self._execution_adapter.dispatch_trajectory(
            resolution.trajectory,
            resolution.trajectory_id,
        )
        response.message = (
            f"{resolution.message} {dispatch_msg}"
        )
        return response

    def _handle_get_next_assembly_task(
        self,
        request: GetNextAssemblyTask.Request,
        response: GetNextAssemblyTask.Response,
    ) -> GetNextAssemblyTask.Response:
        plan_name = request.wall_plan_name.strip() or self._default_wall_plan_name
        plan_name = plan_name.lower()
        if plan_name not in self._wall_plans:
            available = ", ".join(sorted(self._wall_plans.keys()))
            response.success = False
            response.has_task = False
            response.message = (
                f"Unknown wall plan '{plan_name}'. Available: [{available}]"
            )
            return response

        if request.reset_plan:
            self._wall_plan_progress[plan_name] = 0

        idx = self._wall_plan_progress.get(plan_name, 0)
        tasks = self._wall_plans[plan_name]
        if idx >= len(tasks):
            response.success = True
            response.has_task = False
            response.message = f"Wall plan '{plan_name}' completed."
            return response

        task = tasks[idx]
        self._wall_plan_progress[plan_name] = idx + 1

        response.success = True
        response.has_task = True
        response.task_id = task.task_id
        response.target_block_id = task.target_block_id
        response.reference_block_id = task.reference_block_id
        response.target_pose = task.target_pose
        response.reference_pose = task.reference_pose
        response.message = (
            f"Task {idx + 1}/{len(tasks)} from wall plan '{plan_name}': {task.task_id}"
        )
        return response
