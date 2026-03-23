from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import SwitchController
from concrete_block_perception.srv import GetCoarseBlocks

from concrete_block_motion_planning.srv import (
    ComputeTrajectory,
    ExecuteNamedConfiguration,
    ExecuteTrajectory,
    GetNextAssemblyTask,
    PlanAndComputeTrajectory,
    PlanGeometricPath,
)

from .ids import make_geometric_plan_id, make_named_trajectory_id, make_trajectory_id
from .types import StoredTrajectory


class ServiceHandlersMixin:
    def _switch_execution_controller(
        self,
        activate: bool,
        timeout_s: float = 2.0,
    ) -> Tuple[bool, str]:
        if not self._execution_switch_controller:
            return True, "controller switching disabled"
        if self._switch_controller_client is None:
            return False, "controller switch client is not initialized"
        if not self._execution_activate_controller:
            return False, "execution.activate_controller is empty"
        if not self._switch_controller_client.wait_for_service(timeout_sec=timeout_s):
            return (
                False,
                f"controller switch service '{self._execution_switch_service}' unavailable",
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
            req.activate_controllers = [self._execution_activate_controller]
            req.deactivate_controllers = []
        else:
            req.activate_controllers = []
            req.deactivate_controllers = [self._execution_activate_controller]

        future = self._switch_controller_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        if not future.done():
            return False, "controller switch call timed out"
        res = future.result()
        if res is None:
            return False, "controller switch call returned no response"
        if not bool(res.ok):
            return False, "controller switch request rejected"

        action = "activated" if activate else "deactivated"
        return True, f"controller '{self._execution_activate_controller}' {action}"

    def _dispatch_trajectory_topic(self, trajectory, trajectory_id: str) -> tuple[bool, str]:
        if self._trajectory_cmd_pub is None:
            return (
                False,
                "Execution publisher is not initialized.",
            )
        if self._trajectory_cmd_pub.get_subscription_count() <= 0:
            return (
                False,
                f"No subscribers on execution topic '{self._execution_trajectory_topic}'.",
            )
        self._trajectory_cmd_pub.publish(trajectory)
        return (
            True,
            f"Dispatched trajectory '{trajectory_id}' to '{self._execution_trajectory_topic}'.",
        )

    def _dispatch_trajectory_action(self, trajectory, trajectory_id: str) -> tuple[bool, str]:
        if self._trajectory_action_client is None:
            return False, "Execution action client is not initialized."
        if not self._trajectory_action_client.wait_for_server(timeout_sec=2.0):
            return (
                False,
                f"Execution action server '{self._execution_action_name}' unavailable.",
            )

        switched, switch_msg = self._switch_execution_controller(activate=True)
        if not switched:
            return False, f"Failed to activate execution controller: {switch_msg}"

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        send_future = self._trajectory_action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=5.0)
        if not send_future.done():
            return False, "Timed out while sending trajectory action goal."
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return (
                False,
                f"Execution action goal rejected by '{self._execution_action_name}'.",
            )

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self,
            result_future,
            timeout_sec=float(self._execution_result_timeout_s),
        )
        if not result_future.done():
            return (
                False,
                "Timed out waiting for trajectory action result "
                f"(timeout={self._execution_result_timeout_s:.1f}s).",
            )

        wrapped = result_future.result()
        if wrapped is None:
            return False, "Trajectory action completed with empty result."
        result = wrapped.result
        status_ok = wrapped.status == GoalStatus.STATUS_SUCCEEDED
        code_ok = int(result.error_code) == int(result.SUCCESSFUL)
        ok = bool(status_ok and code_ok)
        if self._execution_deactivate_after_execution:
            switched_down, switch_down_msg = self._switch_execution_controller(activate=False)
            if not switched_down:
                self.get_logger().warn(
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
            f"Trajectory '{trajectory_id}' executed via action '{self._execution_action_name}'.",
        )

    def _dispatch_trajectory(self, trajectory, trajectory_id: str) -> tuple[bool, str]:
        if not self._execution_enabled:
            return (
                False,
                "Execution disabled (execution.enabled=false).",
            )
        if not trajectory.points:
            return (
                False,
                f"Trajectory '{trajectory_id}' has no points.",
            )
        if self._execution_backend == "action":
            return self._dispatch_trajectory_action(trajectory, trajectory_id)
        if self._execution_backend != "topic":
            return (
                False,
                f"Unsupported execution backend '{self._execution_backend}'.",
            )
        return self._dispatch_trajectory_topic(trajectory, trajectory_id)

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
        }
        if not use_world_model:
            return planning_context, "world model disabled by request"

        if self._get_coarse_blocks_client is None:
            return planning_context, "world model service client not initialized; fallback to manual planning"
        if not self._get_coarse_blocks_client.wait_for_service(timeout_sec=1.0):
            return planning_context, (
                f"world model service '{self._world_model_get_coarse_blocks_service}' unavailable; "
                "fallback to manual planning"
            )

        req = GetCoarseBlocks.Request()
        req.force_refresh = False
        req.timeout_s = max(0.0, float(timeout_s))
        req.query_stamp = self.get_clock().now().to_msg()
        future = self._get_coarse_blocks_client.call_async(req)
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=max(1.0, float(timeout_s) + 0.5),
        )
        if not future.done():
            return planning_context, "world model request timed out; fallback to manual planning"
        res = future.result()
        if res is None:
            return planning_context, "world model request returned no response; fallback to manual planning"
        if not res.success:
            return planning_context, f"world model request failed: {res.message}; fallback to manual planning"

        block_payload = []
        for b in res.blocks.blocks:
            block_payload.append(
                {
                    "id": str(b.id),
                    "pose": b.pose,
                    "task_status": int(b.task_status),
                    "pose_status": int(b.pose_status),
                }
            )
        planning_context["world_model_blocks"] = block_payload
        return planning_context, f"world model ok ({len(block_payload)} blocks)"

    def _handle_plan_geometric(
        self,
        request: PlanGeometricPath.Request,
        response: PlanGeometricPath.Response,
    ) -> PlanGeometricPath.Response:
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
        return response

    def _handle_plan_and_compute_trajectory(
        self,
        request: PlanAndComputeTrajectory.Request,
        response: PlanAndComputeTrajectory.Response,
    ) -> PlanAndComputeTrajectory.Response:
        plan_req = PlanGeometricPath.Request()
        plan_req.start_pose = request.start_pose
        plan_req.goal_pose = request.goal_pose
        plan_req.target_block_id = request.target_block_id
        plan_req.reference_block_id = request.reference_block_id
        plan_req.method = request.geometric_method
        plan_req.timeout_s = request.geometric_timeout_s
        plan_req.use_world_model = request.use_world_model
        plan_res = self._handle_plan_geometric(plan_req, PlanGeometricPath.Response())
        if not plan_res.success:
            response.success = False
            response.geometric_plan_id = plan_res.geometric_plan_id
            response.cartesian_path = plan_res.cartesian_path
            response.message = f"Geometric planning failed: {plan_res.message}"
            return response

        traj_req = ComputeTrajectory.Request()
        traj_req.geometric_plan_id = plan_res.geometric_plan_id
        traj_req.method = request.trajectory_method
        traj_req.timeout_s = request.trajectory_timeout_s
        traj_req.validate_dynamics = request.validate_dynamics
        traj_res = self._handle_compute_trajectory(traj_req, ComputeTrajectory.Response())
        response.geometric_plan_id = plan_res.geometric_plan_id
        response.cartesian_path = plan_res.cartesian_path
        response.trajectory_id = traj_res.trajectory_id
        response.trajectory = traj_res.trajectory
        response.success = bool(traj_res.success)
        if traj_res.success:
            response.message = (
                "Combined plan+compute success. "
                f"{plan_res.message} | {traj_res.message}"
            )
        else:
            response.message = (
                "Trajectory stage failed after geometric success. "
                f"{plan_res.message} | {traj_res.message}"
            )
        return response

    def _handle_compute_trajectory(
        self,
        request: ComputeTrajectory.Request,
        response: ComputeTrajectory.Response,
    ) -> ComputeTrajectory.Response:
        method = request.method.strip() or self._default_trajectory_method

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

        start_pose = path.poses[0]
        goal_pose = path.poses[-1]

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

        ctrl_pts_xyz, ctrl_pts_yaw = self._extract_control_points(path)

        try:
            optimizer, optimizer_kind = self._get_trajectory_optimizer(method)
            from motion_planning.core.types import TrajectoryRequest

            req_cfg: Dict[str, Any] = {
                "q0": q_start,
                "q_goal": q_goal,
                "dq0": np.zeros_like(q_start),
                "ctrl_pts_xyz": ctrl_pts_xyz,
                "ctrl_pts_yaw": ctrl_pts_yaw,
                "spline_ctrl_points": int(ctrl_pts_xyz.shape[0]),
                "acados_verbose": bool(self._traj_acados_verbose),
                # Work around the current free-time OCP sensitivity by solving
                # a fixed-duration path-following problem for runtime use.
                "optimize_time": False,
                "fixed_time_duration_s": 10.0,
                "fixed_time_duration_candidates": (10.0,),
                "T_min": 10.0,
                "T_max": 10.0,
            }

            method_upper = method.upper()
            if "FAST" in method_upper:
                req_cfg.update(
                    {
                        "nlp_solver_max_iter": 150,
                        "qp_solver_iter_max": 80,
                        "terminal_hold_steps": 0,
                    }
                )
            elif "STABLE" in method_upper or "COMMISSION" in method_upper:
                req_cfg.update(
                    {
                        "nlp_solver_max_iter": 350,
                        "qp_solver_iter_max": 140,
                        "terminal_hold_steps": 4,
                    }
                )

            if request.timeout_s > 0.0:
                req_cfg["nlp_solver_max_iter"] = max(
                    int(req_cfg.get("nlp_solver_max_iter", 300)),
                    int(40.0 * float(request.timeout_s)),
                )

            traj_req = TrajectoryRequest(
                scenario=None,
                path=None,
                config=req_cfg,
            )
            traj_result = optimizer.optimize(traj_req)
        except Exception as exc:
            response.success = False
            response.message = f"Trajectory optimization failed: {exc}"
            self._publish_backend_status("UNAVAILABLE", response.message)
            return response

        trajectory = self._trajectory_result_to_joint_trajectory(traj_result)
        trajectory_id = make_trajectory_id()
        stored = StoredTrajectory(
            trajectory_id=trajectory_id,
            trajectory=trajectory,
            success=bool(traj_result.success),
            message=traj_result.message,
            method=optimizer_kind,
            geometric_plan_id=geometric_plan_id,
        )
        self._trajectories[trajectory_id] = stored

        response.success = stored.success
        response.trajectory_id = trajectory_id
        response.trajectory = trajectory
        response.message = (
            f"Trajectory computed using {optimizer_kind}. "
            f"{stored.message}"
        )
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
            response.message = f"Dry-run accepted for '{request.trajectory_id}'."
            return response

        response.success, response.message = self._dispatch_trajectory(
            stored.trajectory,
            request.trajectory_id,
        )
        return response

    def _handle_execute_named_configuration(
        self,
        request: ExecuteNamedConfiguration.Request,
        response: ExecuteNamedConfiguration.Response,
    ) -> ExecuteNamedConfiguration.Response:
        cfg_name = request.configuration_name.strip()
        if not cfg_name:
            response.success = False
            response.message = "configuration_name must not be empty."
            return response

        trajectory = self._named_configurations.get(cfg_name)
        if trajectory is None:
            available = ", ".join(sorted(self._named_configurations.keys()))
            response.success = False
            response.message = (
                f"Unknown named configuration '{cfg_name}'. "
                f"Available: [{available}]"
            )
            return response

        trajectory_id = make_named_trajectory_id(cfg_name)
        self._trajectories[trajectory_id] = StoredTrajectory(
            trajectory_id=trajectory_id,
            trajectory=trajectory,
            success=True,
            message=f"Named configuration trajectory '{cfg_name}'.",
            method="NAMED_CONFIGURATION",
            geometric_plan_id="",
        )
        response.trajectory_id = trajectory_id

        if request.dry_run:
            response.success = True
            response.message = (
                f"Dry-run accepted for named configuration '{cfg_name}' "
                f"(trajectory_id={trajectory_id})."
            )
            return response

        response.success, dispatch_msg = self._dispatch_trajectory(trajectory, trajectory_id)
        response.message = (
            f"Named configuration '{cfg_name}' converted to trajectory_id={trajectory_id}. "
            f"{dispatch_msg}"
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
