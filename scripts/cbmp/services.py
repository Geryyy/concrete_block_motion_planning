from __future__ import annotations

from typing import Any, Dict

import numpy as np

from concrete_block_motion_planning.srv import (
    ComputeTrajectory,
    ExecuteNamedConfiguration,
    ExecuteTrajectory,
    GetNextAssemblyTask,
    PlanGeometricPath,
)

from .ids import make_geometric_plan_id, make_named_trajectory_id, make_trajectory_id
from .types import StoredTrajectory


class ServiceHandlersMixin:
    def _handle_plan_geometric(
        self,
        request: PlanGeometricPath.Request,
        response: PlanGeometricPath.Response,
    ) -> PlanGeometricPath.Response:
        method = request.method.strip() or self._default_geometric_method
        geometric_plan_id = make_geometric_plan_id()

        result = self._build_geometric_plan(
            request.start_pose,
            request.goal_pose,
            method,
        )
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
            }

            if "FAST" in method.upper() or not bool(request.validate_dynamics):
                req_cfg.update(
                    {
                        "nlp_solver_max_iter": 150,
                        "qp_solver_iter_max": 80,
                        "terminal_hold_steps": 0,
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

        response.success = False
        response.message = (
            "Planning-only node: execution is disabled. "
            "Send this JointTrajectory to an A2B execution server (feed-forward / MPC / jerk) "
            "outside this node."
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

        response.success = False
        response.message = (
            f"Named configuration '{cfg_name}' converted to trajectory_id={trajectory_id}, "
            "but execution is disabled in this planning-only node. "
            "Dispatch externally to A2B execution backends."
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
