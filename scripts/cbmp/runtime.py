from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Path as NavPath
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .ids import make_trajectory_id
from .types import StoredGeometricPlan, StoredTrajectory, WallPlanTask


class RuntimeHelpersMixin:
    def _canonical_online_joint_names(self) -> list[str]:
        if self._analytic_cfg is None:
            return list(self._reduced_joint_names)
        return [str(name) for name in self._analytic_cfg.actuated_joints]

    @staticmethod
    def _format_named_joint_vector(names: Sequence[str], values: Sequence[float]) -> str:
        pairs = []
        for name, value in zip(names, values):
            pairs.append(f"{name}={float(value):.3f}")
        return "[" + ", ".join(pairs) + "]"

    def _fk_nav_path_from_joint_trajectory(
        self,
        trajectory: JointTrajectory,
        *,
        frame_id: str = "world",
    ) -> NavPath:
        nav_path = NavPath()
        nav_path.header.frame_id = frame_id
        if not trajectory.points:
            return nav_path

        planner = self._get_joint_space_planner()
        if planner is None:
            return nav_path

        joint_names = [str(name) for name in trajectory.joint_names]
        canonical_names = self._canonical_online_joint_names()
        if not joint_names or not canonical_names:
            return nav_path

        for point in trajectory.points:
            joint_map = {
                name: float(point.positions[idx])
                for idx, name in enumerate(joint_names)
                if idx < len(point.positions)
            }
            q = self._canonical_q_from_joint_map(joint_map)
            xyz, yaw, _ = planner.fk_world_pose(q)
            pose = PoseStamped()
            pose.header.frame_id = frame_id
            pose.pose.position.x = float(xyz[0])
            pose.pose.position.y = float(xyz[1])
            pose.pose.position.z = float(xyz[2])
            pose.pose.orientation = self._yaw_to_quaternion(float(yaw))
            nav_path.poses.append(pose)
        return nav_path

    def _build_geometric_plan(
        self,
        start_pose: PoseStamped,
        goal_pose: PoseStamped,
        method: str,
        planning_context: Optional[Dict[str, Any]] = None,
    ) -> StoredGeometricPlan:
        if not self._geometric_runtime_available:
            return StoredGeometricPlan(
                geometric_plan_id="",
                path=NavPath(),
                success=False,
                message=(
                    f"Geometric backend unavailable: {self._geometric_runtime_reason}"
                ),
                method=method,
            )

        if not self._planning_runtime_ready:
            return StoredGeometricPlan(
                geometric_plan_id="",
                path=NavPath(),
                success=False,
                message=(
                    "Geometric planning backend unavailable: "
                    f"{self._planning_runtime_reason}"
                ),
                method=method,
            )

        try:
            from motion_planning.types import Scenario
            from motion_planning.optimized_params import canonical_method_name
            from motion_planning.api import run_geometric_planning
        except Exception as exc:
            return StoredGeometricPlan(
                geometric_plan_id="",
                path=NavPath(),
                success=False,
                message=f"Failed to import geometric planning modules: {exc}",
                method=method,
            )

        planning_context = dict(planning_context or {})
        scene_ok, scene_reason = self._ensure_planner_scene()
        if not scene_ok:
            return StoredGeometricPlan(
                geometric_plan_id="",
                path=NavPath(),
                success=False,
                message=f"Geometric scene backend unavailable: {scene_reason}",
                method=method,
            )

        start_xyz = self._pose_xyz(start_pose)
        goal_xyz = self._pose_xyz(goal_pose)
        start_yaw = math.degrees(self._quaternion_to_yaw(start_pose.pose.orientation))
        goal_yaw = math.degrees(self._quaternion_to_yaw(goal_pose.pose.orientation))

        canonical = canonical_method_name(method)
        cfg: Dict[str, Any] = {}
        opts: Dict[str, Any] = {}
        if canonical in self._optimized_planner_params:
            entry = self._optimized_planner_params[canonical]
            cfg = dict(entry.get("config", {}))
            opts = dict(entry.get("options", {}))

        n_vias = int(cfg.get("n_vias", 2))
        if n_vias > 0:
            t = np.linspace(0.0, 1.0, n_vias + 2, dtype=float)[1:-1]
            start_arr = np.asarray(start_xyz, dtype=float).reshape(3)
            goal_arr = np.asarray(goal_xyz, dtype=float).reshape(3)
            cfg["initial_vias"] = np.vstack(
                [(1.0 - a) * start_arr + a * goal_arr for a in t]
            )

        scenario_scene = self._planner_scene
        world_model_note = ""
        scene_objects = list(planning_context.get("planning_scene_objects", []))
        world_blocks = list(planning_context.get("world_model_blocks", []))
        if scene_objects:
            built_scene, world_model_note = self._scene_from_planning_scene_objects(scene_objects)
            if built_scene is not None:
                scenario_scene = built_scene
            elif world_model_note:
                world_model_note = f"{world_model_note}; fallback to base scene"

        goal_normals = self._resolve_goal_approach_normals(
            start_xyz=start_xyz,
            goal_xyz=goal_xyz,
            planning_context=planning_context,
            world_blocks=world_blocks,
        )

        scenario = Scenario(
            scene=scenario_scene,
            start=start_xyz,
            goal=goal_xyz,
            moving_block_size=self._moving_block_size,
            start_yaw_deg=float(start_yaw),
            goal_yaw_deg=float(goal_yaw),
            goal_normals=goal_normals,
        )

        try:
            geo_result = run_geometric_planning(
                scenario=scenario,
                method=canonical,
                config=cfg,
                options=opts,
            )
        except Exception as exc:
            return StoredGeometricPlan(
                geometric_plan_id="",
                path=NavPath(),
                success=False,
                message=f"Geometric planning failed: {exc}",
                method=canonical,
            )

        nav_path = NavPath()
        nav_path.header = goal_pose.header
        nav_path.header.frame_id = (
            goal_pose.header.frame_id or start_pose.header.frame_id or "world"
        )

        if geo_result.path is not None:
            sample_count = max(2, self._n_points)
            xyz = np.asarray(geo_result.path.sample(sample_count), dtype=float).reshape(
                -1, 3
            )
            yaw = np.asarray(
                geo_result.path.sample_yaw(sample_count), dtype=float
            ).reshape(-1)
            if yaw.shape[0] != xyz.shape[0]:
                yaw = np.linspace(
                    self._quaternion_to_yaw(start_pose.pose.orientation),
                    self._quaternion_to_yaw(goal_pose.pose.orientation),
                    xyz.shape[0],
                )

            for i in range(xyz.shape[0]):
                p = PoseStamped()
                p.header = nav_path.header
                p.pose.position.x = float(xyz[i, 0])
                p.pose.position.y = float(xyz[i, 1])
                p.pose.position.z = float(xyz[i, 2])
                p.pose.orientation = self._yaw_to_quaternion(float(yaw[i]))
                nav_path.poses.append(p)

        metrics = geo_result.metrics or {}
        metric_txt = ", ".join([f"{k}={float(v):.4f}" for k, v in metrics.items()])
        msg = geo_result.message
        if metric_txt:
            msg = f"{msg} | {metric_txt}"
        if world_model_note:
            msg = f"{msg} | {world_model_note}"

        return StoredGeometricPlan(
            geometric_plan_id="",
            path=nav_path,
            success=bool(geo_result.success),
            message=msg,
            method=canonical,
            planner_path=geo_result.path,
        )

    def _scene_from_planning_scene_objects(
        self,
        scene_objects: List[Dict[str, Any]],
    ) -> Tuple[Optional[Any], str]:
        try:
            from motion_planning import Scene
        except Exception as exc:
            return None, f"cannot import Scene for world model integration: {exc}"

        scene = Scene()
        added = 0
        for item in scene_objects:
            pose = item.get("pose")
            if pose is None:
                continue
            try:
                position = (
                    float(pose.position.x),
                    float(pose.position.y),
                    float(pose.position.z),
                )
                quat = (
                    float(pose.orientation.x),
                    float(pose.orientation.y),
                    float(pose.orientation.z),
                    float(pose.orientation.w),
                )
                oid = str(item.get("id", "")).strip() or None
                scene.add_block(
                    size=item.get("dimensions", self._moving_block_size),
                    position=position,
                    quat=quat,
                    object_id=oid,
                )
                added += 1
            except Exception:
                continue
        if added <= 0:
            return None, "world model returned no valid planning-scene objects"
        return scene, f"world_model_objects_used={added}"

    def _resolve_goal_approach_normals(
        self,
        start_xyz: Tuple[float, float, float],
        goal_xyz: Tuple[float, float, float],
        planning_context: Dict[str, Any],
        world_blocks: List[Dict[str, Any]],
    ) -> Tuple[Tuple[float, float, float], ...]:
        fallback = ((0.0, 0.0, 1.0),)
        target_id = str(planning_context.get("target_block_id", "")).strip()
        reference_id = str(planning_context.get("reference_block_id", "")).strip()
        if not target_id:
            return fallback

        target = None
        reference = None
        for item in world_blocks:
            bid = str(item.get("id", "")).strip()
            if bid == target_id:
                target = item
            if bid == reference_id:
                reference = item
        if target is None:
            return fallback

        try:
            tp = target["pose"].position
            target_vec = np.array([float(tp.x), float(tp.y), float(tp.z)], dtype=float)
        except Exception:
            return fallback

        if reference is not None:
            try:
                rp = reference["pose"].position
                ref_vec = np.array([float(rp.x), float(rp.y), float(rp.z)], dtype=float)
                n = target_vec - ref_vec
                n_norm = float(np.linalg.norm(n))
                if n_norm > 1e-6:
                    n = n / n_norm
                    return ((float(n[0]), float(n[1]), float(n[2])), (0.0, 0.0, 1.0))
            except Exception:
                pass

        start_arr = np.asarray(start_xyz, dtype=float)
        goal_arr = np.asarray(goal_xyz, dtype=float)
        n = goal_arr - start_arr
        n_norm = float(np.linalg.norm(n))
        if n_norm <= 1e-6:
            return fallback
        n = n / n_norm
        return ((float(n[0]), float(n[1]), float(n[2])), (0.0, 0.0, 1.0))

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
            return (
                None,
                request.geometric_plan_id,
                f"Unknown geometric_plan_id '{request.geometric_plan_id}'.",
            )
        if not stored.success:
            return (
                None,
                request.geometric_plan_id,
                f"Geometric plan '{request.geometric_plan_id}' is invalid: {stored.message}",
            )
        return stored.path, request.geometric_plan_id, None

    def _get_trajectory_optimizer(self, method: str) -> Tuple[Any, str]:
        method_norm = method.strip().upper()
        if method_norm in ("", "DEFAULT"):
            method_norm = self._default_trajectory_method.strip().upper()

        timber_aliases = {
            "TIMBER_MPC_ILQR": "ACADOS_PATH_FOLLOWING",
            "TIMBER_MPC_ILQR_FAST": "ACADOS_PATH_FOLLOWING_FAST",
            "TIMBER_MPC_ILQR_STABLE": "ACADOS_PATH_FOLLOWING_STABLE",
            "MPC_ILQR": "ACADOS_PATH_FOLLOWING",
            "MPC_ILQR_FAST": "ACADOS_PATH_FOLLOWING_FAST",
            "MPC_ILQR_STABLE": "ACADOS_PATH_FOLLOWING_STABLE",
            "A2B_ILQR": "ACADOS_PATH_FOLLOWING",
            "A2B_ILQR_FAST": "ACADOS_PATH_FOLLOWING_FAST",
            "A2B_ILQR_STABLE": "ACADOS_PATH_FOLLOWING_STABLE",
        }
        profile = timber_aliases.get(method_norm, method_norm)
        if profile not in (
            "ACADOS_PATH_FOLLOWING",
            "ACADOS_PATH_FOLLOWING_FAST",
            "ACADOS_PATH_FOLLOWING_STABLE",
        ):
            profile = "ACADOS_PATH_FOLLOWING"
            if "FAST" in method_norm:
                profile = "ACADOS_PATH_FOLLOWING_FAST"
            elif "STABLE" in method_norm or "COMMISSION" in method_norm:
                profile = "ACADOS_PATH_FOLLOWING_STABLE"

        if profile in self._trajectory_optimizers:
            return self._trajectory_optimizers[profile], method_norm

        if self._analytic_cfg is None:
            raise RuntimeError("Analytic crane configuration not initialized.")

        from motion_planning.trajectory.cartesian_path_following import (
            CartesianPathFollowingConfig,
            CartesianPathFollowingOptimizer,
        )

        horizon = self._traj_default_horizon
        if profile.endswith("FAST"):
            horizon = self._traj_fast_horizon
        elif profile.endswith("STABLE"):
            horizon = int(max(self._traj_default_horizon, self._traj_fast_horizon + 25))

        cfg = CartesianPathFollowingConfig(
            urdf_path=Path(self._analytic_cfg.urdf_path),
            horizon_steps=int(max(20, horizon)),
            optimize_time=False,
            fixed_time_duration_s=10.0,
            fixed_time_duration_candidates=(10.0,),
            code_export_dir=Path(f"/tmp/concrete_block_{profile.lower()}_codegen"),
            solver_json_name=f"{profile.lower()}_ocp.json",
            precompile_on_init=False,
        )

        optimizer = CartesianPathFollowingOptimizer(cfg)
        self._trajectory_optimizers[profile] = optimizer
        return optimizer, method_norm

    def _trajectory_joint_names(self) -> list[str]:
        configured = [jn for jn in self._default_named_joint_names if jn in self._reduced_joint_names]
        return configured or list(self._reduced_joint_names)

    def _project_q_to_trajectory_joints(
        self,
        q_values: np.ndarray,
        joint_names: list[str],
    ) -> np.ndarray:
        q_arr = np.asarray(q_values, dtype=float)
        if q_arr.ndim == 1:
            q_arr = q_arr.reshape(1, -1)
        indices = [self._reduced_joint_names.index(jn) for jn in joint_names]
        return q_arr[:, indices]

    def _reduced_q_from_joint_map(self, q_map: dict[str, float]) -> np.ndarray:
        q_red = np.zeros(len(self._reduced_joint_names), dtype=float)
        for i, name in enumerate(self._reduced_joint_names):
            q_red[i] = float(q_map.get(name, 0.0))
        return q_red

    def _canonical_q_from_joint_map(self, q_map: Mapping[str, float]) -> np.ndarray:
        names = self._canonical_online_joint_names()
        q = np.zeros(len(names), dtype=float)
        for i, name in enumerate(names):
            q[i] = float(q_map.get(name, 0.0))
        return q

    def _complete_dynamic_state_from_actuated(
        self,
        q_map: Mapping[str, float],
        *,
        q_seed: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        if self._steady_state_solver is None or self._analytic_cfg is None:
            return {str(k): float(v) for k, v in q_map.items()}

        act_map = {
            str(name): float(q_map.get(name, 0.0)) for name in self._analytic_cfg.actuated_joints
        }
        completed = self._steady_state_solver.complete_from_actuated(
            act_map,
            q_seed=q_seed,
        )
        if completed.success:
            out = dict(completed.q_dynamic)
        else:
            out = dict(act_map)
        for follower, leader in self._analytic_cfg.tied_joints.items():
            if leader in out:
                out[follower] = float(out[leader])
        return out

    def _joint_goal_linearization_fallback(
        self,
        *,
        q_start: np.ndarray,
        goal_probe,
        method: str,
        geometric_plan_id: str,
        path: NavPath,
        probe_goal_summary: str,
        reason: str,
    ) -> StoredTrajectory | None:
        if goal_probe is None or not goal_probe.success:
            return None
        q_goal_probe = self._reduced_q_from_joint_map(goal_probe.q_dynamic)
        fallback = self._build_fixed_time_interpolation_trajectory(
            q_start=np.asarray(q_start, dtype=float),
            q_goal=np.asarray(q_goal_probe, dtype=float),
            duration_s=self._traj_fixed_duration_s,
            num_points=self._traj_fixed_num_points,
            method=method,
            geometric_plan_id=geometric_plan_id,
            path=path,
        )
        fallback.message = (
            f"{fallback.message} | TOPP-RA path fallback via JOINT_GOAL_LINEARIZATION"
            f"{probe_goal_summary} | fallback_reason={reason}"
        )
        return fallback

    def _get_joint_goal_stage(self):
        if getattr(self, "_joint_goal_stage", None) is not None:
            return self._joint_goal_stage
        try:
            from motion_planning import JointGoalStage

            self._joint_goal_stage = JointGoalStage()
        except Exception as exc:
            self.get_logger().warn(f"Failed to initialize JointGoalStage: {exc}")
            self._joint_goal_stage = None
        return self._joint_goal_stage

    def _get_joint_space_planner(self):
        if getattr(self, "_joint_space_planner", None) is not None:
            return self._joint_space_planner
        try:
            from motion_planning import JointSpaceCartesianPlanner
            from motion_planning.trajectory.planning_limits import load_planning_limits_yaml
            import inspect

            position_limits, _ = load_planning_limits_yaml(
                Path(self._traj_joint_position_limits_file)
            )
            self.get_logger().info(
                "JointSpaceCartesianPlanner import"
                f" | module={inspect.getsourcefile(JointSpaceCartesianPlanner) or '<unknown>'}"
            )
            try:
                plan_src = inspect.getsource(JointSpaceCartesianPlanner.plan)
                self.get_logger().info(
                    "JointSpaceCartesianPlanner plan source"
                    f" | anchor_v1={'ANCHOR_V1' in plan_src}"
                    f" | anchor_count_key={'anchor_count' in plan_src}"
                    f" | projected_waypoints_key={'projected_waypoints' in plan_src}"
                )
            except Exception as exc:
                self.get_logger().warn(f"Failed to inspect JointSpaceCartesianPlanner.plan: {exc}")
            self._joint_space_planner = JointSpaceCartesianPlanner(
                urdf_path=self._analytic_cfg.urdf_path,
                target_frame=self._analytic_cfg.target_frame,
                reduced_joint_names=self._canonical_online_joint_names(),
                joint_position_limits=position_limits,
            )
        except Exception as exc:
            self.get_logger().warn(f"Failed to initialize JointSpaceCartesianPlanner: {exc}")
            self._joint_space_planner = None
        return self._joint_space_planner

    def _parameterize_joint_waypoints_with_toppra(
        self,
        *,
        q_waypoints: np.ndarray,
        joint_names: list[str],
        geometric_plan_id: str,
        method: str,
        base_message: str,
        path: NavPath,
    ) -> StoredTrajectory:
        from motion_planning.trajectory.limits import load_joint_accel_limits_yaml
        from motion_planning.trajectory.planning_limits import load_planning_limits_yaml
        import toppra as ta
        import toppra.algorithm as algo
        import toppra.constraint as constraint

        q_waypoints = np.asarray(q_waypoints, dtype=float)
        if q_waypoints.ndim != 2 or q_waypoints.shape[0] < 2:
            raise ValueError("TOPP-RA requires at least two joint waypoints.")

        position_limits, _ = load_planning_limits_yaml(Path(self._traj_joint_position_limits_file))
        for idx, name in enumerate(joint_names):
            lo, hi = position_limits.get(name, (None, None))
            if lo is not None and np.min(q_waypoints[:, idx]) < float(lo) - 1e-6:
                bad_idx = int(np.argmin(q_waypoints[:, idx]))
                raise ValueError(
                    "TOPP-RA waypoint path violates min position limit for "
                    f"'{name}' (path_min={np.min(q_waypoints[:, idx]):.4f}, limit={float(lo):.4f}, "
                    f"waypoint_index={bad_idx})"
                )
            if hi is not None and np.max(q_waypoints[:, idx]) > float(hi) + 1e-6:
                bad_idx = int(np.argmax(q_waypoints[:, idx]))
                raise ValueError(
                    "TOPP-RA waypoint path violates max position limit for "
                    f"'{name}' (path_max={np.max(q_waypoints[:, idx]):.4f}, limit={float(hi):.4f}, "
                    f"waypoint_index={bad_idx})"
                )

        _, default_acc = load_joint_accel_limits_yaml(Path(self._traj_joint_accel_limits_file))
        accel_yaml, _ = load_joint_accel_limits_yaml(Path(self._traj_joint_accel_limits_file))
        velocity_limits = []
        acceleration_limits = []
        for name in joint_names:
            vlim = abs(float(self._reduced_joint_velocity_limits.get(name, 0.0)))
            if vlim <= 1e-6:
                raise ValueError(f"Missing reduced velocity limit for joint '{name}'.")
            velocity_limits.append(vlim)
            lo_acc, hi_acc = accel_yaml.get(name, default_acc)
            acceleration_limits.append(max(abs(float(lo_acc)), abs(float(hi_acc))))

        ss = np.linspace(0.0, 1.0, q_waypoints.shape[0], dtype=float)
        path_interp = ta.SplineInterpolator(ss, q_waypoints)
        gridpoint_count = max(self._traj_toppra_gridpoints, q_waypoints.shape[0])
        gridpoints = np.linspace(0.0, 1.0, gridpoint_count, dtype=float)
        toppra_constraints = [
            constraint.JointVelocityConstraint(np.asarray(velocity_limits, dtype=float)),
            constraint.JointAccelerationConstraint(np.asarray(acceleration_limits, dtype=float)),
        ]
        parametrizer = algo.TOPPRA(
            toppra_constraints,
            path_interp,
            gridpoints=gridpoints,
            parametrizer="ParametrizeSpline",
        )
        traj = parametrizer.compute_trajectory(0.0, 0.0)
        if traj is None:
            raise ValueError("TOPP-RA failed to compute a time-parameterized trajectory.")

        duration_s = float(getattr(traj, "duration", 0.0))
        if duration_s <= 0.0:
            raise ValueError("TOPP-RA produced a non-positive trajectory duration.")

        sample_count = max(self._traj_fixed_num_points, q_waypoints.shape[0], 20)
        time_s = np.linspace(0.0, duration_s, sample_count, dtype=float)
        q_traj = np.asarray([traj(t) for t in time_s], dtype=float)
        dq_traj = np.asarray([traj(t, 1) for t in time_s], dtype=float)
        if q_traj.shape != dq_traj.shape:
            raise ValueError("TOPP-RA returned inconsistent position/velocity samples.")

        path_len = 0.0
        try:
            pts = np.asarray(
                [[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in path.poses],
                dtype=float,
            )
            if pts.shape[0] > 1:
                path_len = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        except Exception:
            path_len = 0.0

        traj_result = SimpleNamespace(
            success=True,
            message=f"{base_message} ({sample_count} points, {duration_s:.2f}s).",
            state=q_traj,
            time_s=time_s,
            diagnostics={
                "q_trajectory": q_traj,
                "dq_trajectory": dq_traj,
                "reduced_joint_names": joint_names,
                "fallback_mode": "TOPPRA_JOINT_SPACE",
                "duration_s": duration_s,
                "num_points": sample_count,
                "path_length_m": path_len,
            },
        )
        trajectory = self._trajectory_result_to_joint_trajectory(traj_result)
        cartesian_path = self._fk_nav_path_from_joint_trajectory(trajectory)
        trajectory_id = make_trajectory_id()
        return StoredTrajectory(
            trajectory_id=trajectory_id,
            trajectory=trajectory,
            success=True,
            message=traj_result.message,
            method=method,
            geometric_plan_id=geometric_plan_id,
            cartesian_path=cartesian_path,
        )

    def _solve_canonical_q_from_pose(
        self,
        pose: PoseStamped,
    ) -> Tuple[bool, np.ndarray, str]:
        joint_goal_stage = self._get_joint_goal_stage()
        if joint_goal_stage is None:
            return False, np.zeros(0, dtype=float), "joint goal stage is unavailable"

        goal_pose = pose.pose
        try:
            yaw = self._quaternion_to_yaw(goal_pose.orientation)
            result = joint_goal_stage.solve_world_pose(
                goal_world=(
                    float(goal_pose.position.x),
                    float(goal_pose.position.y),
                    float(goal_pose.position.z),
                ),
                target_yaw_rad=float(yaw),
                q_seed=dict(self._ik_seed_map),
            )
        except Exception as exc:
            return False, np.zeros(0, dtype=float), str(exc)

        if not result.success:
            return False, np.zeros(0, dtype=float), result.message

        for name, value in result.q_dynamic.items():
            if name in self._ik_seed_map:
                self._ik_seed_map[name] = float(value)

        return True, self._canonical_q_from_joint_map(result.q_actuated), result.message

    def _build_joint_space_cartesian_trajectory(
        self,
        *,
        path: NavPath,
        geometric_plan_id: str,
        method: str,
        q_start: np.ndarray,
        q_goal: np.ndarray,
    ) -> StoredTrajectory:
        planner = self._get_joint_space_planner()
        if planner is None:
            raise ValueError("joint-space planner unavailable")

        ref_xyz, ref_yaw = self._extract_control_points(path)
        plan_result = planner.plan(
            q_start=np.asarray(q_start, dtype=float),
            q_goal=np.asarray(q_goal, dtype=float),
            reference_xyz=ref_xyz,
            reference_yaw=ref_yaw,
        )
        if not plan_result.success:
            raise ValueError(plan_result.message)

        diagnostics_text = ", ".join(
            [
                f"{key}={float(val):.4f}"
                for key, val in plan_result.diagnostics.items()
                if np.isscalar(val)
            ]
        )
        base_message = "Generated TOPP-RA joint-space Cartesian-cost trajectory"
        if diagnostics_text:
            base_message = f"{base_message} | {plan_result.message} | {diagnostics_text}"
        else:
            base_message = f"{base_message} | {plan_result.message}"

        return self._parameterize_joint_waypoints_with_toppra(
            q_waypoints=plan_result.q_waypoints,
            joint_names=self._canonical_online_joint_names(),
            geometric_plan_id=geometric_plan_id,
            method=method,
            base_message=base_message,
            path=path,
        )

    def _build_toppra_path_following_trajectory(
        self,
        *,
        path: NavPath,
        geometric_plan_id: str,
        method: str,
    ) -> StoredTrajectory:
        from motion_planning.trajectory.limits import load_joint_accel_limits_yaml
        from motion_planning.trajectory.planning_limits import load_planning_limits_yaml
        import toppra as ta
        import toppra.algorithm as algo
        import toppra.constraint as constraint

        if not path.poses or len(path.poses) < 2:
          raise ValueError("TOPP-RA path following requires at least two path poses.")

        goal_probe = None
        probe_goal_summary = ""
        q_waypoints_full: list[np.ndarray] = []
        path_ik_failure = None
        for pose in path.poses:
            ok, q_pose, msg = self._solve_reduced_q_from_pose(pose)
            if not ok:
                path_ik_failure = str(msg)
                break
            q_waypoints_full.append(np.asarray(q_pose, dtype=float))

        joint_names = self._trajectory_joint_names()
        try:
            joint_goal_stage = self._get_joint_goal_stage()
            if joint_goal_stage is not None:
                goal_pose = path.poses[-1].pose
                goal_yaw = self._quaternion_to_yaw(goal_pose.orientation)
                goal_probe = joint_goal_stage.solve_world_pose(
                    goal_world=(
                        float(goal_pose.position.x),
                        float(goal_pose.position.y),
                        float(goal_pose.position.z),
                    ),
                    target_yaw_rad=float(goal_yaw),
                )
                if goal_probe.success:
                    probe_goal_summary = (
                        " | joint_goal_probe: success "
                        f"q4={float(goal_probe.q_dynamic.get('q4_big_telescope', float('nan'))):.4f}"
                    )
                else:
                    probe_goal_summary = f" | joint_goal_probe: failure ({goal_probe.message})"
        except Exception as exc:
            probe_goal_summary = f" | joint_goal_probe: error ({exc})"

        if path_ik_failure is not None:
            if q_waypoints_full:
                fallback = self._joint_goal_linearization_fallback(
                    q_start=np.asarray(q_waypoints_full[0], dtype=float),
                    goal_probe=goal_probe,
                    method=method,
                    geometric_plan_id=geometric_plan_id,
                    path=path,
                    probe_goal_summary=probe_goal_summary,
                    reason=f"ik_failed_along_path:{path_ik_failure}",
                )
                if fallback is not None:
                    return fallback
            else:
                raise ValueError(f"IK failed along path: {path_ik_failure}{probe_goal_summary}")

        q_waypoints = self._project_q_to_trajectory_joints(np.vstack(q_waypoints_full), joint_names)

        position_limits, _ = load_planning_limits_yaml(Path(self._traj_joint_position_limits_file))
        for idx, name in enumerate(joint_names):
            lo, hi = position_limits.get(name, (None, None))
            if lo is not None and np.min(q_waypoints[:, idx]) < float(lo) - 1e-6:
                fallback = self._joint_goal_linearization_fallback(
                    q_start=np.asarray(q_waypoints_full[0], dtype=float),
                    goal_probe=goal_probe,
                    method=method,
                    geometric_plan_id=geometric_plan_id,
                    path=path,
                    probe_goal_summary=probe_goal_summary,
                    reason=(
                        f"min_limit:{name}:"
                        f"{np.min(q_waypoints[:, idx]):.4f}<{float(lo):.4f}"
                    ),
                )
                if fallback is not None:
                    return fallback
                raise ValueError(
                    "TOPP-RA waypoint path violates min position limit for "
                    f"'{name}' (path_min={np.min(q_waypoints[:, idx]):.4f}, limit={float(lo):.4f})"
                    f"{probe_goal_summary}."
                )
            if hi is not None and np.max(q_waypoints[:, idx]) > float(hi) + 1e-6:
                fallback = self._joint_goal_linearization_fallback(
                    q_start=np.asarray(q_waypoints_full[0], dtype=float),
                    goal_probe=goal_probe,
                    method=method,
                    geometric_plan_id=geometric_plan_id,
                    path=path,
                    probe_goal_summary=probe_goal_summary,
                    reason=(
                        f"max_limit:{name}:"
                        f"{np.max(q_waypoints[:, idx]):.4f}>{float(hi):.4f}"
                    ),
                )
                if fallback is not None:
                    return fallback
                raise ValueError(
                    "TOPP-RA waypoint path violates max position limit for "
                    f"'{name}' (path_max={np.max(q_waypoints[:, idx]):.4f}, limit={float(hi):.4f})"
                    f"{probe_goal_summary}."
                )

        return self._parameterize_joint_waypoints_with_toppra(
            q_waypoints=q_waypoints,
            joint_names=joint_names,
            geometric_plan_id=geometric_plan_id,
            method=method,
            base_message="Generated TOPP-RA path-following joint trajectory",
            path=path,
        )

    def _build_fixed_time_interpolation_trajectory(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        duration_s: float,
        num_points: int,
        method: str,
        geometric_plan_id: str,
        path: NavPath,
    ) -> StoredTrajectory:
        duration_s = max(0.1, float(duration_s))
        num_points = max(2, int(num_points))

        fallback_mode = "JOINT_LINEAR_INTERPOLATION"
        ik_waypoints: list[np.ndarray] = []
        if path.poses:
            for pose in path.poses:
                ok, q_pose, _ = self._solve_reduced_q_from_pose(pose)
                if not ok:
                    ik_waypoints = []
                    break
                ik_waypoints.append(np.asarray(q_pose, dtype=float).reshape(1, -1))

        if len(ik_waypoints) >= 2:
            q_waypoints = np.vstack(ik_waypoints)
            waypoint_param = np.linspace(0.0, 1.0, q_waypoints.shape[0], dtype=float)
            sample_param = np.linspace(0.0, 1.0, num_points, dtype=float)
            q_traj = np.column_stack(
                [
                    np.interp(sample_param, waypoint_param, q_waypoints[:, i])
                    for i in range(q_waypoints.shape[1])
                ]
            )
            fallback_mode = "PATH_IK_INTERPOLATION"
        else:
            alpha = np.linspace(0.0, 1.0, num_points, dtype=float).reshape(-1, 1)
            q_traj = (1.0 - alpha) * q_start.reshape(1, -1) + alpha * q_goal.reshape(1, -1)

        time_s = np.linspace(0.0, duration_s, num_points, dtype=float)
        dq_traj = np.zeros_like(q_traj)
        if num_points >= 2:
            dt = float(time_s[1] - time_s[0]) if num_points > 1 else duration_s
            if dt <= 0.0:
                dt = duration_s / max(1, num_points - 1)
            dq_traj[1:-1, :] = (q_traj[2:, :] - q_traj[:-2, :]) / (2.0 * dt)
            dq_traj[0, :] = (q_traj[1, :] - q_traj[0, :]) / dt
            dq_traj[-1, :] = (q_traj[-1, :] - q_traj[-2, :]) / dt

        q_delta = q_goal - q_start
        path_len = 0.0
        try:
            if path.poses:
                pts = np.asarray(
                    [
                        [p.pose.position.x, p.pose.position.y, p.pose.position.z]
                        for p in path.poses
                    ],
                    dtype=float,
                )
                if pts.shape[0] > 1:
                    path_len = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        except Exception:
            path_len = 0.0

        traj_result = SimpleNamespace(
            success=True,
            message=(
                "Generated fixed-time interpolated joint trajectory "
                f"({num_points} points, {duration_s:.2f}s) without acados "
                f"using {fallback_mode}."
            ),
            state=q_traj,
            time_s=time_s,
            diagnostics={
                "q_trajectory": q_traj,
                "dq_trajectory": dq_traj,
                "reduced_joint_names": list(self._reduced_joint_names),
                "fallback_mode": fallback_mode,
                "duration_s": duration_s,
                "num_points": num_points,
                "joint_delta_norm": float(np.linalg.norm(q_delta)),
                "path_length_m": path_len,
            },
        )

        trajectory = self._trajectory_result_to_joint_trajectory(traj_result)
        cartesian_path = self._fk_nav_path_from_joint_trajectory(trajectory)
        trajectory_id = make_trajectory_id()
        return StoredTrajectory(
            trajectory_id=trajectory_id,
            trajectory=trajectory,
            success=True,
            message=traj_result.message,
            method=method,
            geometric_plan_id=geometric_plan_id,
            cartesian_path=cartesian_path,
        )

    def _trajectory_result_to_joint_trajectory(
        self, traj_result: Any
    ) -> JointTrajectory:
        traj = JointTrajectory()

        diagnostics = dict(getattr(traj_result, "diagnostics", {}) or {})
        q_traj = np.asarray(
            diagnostics.get("q_trajectory", np.asarray(traj_result.state, dtype=float)),
            dtype=float,
        )
        dq_traj = np.asarray(
            diagnostics.get("dq_trajectory", np.zeros_like(q_traj)),
            dtype=float,
        )
        time_s = np.asarray(getattr(traj_result, "time_s", []), dtype=float).reshape(-1)

        if q_traj.ndim != 2:
            raise ValueError(
                "Trajectory optimizer returned invalid q_trajectory shape."
            )
        if dq_traj.shape != q_traj.shape:
            dq_traj = np.zeros_like(q_traj)

        if time_s.size != q_traj.shape[0]:
            time_s = np.linspace(
                0.0, max(0.1, 0.1 * (q_traj.shape[0] - 1)), q_traj.shape[0]
            )

        names = list(diagnostics.get("reduced_joint_names", self._reduced_joint_names))
        if not names:
            names = [f"joint_{i + 1}" for i in range(q_traj.shape[1])]

        traj.joint_names = names
        for k in range(q_traj.shape[0]):
            p = JointTrajectoryPoint()
            p.positions = [float(v) for v in q_traj[k, :].tolist()]
            p.velocities = [float(v) for v in dq_traj[k, :].tolist()]
            p.time_from_start = self._duration_from_seconds(float(time_s[k]))
            traj.points.append(p)

        return traj

    def _solve_reduced_q_from_pose(
        self, pose: PoseStamped
    ) -> Tuple[bool, np.ndarray, str]:
        if self._steady_state_solver is None or self._analytic_cfg is None:
            return False, np.zeros(0, dtype=float), "steady-state solver is unavailable"

        p = np.asarray(
            [
                float(pose.pose.position.x),
                float(pose.pose.position.y),
                float(pose.pose.position.z),
            ],
            dtype=float,
        )

        frame_id = (pose.header.frame_id or "").strip()
        if frame_id == self._analytic_cfg.base_frame:
            p_base = p
            R_base_pose = self._quat_to_rot_matrix(pose.pose.orientation)
        else:
            if frame_id not in ("", "world", "map"):
                self.get_logger().warn(
                    f"Unsupported pose frame '{frame_id}'. Assuming world frame for IK solve."
                )
            p_h = np.concatenate([p, [1.0]], dtype=float)
            p_base = (self._T_base_world @ p_h)[:3]
            R_world_pose = self._quat_to_rot_matrix(pose.pose.orientation)
            R_base_pose = self._T_base_world[:3, :3] @ R_world_pose

        yaw_base = float(math.atan2(R_base_pose[1, 0], R_base_pose[0, 0]))

        seed = dict(self._ik_seed_map)
        ss_result = self._steady_state_solver.compute(
            target_pos=p_base,
            target_yaw=yaw_base,
            q_seed=seed,
        )
        if not ss_result.success:
            return False, np.zeros(0, dtype=float), ss_result.message

        q_map = dict(ss_result.q_dynamic)
        for follower, leader in self._analytic_cfg.tied_joints.items():
            if leader in q_map:
                q_map[follower] = float(q_map[leader])

        q_red = np.zeros(len(self._reduced_joint_names), dtype=float)
        for i, name in enumerate(self._reduced_joint_names):
            q_red[i] = float(q_map.get(name, 0.0))

        for name in self._ik_seed_map.keys():
            if name in ss_result.q_dynamic:
                self._ik_seed_map[name] = float(ss_result.q_dynamic[name])

        return True, q_red, ss_result.message

    def _extract_control_points(self, path: NavPath) -> Tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(
            [
                [
                    float(p.pose.position.x),
                    float(p.pose.position.y),
                    float(p.pose.position.z),
                ]
                for p in path.poses
            ],
            dtype=float,
        )
        yaw = np.asarray(
            [self._quaternion_to_yaw(p.pose.orientation) for p in path.poses],
            dtype=float,
        )

        if pts.shape[0] < self._traj_ctrl_pts_min:
            n = max(self._traj_ctrl_pts_min, 2)
            idx = np.linspace(0, max(0, pts.shape[0] - 1), n)
            if pts.shape[0] == 1:
                pts = np.repeat(pts, n, axis=0)
                yaw = np.repeat(yaw, n, axis=0)
            else:
                pts = np.vstack([self._interp_vec(pts, s) for s in idx])
                yaw = np.asarray(
                    [self._interp_scalar(yaw, s) for s in idx], dtype=float
                )
            return pts, yaw

        n_keep = min(
            max(self._traj_ctrl_pts_min, pts.shape[0]), self._traj_ctrl_pts_max
        )
        sample_idx = np.linspace(0, pts.shape[0] - 1, n_keep)
        ctrl_pts = np.vstack([self._interp_vec(pts, s) for s in sample_idx])
        ctrl_yaw = np.asarray(
            [self._interp_scalar(yaw, s) for s in sample_idx], dtype=float
        )
        return ctrl_pts, ctrl_yaw

    def _initialize_planning_runtime(self) -> None:
        try:
            import pinocchio as pin
            from motion_planning.optimized_params import (
                load_optimized_planner_params,
            )
            from motion_planning.mechanics import CraneKinematics
            from motion_planning.mechanics import (
                CraneSteadyState,
                ModelDescription,
                create_crane_config,
            )
        except Exception as exc:
            self._planning_runtime_ready = False
            self._planning_runtime_reason = f"import failure: {exc}"
            return

        try:
            self._analytic_cfg = create_crane_config()
            urdf_resolved = self._resolve_runtime_urdf_path(
                self._analytic_cfg.urdf_path
            )
            self._analytic_cfg.urdf_path = str(urdf_resolved)
            desc = ModelDescription(self._analytic_cfg)
            self._steady_state_solver = CraneSteadyState(desc, self._analytic_cfg)

            kin = CraneKinematics(self._analytic_cfg.urdf_path)
            q_neutral = pin.neutral(kin.model)
            fk_base = kin.forward_kinematics(
                q_neutral,
                base_frame="world",
                end_frame=self._analytic_cfg.base_frame,
            )
            self._T_world_base = np.asarray(
                fk_base["base_to_end"]["homogeneous"], dtype=float
            )
            self._T_base_world = np.linalg.inv(self._T_world_base)

            full_model = pin.buildModelFromUrdf(str(self._analytic_cfg.urdf_path))
            full_name_to_jid = {
                str(full_model.names[jid]): int(jid)
                for jid in range(1, full_model.njoints)
            }
            lock_ids = [
                full_name_to_jid[jn]
                for jn in self._analytic_cfg.locked_joints
                if jn in full_name_to_jid
            ]
            reduced_model = pin.buildReducedModel(
                full_model, lock_ids, pin.neutral(full_model)
            )
            self._reduced_joint_names = [
                str(reduced_model.names[jid]) for jid in range(1, reduced_model.njoints)
            ]
            self._reduced_joint_velocity_limits = {
                str(reduced_model.names[jid]): float(reduced_model.velocityLimit[int(reduced_model.joints[jid].idx_v)])
                for jid in range(1, reduced_model.njoints)
                if int(reduced_model.joints[jid].nv) == 1
            }

            self._ik_seed_map = {jn: 0.0 for jn in self._analytic_cfg.dynamic_joints}
            self._planner_scene = None

            if self._optimized_params_file:
                params_path = Path(self._optimized_params_file).expanduser()
                if not params_path.is_absolute():
                    params_path = Path.cwd() / params_path
                if params_path.exists():
                    self._optimized_planner_params = load_optimized_planner_params(
                        params_path
                    )
                else:
                    self.get_logger().warn(
                        f"geometric_optimized_params_file '{params_path}' does not exist; using defaults."
                    )

            self._planning_runtime_ready = True
            self._planning_runtime_reason = "initialized"
        except Exception as exc:
            self._planning_runtime_ready = False
            self._planning_runtime_reason = f"initialization failure: {exc}"

    def _ensure_planner_scene(self) -> Tuple[bool, str]:
        if self._planner_scene is not None:
            return True, "scene ready"
        try:
            from motion_planning import Scene

            self._planner_scene = Scene()
            return True, "scene initialized"
        except Exception as exc:
            return False, str(exc)

    def _ensure_motion_planning_module_path(self) -> None:
        try:
            import motion_planning  # noqa: F401

            return
        except Exception:
            pass

        candidates: List[Path] = []
        try:
            from ament_index_python.packages import get_package_share_directory

            candidates.append(
                Path(get_package_share_directory("concrete_block_motion_planning"))
            )
        except Exception:
            pass

        this_file = Path(__file__).resolve()
        for parent in [this_file.parent] + list(this_file.parents):
            candidates.append(parent)
            candidates.append(
                parent
                / "src"
                / "concrete_block_stack"
                / "concrete_block_motion_planning"
            )

        for cand in candidates:
            if not cand.exists():
                continue
            if (cand / "motion_planning").exists():
                s = str(cand)
                if s not in sys.path:
                    sys.path.insert(0, s)

        try:
            import motion_planning  # noqa: F401
        except Exception as exc:
            self.get_logger().warn(
                f"Failed to expose motion_planning module path: {exc}"
            )

    @staticmethod
    def _resolve_existing_urdf_path(initial_path: str) -> Path:
        p = Path(initial_path).expanduser()
        if p.exists():
            return p.resolve()

        env_raw = os.environ.get("CRANE_URDF_PATH", "").strip()
        env_path = Path(env_raw).expanduser() if env_raw else None
        if env_path is not None and env_path.exists():
            return env_path.resolve()

        this_file = Path(__file__).resolve()
        for parent in [Path.cwd(), this_file.parent, *this_file.parents]:
            cand = parent / "crane_urdf" / "crane.urdf"
            if cand.exists():
                return cand.resolve()

        raise FileNotFoundError(
            f"Cannot locate crane URDF. Tried configured path '{initial_path}' "
            "and local crane_urdf/crane.urdf candidates."
        )

    def _resolve_runtime_urdf_path(self, initial_path: str) -> Path:
        robot_description_xml = str(
            getattr(self, "_robot_description_xml", "") or ""
        ).strip()
        if robot_description_xml:
            urdf_path = (
                Path(tempfile.gettempdir())
                / "concrete_block_motion_planning_robot_description.urdf"
            )
            urdf_path.write_text(robot_description_xml, encoding="utf-8")
            return urdf_path
        return self._resolve_existing_urdf_path(initial_path)

    def _load_named_configurations_from_file(self, path_value: str) -> None:
        if not path_value:
            return

        config_path = Path(path_value).expanduser()
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path

        if not config_path.exists():
            self.get_logger().warn(
                f"named_configurations_file '{config_path}' does not exist; "
                "named configuration execution disabled."
            )
            return

        try:
            with config_path.open("r", encoding="utf-8") as fh:
                payload = yaml.safe_load(fh) or {}
        except Exception as exc:  # pragma: no cover - defensive runtime handling
            self.get_logger().error(
                f"Failed to parse named configurations file '{config_path}': {exc}"
            )
            return

        if not isinstance(payload, dict):
            self.get_logger().error(
                f"Invalid named configurations payload in '{config_path}' (expected mapping)."
            )
            return

        default_joint_names = payload.get(
            "joint_names", self._default_named_joint_names
        )
        if not isinstance(default_joint_names, list):
            default_joint_names = self._default_named_joint_names
        default_joint_names = [str(v) for v in default_joint_names]

        configurations = payload.get("configurations", {})
        if not isinstance(configurations, dict):
            self.get_logger().error(
                f"Invalid 'configurations' section in '{config_path}' (expected mapping)."
            )
            return

        loaded = 0
        seen_names: set[str] = set()
        for raw_name, cfg in configurations.items():
            name = str(raw_name).strip()
            if not name:
                self.get_logger().warn("Skipping unnamed configuration entry.")
                continue
            name_key = name.lower()
            if name_key in seen_names:
                self.get_logger().warn(
                    f"Skipping '{name}': duplicate configuration name (case-insensitive)."
                )
                continue
            seen_names.add(name_key)

            if not isinstance(cfg, dict):
                self.get_logger().warn(f"Skipping '{name}': expected mapping value.")
                continue

            positions_raw = cfg.get("positions")
            if not isinstance(positions_raw, list) or not positions_raw:
                self.get_logger().warn(
                    f"Skipping '{name}': missing or empty positions list."
                )
                continue
            try:
                positions = [float(v) for v in positions_raw]
            except Exception:
                self.get_logger().warn(f"Skipping '{name}': positions must be numeric.")
                continue
            if not all(math.isfinite(v) for v in positions):
                self.get_logger().warn(
                    f"Skipping '{name}': positions must be finite numeric values."
                )
                continue

            joint_names_raw = cfg.get("joint_names", default_joint_names)
            if not isinstance(joint_names_raw, list) or not joint_names_raw:
                self.get_logger().warn(f"Skipping '{name}': missing joint_names.")
                continue
            joint_names = [str(v).strip() for v in joint_names_raw]
            if any(not jn for jn in joint_names):
                self.get_logger().warn(
                    f"Skipping '{name}': joint_names must not contain empty entries."
                )
                continue
            if len(set(joint_names)) != len(joint_names):
                self.get_logger().warn(
                    f"Skipping '{name}': joint_names must be unique."
                )
                continue

            if len(joint_names) != len(positions):
                self.get_logger().warn(
                    f"Skipping '{name}': joint_names({len(joint_names)}) "
                    f"!= positions({len(positions)})."
                )
                continue

            try:
                duration_s = float(
                    cfg.get("duration_s", self._named_cfg_default_duration_s)
                )
            except Exception:
                self.get_logger().warn(
                    f"Skipping '{name}': duration_s must be numeric."
                )
                continue
            if not math.isfinite(duration_s) or duration_s <= 0.0:
                self.get_logger().warn(
                    f"Skipping '{name}': duration_s must be > 0 and finite."
                )
                continue

            traj = JointTrajectory()
            traj.joint_names = joint_names
            point = JointTrajectoryPoint()
            point.positions = positions
            point.velocities = [0.0 for _ in positions]
            point.time_from_start = self._duration_from_seconds(duration_s)
            traj.points = [point]

            self._named_configurations[name] = traj
            loaded += 1

        self.get_logger().info(
            f"Loaded {loaded} named configurations from '{config_path}'."
        )

    def _load_wall_plans_from_file(self, path_value: str) -> None:
        if not path_value:
            return

        config_path = Path(path_value).expanduser()
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path

        if not config_path.exists():
            self.get_logger().warn(
                f"wall_plan_file '{config_path}' does not exist; wall plan task service disabled."
            )
            return

        try:
            with config_path.open("r", encoding="utf-8") as fh:
                payload = yaml.safe_load(fh) or {}
        except Exception as exc:  # pragma: no cover - defensive runtime handling
            self.get_logger().error(
                f"Failed to parse wall plan file '{config_path}': {exc}"
            )
            return

        if not isinstance(payload, dict):
            self.get_logger().error(
                f"Invalid wall plan payload in '{config_path}' (expected mapping)."
            )
            return

        plans = payload.get("wall_plans", {})
        if not isinstance(plans, dict):
            self.get_logger().error(
                f"Invalid wall plan file '{config_path}': 'wall_plans' must be a mapping."
            )
            return

        loaded = 0
        for raw_plan_name, plan_cfg in plans.items():
            plan_name = str(raw_plan_name).strip()
            if not plan_name:
                self.get_logger().warn("Skipping unnamed wall plan.")
                continue
            if not isinstance(plan_cfg, dict):
                self.get_logger().warn(
                    f"Skipping wall plan '{plan_name}': expected mapping value."
                )
                continue
            sequence = plan_cfg.get("sequence", [])
            if not isinstance(sequence, list) or not sequence:
                self.get_logger().warn(
                    f"Skipping wall plan '{plan_name}': missing or empty sequence."
                )
                continue

            resolved_positions: Dict[str, Tuple[float, float, float]] = {}
            tasks: List[WallPlanTask] = []
            valid = True
            invalid_reason = "invalid sequence entry"
            for idx, item in enumerate(sequence):
                if not isinstance(item, dict):
                    valid = False
                    invalid_reason = f"item {idx + 1} is not a mapping"
                    break

                block_id = str(item.get("id", "")).strip()
                if not block_id:
                    valid = False
                    invalid_reason = f"item {idx + 1} has empty id"
                    break
                if block_id in resolved_positions:
                    valid = False
                    invalid_reason = f"item {idx + 1} duplicates block id '{block_id}'"
                    break

                if "absolute_position" in item:
                    pos = self._vec3_or_none(item.get("absolute_position"))
                    if pos is None:
                        valid = False
                        invalid_reason = f"item {idx + 1} has invalid absolute_position"
                        break
                    reference_block_id = ""
                else:
                    reference_block_id = str(item.get("relative_to", "")).strip()
                    if (
                        not reference_block_id
                        or reference_block_id not in resolved_positions
                    ):
                        valid = False
                        invalid_reason = f"item {idx + 1} references unknown block '{reference_block_id}'"
                        break
                    offset = self._vec3_or_none(item.get("offset", [0.0, 0.0, 0.0]))
                    if offset is None:
                        valid = False
                        invalid_reason = f"item {idx + 1} has invalid offset"
                        break
                    ref = resolved_positions[reference_block_id]
                    pos = (ref[0] + offset[0], ref[1] + offset[1], ref[2] + offset[2])
                if not all(math.isfinite(v) for v in pos):
                    valid = False
                    invalid_reason = f"item {idx + 1} resolved to non-finite position"
                    break

                resolved_positions[block_id] = pos

                pickup_pose = PoseStamped()
                pickup_pose.header.frame_id = self._wall_plan_frame_id
                pickup_pose.pose.orientation.w = 1.0

                target_pose = PoseStamped()
                target_pose.header.frame_id = self._wall_plan_frame_id
                target_pose.pose.position.x = float(pos[0])
                target_pose.pose.position.y = float(pos[1])
                target_pose.pose.position.z = float(pos[2])
                target_pose.pose.orientation.w = 1.0

                reference_pose = PoseStamped()
                reference_pose.header.frame_id = self._wall_plan_frame_id
                if reference_block_id and reference_block_id in resolved_positions:
                    ref_pos = resolved_positions[reference_block_id]
                    reference_pose.pose.position.x = float(ref_pos[0])
                    reference_pose.pose.position.y = float(ref_pos[1])
                    reference_pose.pose.position.z = float(ref_pos[2])
                reference_pose.pose.orientation.w = 1.0

                task = WallPlanTask(
                    task_id=f"{plan_name}_{idx + 1:02d}_{block_id}",
                    target_block_id=block_id,
                    reference_block_id=reference_block_id,
                    pickup_pose=pickup_pose,
                    target_pose=target_pose,
                    reference_pose=reference_pose,
                )
                tasks.append(task)

            if not valid or not tasks:
                self.get_logger().warn(
                    f"Skipping invalid wall plan '{plan_name}': {invalid_reason}."
                )
                continue

            name = str(plan_name).strip().lower()
            self._wall_plans[name] = tasks
            self._wall_plan_progress[name] = 0
            loaded += 1

        self.get_logger().info(f"Loaded {loaded} wall plans from '{config_path}'.")

    @staticmethod
    def _interp_vec(values: np.ndarray, index: float) -> np.ndarray:
        idx0 = int(math.floor(index))
        idx1 = min(idx0 + 1, values.shape[0] - 1)
        a = float(index - idx0)
        return (1.0 - a) * values[idx0] + a * values[idx1]

    @staticmethod
    def _interp_scalar(values: np.ndarray, index: float) -> float:
        idx0 = int(math.floor(index))
        idx1 = min(idx0 + 1, values.shape[0] - 1)
        a = float(index - idx0)
        return float((1.0 - a) * values[idx0] + a * values[idx1])

    @staticmethod
    def _pose_xyz(pose: PoseStamped) -> Tuple[float, float, float]:
        return (
            float(pose.pose.position.x),
            float(pose.pose.position.y),
            float(pose.pose.position.z),
        )

    @staticmethod
    def _quat_to_rot_matrix(q: Quaternion) -> np.ndarray:
        x = float(q.x)
        y = float(q.y)
        z = float(q.z)
        w = float(q.w)
        n = math.sqrt(x * x + y * y + z * z + w * w)
        if n <= 1e-9:
            return np.eye(3, dtype=float)
        x /= n
        y /= n
        z /= n
        w /= n

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=float,
        )

    @staticmethod
    def _quaternion_to_yaw(q: Quaternion) -> float:
        x = float(q.x)
        y = float(q.y)
        z = float(q.z)
        w = float(q.w)
        n = math.sqrt(x * x + y * y + z * z + w * w)
        if n <= 1e-9:
            return 0.0
        x /= n
        y /= n
        z /= n
        w /= n
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _yaw_to_quaternion(yaw: float) -> Quaternion:
        q = Quaternion()
        h = 0.5 * float(yaw)
        q.z = math.sin(h)
        q.w = math.cos(h)
        return q

    @staticmethod
    def _vec3_or_none(values: Any) -> Optional[Tuple[float, float, float]]:
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            return None
        return (float(values[0]), float(values[1]), float(values[2]))

    @staticmethod
    def _duration_from_seconds(seconds: float) -> Duration:
        s = max(0.0, float(seconds))
        sec = int(math.floor(s))
        nanosec = int((s - sec) * 1e9)
        return Duration(sec=sec, nanosec=nanosec)

    @staticmethod
    def _check_geometric_runtime() -> Tuple[bool, str]:
        missing = []
        for module in ("fcl", "scipy"):
            try:
                __import__(module)
            except Exception:
                missing.append(module)
        if missing:
            return False, f"missing modules: {', '.join(missing)}"
        return True, "geometric planner runtime modules are available"

    @staticmethod
    def _check_trajectory_runtime() -> Tuple[bool, str]:
        missing: List[str] = []

        try:
            import pinocchio  # noqa: F401
        except Exception:
            missing.append("pinocchio")

        for module in ("toppra", "scipy"):
            try:
                __import__(module)
            except Exception:
                missing.append(module)

        if missing:
            return False, f"missing modules: {', '.join(missing)}"
        return True, "trajectory runtime modules are available (pinocchio, scipy, toppra)"

    def _publish_backend_status(self, state: str, detail: str) -> None:
        msg = String()
        msg.data = f"trajectory_backend={state}; detail={detail}"
        self._status_pub.publish(msg)
