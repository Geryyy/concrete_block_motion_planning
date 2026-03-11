from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Path as NavPath
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .types import StoredGeometricPlan, WallPlanTask


class RuntimeHelpersMixin:
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
                    "Geometric backend unavailable: "
                    f"{self._geometric_runtime_reason}"
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
            from motion_planning.core.types import Scenario
            from motion_planning.io.optimized_params import canonical_method_name
            from motion_planning.pipeline.geometric_stage import run_geometric_planning
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
            cfg["initial_vias"] = np.vstack([(1.0 - a) * start_arr + a * goal_arr for a in t])

        scenario_scene = self._planner_scene
        world_model_note = ""
        world_blocks = list(planning_context.get("world_model_blocks", []))
        if world_blocks:
            built_scene, world_model_note = self._scene_from_world_blocks(world_blocks)
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
        nav_path.header.frame_id = goal_pose.header.frame_id or start_pose.header.frame_id or "world"

        if geo_result.path is not None:
            sample_count = max(2, self._n_points)
            xyz = np.asarray(geo_result.path.sample(sample_count), dtype=float).reshape(-1, 3)
            yaw = np.asarray(geo_result.path.sample_yaw(sample_count), dtype=float).reshape(-1)
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

    def _scene_from_world_blocks(
        self,
        world_blocks: List[Dict[str, Any]],
    ) -> Tuple[Optional[Any], str]:
        try:
            from motion_planning import Scene
        except Exception as exc:
            return None, f"cannot import Scene for world model integration: {exc}"

        scene = Scene()
        added = 0
        for item in world_blocks:
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
                    size=self._moving_block_size,
                    position=position,
                    quat=quat,
                    object_id=oid,
                )
                added += 1
            except Exception:
                continue
        if added <= 0:
            return None, "world model returned no valid blocks"
        return scene, f"world_model_blocks_used={added}"

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
            return None, request.geometric_plan_id, f"Unknown geometric_plan_id '{request.geometric_plan_id}'."
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

        profile = "ACADOS_PATH_FOLLOWING"
        if "FAST" in method_norm:
            profile = "ACADOS_PATH_FOLLOWING_FAST"
        elif "STABLE" in method_norm or "COMMISSION" in method_norm:
            profile = "ACADOS_PATH_FOLLOWING_STABLE"

        if profile in self._trajectory_optimizers:
            return self._trajectory_optimizers[profile], profile

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
            code_export_dir=Path(f"/tmp/concrete_block_{profile.lower()}_codegen"),
            solver_json_name=f"{profile.lower()}_ocp.json",
            precompile_on_init=False,
        )

        optimizer = CartesianPathFollowingOptimizer(cfg)
        self._trajectory_optimizers[profile] = optimizer
        return optimizer, profile

    def _trajectory_result_to_joint_trajectory(self, traj_result: Any) -> JointTrajectory:
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
            raise ValueError("Trajectory optimizer returned invalid q_trajectory shape.")
        if dq_traj.shape != q_traj.shape:
            dq_traj = np.zeros_like(q_traj)

        if time_s.size != q_traj.shape[0]:
            time_s = np.linspace(0.0, max(0.1, 0.1 * (q_traj.shape[0] - 1)), q_traj.shape[0])

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

    def _solve_reduced_q_from_pose(self, pose: PoseStamped) -> Tuple[bool, np.ndarray, str]:
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
                yaw = np.asarray([self._interp_scalar(yaw, s) for s in idx], dtype=float)
            return pts, yaw

        n_keep = min(max(self._traj_ctrl_pts_min, pts.shape[0]), self._traj_ctrl_pts_max)
        sample_idx = np.linspace(0, pts.shape[0] - 1, n_keep)
        ctrl_pts = np.vstack([self._interp_vec(pts, s) for s in sample_idx])
        ctrl_yaw = np.asarray([self._interp_scalar(yaw, s) for s in sample_idx], dtype=float)
        return ctrl_pts, ctrl_yaw

    def _initialize_planning_runtime(self) -> None:
        try:
            import pinocchio as pin
            from motion_planning.io.optimized_params import load_optimized_planner_params
            from motion_planning.kinematics import CraneKinematics
            from motion_planning.mechanics.analytic import (
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
            urdf_resolved = self._resolve_existing_urdf_path(self._analytic_cfg.urdf_path)
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
            self._T_world_base = np.asarray(fk_base["base_to_end"]["homogeneous"], dtype=float)
            self._T_base_world = np.linalg.inv(self._T_world_base)

            full_model = pin.buildModelFromUrdf(str(self._analytic_cfg.urdf_path))
            full_name_to_jid = {
                str(full_model.names[jid]): int(jid) for jid in range(1, full_model.njoints)
            }
            lock_ids = [
                full_name_to_jid[jn]
                for jn in self._analytic_cfg.locked_joints
                if jn in full_name_to_jid
            ]
            reduced_model = pin.buildReducedModel(full_model, lock_ids, pin.neutral(full_model))
            self._reduced_joint_names = [
                str(reduced_model.names[jid]) for jid in range(1, reduced_model.njoints)
            ]

            self._ik_seed_map = {jn: 0.0 for jn in self._analytic_cfg.dynamic_joints}
            self._planner_scene = None

            if self._optimized_params_file:
                params_path = Path(self._optimized_params_file).expanduser()
                if not params_path.is_absolute():
                    params_path = Path.cwd() / params_path
                if params_path.exists():
                    self._optimized_planner_params = load_optimized_planner_params(params_path)
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

            candidates.append(Path(get_package_share_directory("concrete_block_motion_planning")))
        except Exception:
            pass

        this_file = Path(__file__).resolve()
        for parent in [this_file.parent] + list(this_file.parents):
            candidates.append(parent)
            candidates.append(parent / "src" / "concrete_block_stack" / "concrete_block_motion_planning")

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
            self.get_logger().warn(f"Failed to expose motion_planning module path: {exc}")

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

        default_joint_names = payload.get("joint_names", self._default_named_joint_names)
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
                self.get_logger().warn(f"Skipping '{name}': missing or empty positions list.")
                continue
            try:
                positions = [float(v) for v in positions_raw]
            except Exception:
                self.get_logger().warn(f"Skipping '{name}': positions must be numeric.")
                continue
            if not all(math.isfinite(v) for v in positions):
                self.get_logger().warn(f"Skipping '{name}': positions must be finite numeric values.")
                continue

            joint_names_raw = cfg.get("joint_names", default_joint_names)
            if not isinstance(joint_names_raw, list) or not joint_names_raw:
                self.get_logger().warn(f"Skipping '{name}': missing joint_names.")
                continue
            joint_names = [str(v).strip() for v in joint_names_raw]
            if any(not jn for jn in joint_names):
                self.get_logger().warn(f"Skipping '{name}': joint_names must not contain empty entries.")
                continue
            if len(set(joint_names)) != len(joint_names):
                self.get_logger().warn(f"Skipping '{name}': joint_names must be unique.")
                continue

            if len(joint_names) != len(positions):
                self.get_logger().warn(
                    f"Skipping '{name}': joint_names({len(joint_names)}) "
                    f"!= positions({len(positions)})."
                )
                continue

            try:
                duration_s = float(cfg.get("duration_s", self._named_cfg_default_duration_s))
            except Exception:
                self.get_logger().warn(f"Skipping '{name}': duration_s must be numeric.")
                continue
            if not math.isfinite(duration_s) or duration_s <= 0.0:
                self.get_logger().warn(f"Skipping '{name}': duration_s must be > 0 and finite.")
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
            self.get_logger().error(f"Failed to parse wall plan file '{config_path}': {exc}")
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
                self.get_logger().warn(f"Skipping wall plan '{plan_name}': expected mapping value.")
                continue
            sequence = plan_cfg.get("sequence", [])
            if not isinstance(sequence, list) or not sequence:
                self.get_logger().warn(f"Skipping wall plan '{plan_name}': missing or empty sequence.")
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
                    if not reference_block_id or reference_block_id not in resolved_positions:
                        valid = False
                        invalid_reason = (
                            f"item {idx + 1} references unknown block '{reference_block_id}'"
                        )
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

        self.get_logger().info(
            f"Loaded {loaded} wall plans from '{config_path}'."
        )

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

        for module in ("acados_template", "casadi"):
            try:
                __import__(module)
            except Exception:
                missing.append(module)

        try:
            import pinocchio  # noqa: F401
        except Exception:
            missing.append("pinocchio")
        else:
            try:
                from pinocchio import casadi as _pinocchio_casadi  # noqa: F401
            except Exception:
                missing.append("pinocchio.casadi")

        if missing:
            return False, f"missing modules: {', '.join(missing)}"
        acados_source = os.environ.get("ACADOS_SOURCE_DIR", "").strip()
        if not acados_source:
            return (
                True,
                "trajectory runtime modules are available "
                "(ACADOS_SOURCE_DIR is not set; source acados_interface_setup.sh if codegen fails)",
            )
        acados_path = Path(acados_source).expanduser()
        if not acados_path.exists():
            return (
                True,
                f"trajectory modules ok, but ACADOS_SOURCE_DIR '{acados_path}' does not exist",
            )
        return True, f"trajectory runtime modules are available (ACADOS_SOURCE_DIR={acados_path})"

    def _publish_backend_status(self, state: str, detail: str) -> None:
        msg = String()
        msg.data = f"trajectory_backend={state}; detail={detail}"
        self._status_pub.publish(msg)
