from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from motion_planning.spline import BSplinePath
from motion_planning.types import PlannerResult, Scenario, TrajectoryRequest, TrajectoryResult
from motion_planning.world_model import WorldModel
from motion_planning.geometry.scene import Scene as GeometryScene

@dataclass
class Trajectory:
    time_s: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    joint_names: list[str]
    actuated_joint_names: list[str] | None = None

    def __post_init__(self) -> None:
        self.time_s = np.asarray(self.time_s, dtype=float).reshape(-1)
        self.q = np.asarray(self.q, dtype=float)
        self.dq = np.asarray(self.dq, dtype=float)
        if self.actuated_joint_names is None:
            self.actuated_joint_names = list(self.joint_names)

@dataclass
class MotionPlanResult:
    success: bool
    message: str
    trajectory: Trajectory | None
    geometric_path: BSplinePath | None


def _build_default_traj_config(model_cfg):
    from motion_planning.trajectory.cartesian_path_following import CartesianPathFollowingConfig

    return CartesianPathFollowingConfig(
        urdf_path=Path(model_cfg.urdf_path),
        actuated_joints=tuple(model_cfg.actuated_joints),
        passive_joints=tuple(model_cfg.passive_joints),
        lock_joint_names=tuple(model_cfg.locked_joints),
        joint_position_overrides=dict(model_cfg.joint_position_overrides),
        dynamics_mode="projected",
        tool_frame_name=model_cfg.target_frame,
        print_model_prep=False,
        xyz_weight=80.0,
        terminal_xyz_weight=200.0,
        s_weight=5.0,
        sdot_weight=3.0,
        qdd_u_weight=0.5,
        v_weight=1.5,
        terminal_s_weight=40.0,
        terminal_sdot_weight=6.0,
        terminal_dq_weight=5.0,
        passive_q_sway_weight=28.0,
        passive_dq_sway_weight=45.0,
        terminal_passive_q_sway_weight=80.0,
        terminal_passive_dq_sway_weight=180.0,
        sdot_ref=0.15,
    )

def plan_trajectory(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    traj_config=None,
    path: Optional[BSplinePath] = None,
    ctrl_pts_xyz: Optional[np.ndarray] = None,
    dq0: np.ndarray | None = None,
) -> "TrajectoryResult":
    from motion_planning.trajectory.cartesian_path_following import (
        CartesianPathFollowingConfig,
        CartesianPathFollowingOptimizer,
    )
    from motion_planning.trajectory.path_following import (
        CranePathFollowingAcadosConfig,
        CranePathFollowingAcadosOptimizer,
    )

    q_start = np.asarray(q_start, dtype=float).reshape(-1)
    q_goal = np.asarray(q_goal, dtype=float).reshape(-1)
    dq0_arr = np.zeros_like(q_start) if dq0 is None else np.asarray(dq0, dtype=float).reshape(-1)

    if traj_config is None:
        from motion_planning.mechanics import create_crane_config

        traj_config = _build_default_traj_config(create_crane_config())

    config: dict[str, Any] = {"q0": q_start, "q_goal": q_goal, "dq0": dq0_arr}
    if ctrl_pts_xyz is not None:
        config["ctrl_pts_xyz"] = np.asarray(ctrl_pts_xyz, dtype=float)

    req = TrajectoryRequest(scenario=None, path=path, config=config)

    if isinstance(traj_config, CartesianPathFollowingConfig):
        return CartesianPathFollowingOptimizer(traj_config).optimize(req)
    return CranePathFollowingAcadosOptimizer(traj_config).optimize(req)

class MotionPlanner:
    def __init__(
        self,
        *,
        scene: WorldModel | GeometryScene | None = None,
        method: str = "POWELL",
        traj_config=None,
        moving_block_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
        optimized_params_file: str | Path | None = None,
        geometric_options: dict[str, Any] | None = None,
    ) -> None:
        self._scene = self._coerce_scene(scene)
        self._method = method
        self._moving_block_size = tuple(float(v) for v in moving_block_size)
        self._optimized_params_file = Path(optimized_params_file) if optimized_params_file else None
        self._geo_options = dict(geometric_options or {})
        self._traj_config = traj_config
        self._optimizer = None
        self._model_cfg = None
        self._kin = None
        self._full_joint_name_to_vidx: dict[str, int] | None = None
        self._reduced_to_full_vidx: list[int] | None = None
        self._full_dec_nv: int | None = None
        self._reduced_dec_nv: int | None = None
        self._T_world_base: np.ndarray | None = None

    @property
    def scene(self) -> WorldModel:
        return self._scene

    def _trajectory_from_result(self, traj_result: TrajectoryResult) -> Trajectory | None:
        if not traj_result.success:
            return None
        diag = traj_result.diagnostics
        return Trajectory(
            time_s=traj_result.time_s,
            q=np.asarray(diag["q_trajectory"], dtype=float),
            dq=np.asarray(diag["dq_trajectory"], dtype=float),
            joint_names=list(diag["reduced_joint_names"]),
            actuated_joint_names=list(diag["actuated_joint_names"]),
        )

    def _trajectory_config(self, q_start: np.ndarray, q_goal: np.ndarray, geo_result: PlannerResult) -> dict[str, Any]:
        config: dict[str, Any] = {"q0": q_start, "q_goal": q_goal, "dq0": np.zeros_like(q_start)}
        if not (geo_result.success and geo_result.path is not None):
            return config
        ctrl_pts_planner = geo_result.diagnostics.get("ctrl_pts_xyz")
        ctrl_pts_xyz = self._base_pts_to_world(
            np.asarray(ctrl_pts_planner if ctrl_pts_planner is not None else geo_result.path.sample(
                getattr(self._traj_config, "spline_ctrl_points", 4)
            ), dtype=float)
        )
        config["ctrl_pts_xyz"] = ctrl_pts_xyz
        config["spline_ctrl_points"] = int(ctrl_pts_xyz.shape[0])
        if "spline_degree" in geo_result.diagnostics:
            config["spline_degree"] = int(geo_result.diagnostics["spline_degree"])
        return config

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> MotionPlanResult:
        self._ensure_initialized()
        q_start = self._to_reduced_dec_q(np.asarray(q_start, dtype=float).reshape(-1))
        q_goal = self._to_reduced_dec_q(np.asarray(q_goal, dtype=float).reshape(-1))

        xyz_start = self._fk_xyz(q_start)
        xyz_goal = self._fk_xyz(q_goal)
        scenario = Scenario(
            scene=self._scene.scene,
            start=tuple(xyz_start),
            goal=tuple(xyz_goal),
            moving_block_size=self._moving_block_size,
        )
        geo_result = self._run_geometric_stage(scenario)
        traj_result = self._optimizer.optimize(
            TrajectoryRequest(
                scenario=scenario,
                path=geo_result.path,
                config=self._trajectory_config(q_start, q_goal, geo_result),
            )
        )

        return MotionPlanResult(
            success=traj_result.success,
            message=traj_result.message,
            trajectory=self._trajectory_from_result(traj_result),
            geometric_path=geo_result.path if geo_result.success else None,
        )

    def compile_trajectory_ocp(self, q_hint: Optional[np.ndarray] = None) -> None:
        self._ensure_initialized()
        q0 = (
            self._to_reduced_dec_q(np.asarray(q_hint, dtype=float).reshape(-1))
            if q_hint is not None
            else np.zeros(int(self._reduced_dec_nv), dtype=float)
        )
        req = TrajectoryRequest(
            scenario=None,
            path=None,
            config={
                "q0": q0,
                "q_goal": q0.copy(),
                "dq0": np.zeros_like(q0),
            },
        )
        self._optimizer.optimize(req)

    @staticmethod
    def _coerce_scene(scene: WorldModel | GeometryScene | None) -> WorldModel:
        if scene is None:
            return WorldModel()
        if isinstance(scene, WorldModel):
            return scene
        if isinstance(scene, GeometryScene):
            return WorldModel.from_scene(scene)
        raise TypeError(f"scene must be WorldModel or geometry.scene.Scene, got {type(scene)!r}")

    def _ensure_initialized(self) -> None:
        if self._kin is not None:
            return

        import pinocchio as pin
        from motion_planning.mechanics import create_crane_config
        from motion_planning.mechanics import CraneKinematics
        from motion_planning.trajectory.path_following import (
            CranePathFollowingAcadosOptimizer,
        )
        from motion_planning.trajectory.cartesian_path_following import (
            CartesianPathFollowingConfig,
            CartesianPathFollowingOptimizer,
        )

        self._model_cfg = create_crane_config()
        self._kin = CraneKinematics(self._model_cfg.urdf_path)

        fk_base = self._kin.forward_kinematics(
            self._kin.neutral(),
            base_frame="world",
            end_frame=self._model_cfg.base_frame,
        )
        self._T_world_base = np.asarray(fk_base["base_to_end"]["homogeneous"], dtype=float)

        full_name_to_jid = {
            str(self._kin.model.names[jid]): int(jid)
            for jid in range(1, self._kin.model.njoints)
        }
        lock_ids = [
            full_name_to_jid[jn]
            for jn in self._model_cfg.locked_joints
            if jn in full_name_to_jid
        ]
        reduced_model = pin.buildReducedModel(
            self._kin.model, lock_ids, pin.neutral(self._kin.model)
        )
        self._full_dec_nv = int(self._kin.model.nv)
        self._reduced_dec_nv = int(reduced_model.nv)
        self._full_joint_name_to_vidx = {
            str(self._kin.model.names[jid]): int(self._kin.model.joints[jid].idx_v)
            for jid in range(1, self._kin.model.njoints)
            if int(self._kin.model.joints[jid].nq) in (1, 2)
        }
        self._reduced_to_full_vidx = [
            self._full_joint_name_to_vidx[str(reduced_model.names[jid])]
            for jid in range(1, reduced_model.njoints)
        ]

        if self._traj_config is None:
            self._traj_config = _build_default_traj_config(self._model_cfg)

        if isinstance(self._traj_config, CartesianPathFollowingConfig):
            self._optimizer = CartesianPathFollowingOptimizer(self._traj_config)
        else:
            self._optimizer = CranePathFollowingAcadosOptimizer(self._traj_config)

    def _base_pts_to_world(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        ones = np.ones((pts.shape[0], 1), dtype=float)
        return (self._T_world_base @ np.hstack([pts, ones]).T).T[:, :3]

    def _dec_to_joint_map(self, q_dec: np.ndarray) -> dict[str, float]:
        q_full = self._to_full_dec_q(np.asarray(q_dec, dtype=float).reshape(-1))
        return {name: float(q_full[idx]) for name, idx in self._full_joint_name_to_vidx.items()}

    def _fk_xyz(self, q_dec: np.ndarray) -> np.ndarray:
        xyz, _, _ = self._kin.pose_from_joint_map(
            self._dec_to_joint_map(q_dec),
            base_frame=self._model_cfg.base_frame,
            end_frame=self._model_cfg.target_frame,
        )
        return xyz

    def _to_reduced_dec_q(self, q_dec: np.ndarray) -> np.ndarray:
        q = np.asarray(q_dec, dtype=float).reshape(-1)
        full_nv = int(self._full_dec_nv)
        red_nv = int(self._reduced_dec_nv)
        if q.shape[0] == red_nv:
            return q.copy()
        if q.shape[0] == full_nv:
            return np.asarray([q[i] for i in self._reduced_to_full_vidx], dtype=float)
        raise ValueError(f"q must have length reduced_nv={red_nv} or full_nv={full_nv}, got {q.shape[0]}.")

    def _to_full_dec_q(self, q_dec: np.ndarray) -> np.ndarray:
        q = np.asarray(q_dec, dtype=float).reshape(-1)
        full_nv = int(self._full_dec_nv)
        red_nv = int(self._reduced_dec_nv)
        if q.shape[0] == full_nv:
            return q.copy()
        if q.shape[0] == red_nv:
            q_full = np.zeros(full_nv, dtype=float)
            for i_red, i_full in enumerate(self._reduced_to_full_vidx):
                q_full[i_full] = q[i_red]
            return q_full
        raise ValueError(f"q must have length reduced_nv={red_nv} or full_nv={full_nv}, got {q.shape[0]}.")

    def _geometric_inputs(self, scenario: Scenario) -> tuple[str, dict[str, Any], dict[str, Any]]:
        from motion_planning.optimized_params import canonical_method_name, load_optimized_planner_params

        method = canonical_method_name(self._method)
        config: dict[str, Any] = {}
        options = dict(self._geo_options)
        if self._optimized_params_file is not None:
            entries = load_optimized_planner_params(self._optimized_params_file)
            if method in entries:
                entry = entries[method]
                config = dict(entry["config"])
                if not options:
                    options = dict(entry["options"])

        start = np.asarray(scenario.start, dtype=float).reshape(3)
        goal = np.asarray(scenario.goal, dtype=float).reshape(3)
        t = np.linspace(0.0, 1.0, int(config.get("n_vias", 3)) + 2, dtype=float)[1:-1]
        config["initial_vias"] = np.vstack([(1.0 - a) * start + a * goal for a in t])
        return method, config, options

    def _run_geometric_stage(self, scenario: Scenario) -> PlannerResult:
        from motion_planning.api import run_geometric_planning

        method, config, options = self._geometric_inputs(scenario)
        return run_geometric_planning(scenario=scenario, method=method, config=config, options=options)
