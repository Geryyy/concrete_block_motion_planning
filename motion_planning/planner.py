"""Two-stage motion planner: geometric path planning + time-parametrized trajectory optimization."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from motion_planning.core.spline import BSplinePath
from motion_planning.core.types import PlannerResult, Scenario, TrajectoryRequest, TrajectoryResult
from motion_planning.core.world_model import WorldModel
from motion_planning.geometry.scene import Scene as GeometryScene
from motion_planning.trajectory.planning_limits import load_planning_limits_yaml


# ── Trajectory ────────────────────────────────────────────────────────────────


@dataclass
class Trajectory:
    """Time-parametrized joint-space trajectory produced by the optimizer."""

    time_s: np.ndarray
    q: np.ndarray              # (T, nq)  joint positions
    dq: np.ndarray             # (T, nq)  joint velocities
    joint_names: List[str]
    actuated_joint_names: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self.time_s = np.asarray(self.time_s, dtype=float).reshape(-1)
        self.q = np.asarray(self.q, dtype=float)
        self.dq = np.asarray(self.dq, dtype=float)
        if self.actuated_joint_names is None:
            self.actuated_joint_names = list(self.joint_names)

    def save(self, path: Union[str, Path]) -> Path:
        """Save to .npz compatible with replay_trajectory_with_pd."""
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            time_s=self.time_s,
            q_trajectory=self.q,
            dq_trajectory=self.dq,
            reduced_joint_names=np.asarray(self.joint_names, dtype=str),
            actuated_joint_names=np.asarray(self.actuated_joint_names, dtype=str),
        )
        return path

    def plot(self) -> None:
        """Plot joint positions and velocities over time."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax = axes[0]
        for i, name in enumerate(self.joint_names):
            ax.plot(self.time_s, np.degrees(self.q[:, i]), label=name)
        ax.set_ylabel("Joint angle [deg]")
        ax.legend(fontsize=8, ncol=3)
        ax.grid(True)
        ax.set_title("Joint Positions")

        ax = axes[1]
        for i, name in enumerate(self.joint_names):
            ax.plot(self.time_s, np.degrees(self.dq[:, i]), label=name)
        ax.set_ylabel("Joint velocity [deg/s]")
        ax.set_xlabel("Time [s]")
        ax.legend(fontsize=8, ncol=3)
        ax.grid(True)
        ax.set_title("Joint Velocities")

        fig.tight_layout()
        plt.show()


# ── MotionPlanResult ──────────────────────────────────────────────────────────


@dataclass
class MotionPlanResult:
    """Combined result of geometric planning and trajectory optimization."""

    success: bool
    message: str
    trajectory: Optional[Trajectory]
    geometric_path: Optional[BSplinePath]

    def plot(self) -> None:
        """Plot the planned trajectory (joint angles and velocities)."""
        if self.trajectory is None:
            raise RuntimeError("No trajectory to plot.")
        self.trajectory.plot()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _dec_to_pin_q(model, q_dec: np.ndarray) -> np.ndarray:
    """Convert decision-space q (nv,) to Pinocchio configuration q (nq,)."""
    import pinocchio as pin

    q_pin = np.asarray(pin.neutral(model), dtype=float)
    for jid in range(1, model.njoints):
        jm = model.joints[jid]
        iv = int(jm.idx_v)
        iq = int(jm.idx_q)
        if int(jm.nq) == 1:
            q_pin[iq] = q_dec[iv]
        elif int(jm.nq) == 2 and int(jm.nv) == 1:
            th = q_dec[iv]
            q_pin[iq] = np.cos(th)
            q_pin[iq + 1] = np.sin(th)
    return q_pin


def _build_default_traj_config(model_cfg):
    """Build the default CartesianPathFollowingConfig from the model config."""
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
        # Tuned weights.
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


# ── Standalone trajectory-only planning ───────────────────────────────────────


def plan_trajectory(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    traj_config=None,
    path: Optional[BSplinePath] = None,
    ctrl_pts_xyz: Optional[np.ndarray] = None,
    dq0: Optional[np.ndarray] = None,
) -> "TrajectoryResult":
    """Run trajectory optimisation without a geometric planning stage.

    Suitable for use when the joint configurations are already known (e.g. from
    inverse kinematics) or when a Cartesian path is supplied directly.

    Args:
        q_start: Start joint configuration — full-model decision space (nv,).
        q_goal:  Goal  joint configuration — full-model decision space (nv,).
        traj_config: A ``CartesianPathFollowingConfig`` (default) or a
            ``CranePathFollowingAcadosConfig``.  When *None*, a pre-tuned
            ``CartesianPathFollowingConfig`` is built from the bundled URDF.
        path: Optional ``BSplinePath`` from a prior geometric planning call.
            Used to generate Cartesian control points when ``ctrl_pts_xyz`` is
            not supplied explicitly.
        ctrl_pts_xyz: Explicit Cartesian control points ``(n_ctrl, 3)`` for the
            ``CartesianPathFollowingOptimizer``.  Takes priority over ``path``.
        dq0: Initial joint velocities.  Defaults to zeros.

    Returns:
        ``TrajectoryResult`` with ``.success``, ``.state``, ``.control``,
        ``.time_s``, and ``.diagnostics``.
    """
    from motion_planning.core.types import TrajectoryRequest, TrajectoryResult
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
        from motion_planning.mechanics.analytic import create_crane_config
        traj_config = _build_default_traj_config(create_crane_config())

    config: Dict[str, Any] = {"q0": q_start, "q_goal": q_goal, "dq0": dq0_arr}
    if ctrl_pts_xyz is not None:
        config["ctrl_pts_xyz"] = np.asarray(ctrl_pts_xyz, dtype=float)

    req = TrajectoryRequest(scenario=None, path=path, config=config)

    if isinstance(traj_config, CartesianPathFollowingConfig):
        return CartesianPathFollowingOptimizer(traj_config).optimize(req)
    return CranePathFollowingAcadosOptimizer(traj_config).optimize(req)


# ── MotionPlanner ─────────────────────────────────────────────────────────────


class MotionPlanner:
    """Two-stage motion planner combining collision-free geometric planning and
    time-parametrized trajectory optimization.

    Two connected stages:

    - **Stage 1** (geometric): finds a collision-free Cartesian path from the
      TCP start to goal positions using the configured method (POWELL by default).
      The resulting ``BSplinePath`` is stored in ``MotionPlanResult.geometric_path``.

    - **Stage 2** (trajectory): produces a dynamically-feasible, time-parametrized
      joint trajectory using the Cartesian path-following acados OCP. The
      geometric path is sampled into Cartesian B-spline control points and used
      as the task-space reference ``xyz_ref(s)`` in the OCP cost via symbolic FK.

    Usage::

        scene = WorldModel()
        scene.add_block(size=(2, 2, 2), position=(5, 0, 1))

        planner = MotionPlanner(scene=scene)
        result = planner.plan(q_start, q_goal)

        result.plot()
    """

    def __init__(
        self,
        *,
        scene: Optional[Union[WorldModel, GeometryScene]] = None,
        method: str = "POWELL",
        traj_config=None,
        moving_block_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        optimized_params_file: Optional[Union[str, Path]] = None,
        geometric_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            scene: Collision scene as ``WorldModel`` or ``geometry.scene.Scene``.
                Mutate between ``plan()`` calls to reflect dynamic environments.
            method: Geometric planner method — ``"CEM"``, ``"POWELL"``, or
                ``"NELDER-MEAD"``.
            traj_config: Custom ``CartesianPathFollowingConfig`` (default) or
                ``CranePathFollowingAcadosConfig``. Defaults to a pre-tuned
                Cartesian path-following configuration derived from the model YAML.
            moving_block_size: Payload bounding box ``(x, y, z)`` in metres
                used for collision checking in the geometric stage.
            optimized_params_file: YAML with pre-tuned geometric planner params.
            geometric_options: Extra solver options forwarded to the geometric
                planner (e.g. ``{"maxiter": 200}``).
        """
        if scene is None:
            self._scene = WorldModel()
        elif isinstance(scene, WorldModel):
            self._scene = scene
        elif isinstance(scene, GeometryScene):
            self._scene = WorldModel.from_scene(scene)
        else:
            raise TypeError(
                "scene must be WorldModel or geometry.scene.Scene, "
                f"got {type(scene)!r}"
            )
        self._method = method
        self._moving_block_size: Tuple[float, float, float] = tuple(
            float(v) for v in moving_block_size
        )
        self._optimized_params_file = (
            Path(optimized_params_file) if optimized_params_file else None
        )
        self._geo_options: Dict[str, Any] = dict(geometric_options or {})

        # Lazy fields — populated on first plan() call.
        self._traj_config = traj_config
        self._optimizer = None
        self._model_cfg = None
        self._kin = None
        self._reduced_joint_names: Optional[List[str]] = None
        self._full_joint_name_to_vidx: Optional[Dict[str, int]] = None
        self._reduced_to_full_vidx: Optional[List[int]] = None
        self._full_dec_nv: Optional[int] = None
        self._reduced_dec_nv: Optional[int] = None
        self._taskspace_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._T_world_base: Optional[np.ndarray] = None   # base frame → world

    @property
    def scene(self) -> WorldModel:
        """The mutable collision scene. Update obstacles between ``plan()`` calls."""
        return self._scene

    # ── Public API ────────────────────────────────────────────────────────────

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> MotionPlanResult:
        """Plan a collision-free, time-parametrized trajectory.

        Args:
            q_start: Start joint configuration in full-model decision space (nv,).
            q_goal: Goal joint configuration in full-model decision space (nv,).

        Returns:
            ``MotionPlanResult`` containing the trajectory, geometric path,
            success flag, and diagnostic message.
        """
        self._ensure_initialized()
        q_start = self._to_reduced_dec_q(np.asarray(q_start, dtype=float).reshape(-1))
        q_goal = self._to_reduced_dec_q(np.asarray(q_goal, dtype=float).reshape(-1))

        # ── Stage 1: geometric planning (Cartesian) ───────────────────────────
        xyz_start = self._fk_xyz(q_start)
        xyz_goal = self._fk_xyz(q_goal)
        self._assert_in_taskspace_bounds(xyz_start, "start_tcp")
        self._assert_in_taskspace_bounds(xyz_goal, "goal_tcp")
        scenario = Scenario(
            scene=self._scene.scene,
            start=tuple(xyz_start),
            goal=tuple(xyz_goal),
            moving_block_size=self._moving_block_size,
        )
        geo_result = self._run_geometric_stage(scenario)

        # ── Stage 2: trajectory optimization ─────────────────────────────────
        # Forward planner control points when available; this preserves the
        # geometric stage path shape in trajectory optimization.
        n_ctrl = getattr(self._traj_config, "spline_ctrl_points", 4)
        ctrl_pts_xyz: Optional[np.ndarray] = None
        if geo_result.success and geo_result.path is not None:
            ctrl_pts_planner = geo_result.diagnostics.get("ctrl_pts_xyz", None)
            if ctrl_pts_planner is not None:
                ctrl_pts_xyz = self._base_pts_to_world(np.asarray(ctrl_pts_planner, dtype=float))
            else:
                # sample returns (n_ctrl, 3) in K0_mounting_base frame; convert
                # to world frame so the OCP Cartesian residual (which uses
                # pinocchio world-frame FK) is computed in the correct frame.
                ctrl_pts_xyz = self._base_pts_to_world(geo_result.path.sample(n_ctrl))
            self._assert_points_in_taskspace_bounds(ctrl_pts_xyz, "geometric_path")

        traj_config: Dict[str, Any] = {
            "q0": q_start,
            "q_goal": q_goal,
            "dq0": np.zeros_like(q_start),
        }
        if ctrl_pts_xyz is not None:
            traj_config["ctrl_pts_xyz"] = ctrl_pts_xyz
            traj_config["spline_ctrl_points"] = int(ctrl_pts_xyz.shape[0])
            if "spline_degree" in geo_result.diagnostics:
                traj_config["spline_degree"] = int(geo_result.diagnostics["spline_degree"])

        req = TrajectoryRequest(
            scenario=scenario,
            path=geo_result.path,
            config=traj_config,
        )
        traj_result = self._optimizer.optimize(req)

        trajectory: Optional[Trajectory] = None
        if traj_result.success:
            diag = traj_result.diagnostics
            trajectory = Trajectory(
                time_s=traj_result.time_s,
                q=np.asarray(diag["q_trajectory"], dtype=float),
                dq=np.asarray(diag["dq_trajectory"], dtype=float),
                joint_names=list(diag["reduced_joint_names"]),
                actuated_joint_names=list(diag["actuated_joint_names"]),
            )

        return MotionPlanResult(
            success=traj_result.success,
            message=traj_result.message,
            trajectory=trajectory,
            geometric_path=geo_result.path if geo_result.success else None,
        )

    def compile_trajectory_ocp(self, q_hint: Optional[np.ndarray] = None) -> None:
        """Compile and warm up the trajectory OCP once for faster first plan()."""
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

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_initialized(self) -> None:
        """Lazy-initialize heavy dependencies on the first plan() call."""
        if self._kin is not None:
            return

        import pinocchio as pin
        from motion_planning.mechanics.analytic import create_crane_config
        from motion_planning.kinematics import CraneKinematics
        from motion_planning.trajectory.path_following import (
            CranePathFollowingAcadosConfig,
            CranePathFollowingAcadosOptimizer,
        )
        from motion_planning.trajectory.cartesian_path_following import (
            CartesianPathFollowingConfig,
            CartesianPathFollowingOptimizer,
        )

        self._model_cfg = create_crane_config()
        self._kin = CraneKinematics(self._model_cfg.urdf_path)

        # Constant transform: K0_mounting_base → world (independent of q).
        # The geometric planner and BSplinePath operate in the base frame;
        # the OCP FK (pinocchio.casadi) expresses the TCP in the world frame.
        # ctrl_pts_xyz must be converted to world frame before being passed to
        # the OCP, otherwise the Cartesian residual is wrong by ~16 m.
        q_neutral = pin.neutral(self._kin.model)
        fk_base = self._kin.forward_kinematics(
            q_neutral,
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
        self._reduced_joint_names = [
            str(reduced_model.names[jid]) for jid in range(1, reduced_model.njoints)
        ]
        self._reduced_to_full_vidx = [self._full_joint_name_to_vidx[name] for name in self._reduced_joint_names]

        if self._traj_config is None:
            self._traj_config = _build_default_traj_config(self._model_cfg)
        limits_yaml = getattr(self._traj_config, "joint_position_limits_yaml", None)
        if limits_yaml is not None:
            _, taskspace_bounds = load_planning_limits_yaml(Path(limits_yaml))
            if taskspace_bounds is not None:
                self._taskspace_bounds = (
                    np.asarray(taskspace_bounds[0], dtype=float).reshape(3),
                    np.asarray(taskspace_bounds[1], dtype=float).reshape(3),
                )

        if isinstance(self._traj_config, CartesianPathFollowingConfig):
            self._optimizer = CartesianPathFollowingOptimizer(self._traj_config)
        else:
            self._optimizer = CranePathFollowingAcadosOptimizer(self._traj_config)

    def _base_pts_to_world(self, pts: np.ndarray) -> np.ndarray:
        """Transform (N, 3) Cartesian points from planning base frame to world frame."""
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        ones = np.ones((pts.shape[0], 1), dtype=float)
        return (self._T_world_base @ np.hstack([pts, ones]).T).T[:, :3]

    def _fk_xyz(self, q_dec: np.ndarray) -> np.ndarray:
        """Forward kinematics: decision q → TCP position (3,) in base frame."""
        q_full = self._to_full_dec_q(np.asarray(q_dec, dtype=float).reshape(-1))
        q_pin = _dec_to_pin_q(self._kin.model, q_full)
        fk = self._kin.forward_kinematics(
            q_pin,
            base_frame=self._model_cfg.base_frame,
            end_frame=self._model_cfg.target_frame,
        )
        return np.asarray(fk["base_to_end"]["translation"], dtype=float)

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

    def _run_geometric_stage(self, scenario: Scenario) -> PlannerResult:
        from motion_planning.pipeline.geometric_stage import run_geometric_planning
        from motion_planning.io.optimized_params import (
            canonical_method_name,
            load_optimized_planner_params,
        )

        canonical = canonical_method_name(self._method)
        config: Dict[str, Any] = {}
        options: Dict[str, Any] = dict(self._geo_options)

        if self._optimized_params_file is not None:
            entries = load_optimized_planner_params(self._optimized_params_file)
            if canonical in entries:
                entry = entries[canonical]
                if not config:
                    config = dict(entry["config"])
                if not options:
                    options = dict(entry["options"])

        # Seed spline optimization from a straight-line via initialization.
        n_vias = int(config.get("n_vias", 3))
        t = np.linspace(0.0, 1.0, n_vias + 2, dtype=float)[1:-1]
        start = np.asarray(scenario.start, dtype=float).reshape(3)
        goal = np.asarray(scenario.goal, dtype=float).reshape(3)
        initial_vias = np.vstack([(1.0 - a) * start + a * goal for a in t])
        config["initial_vias"] = initial_vias

        return run_geometric_planning(
            scenario=scenario,
            method=canonical,
            config=config,
            options=options,
        )

    def _assert_in_taskspace_bounds(self, p_xyz: np.ndarray, label: str) -> None:
        if self._taskspace_bounds is None:
            return
        p = np.asarray(p_xyz, dtype=float).reshape(3)
        p_min, p_max = self._taskspace_bounds
        if np.any(p < p_min) or np.any(p > p_max):
            raise ValueError(
                f"{label}={p.tolist()} violates taskspace limits min={p_min.tolist()} max={p_max.tolist()}."
            )

    def _assert_points_in_taskspace_bounds(self, pts_xyz: np.ndarray, label: str) -> None:
        if self._taskspace_bounds is None:
            return
        pts = np.asarray(pts_xyz, dtype=float).reshape(-1, 3)
        p_min, p_max = self._taskspace_bounds
        bad = np.nonzero(np.any((pts < p_min) | (pts > p_max), axis=1))[0]
        if bad.size > 0:
            k = int(bad[0])
            raise ValueError(
                f"{label} point[{k}]={pts[k].tolist()} violates taskspace limits min={p_min.tolist()} max={p_max.tolist()}."
            )
