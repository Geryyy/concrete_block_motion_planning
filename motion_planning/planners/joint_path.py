from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize

from motion_planning.core.spline import BSplinePath
from motion_planning.core.types import PlannerRequest, PlannerResult
from motion_planning.geometry.arm_model import (
    DEFAULT_PZS100_RAIL_POSITION_M,
    CraneArmCollisionModel,
)
from motion_planning.pipeline.joint_goal_stage import JointGoalStage
from motion_planning.trajectory.planning_limits import load_planning_limits_yaml


def _normalize(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    nrm = float(np.linalg.norm(arr))
    if nrm <= eps:
        return np.zeros_like(arr)
    return arr / nrm


def _wrap_to_pi(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _piecewise_reference(
    start: np.ndarray,
    goal: np.ndarray,
    desired_approach: np.ndarray,
    n_points: int,
    approach_distance: float,
    approach_fraction: float,
) -> np.ndarray:
    u = np.linspace(0.0, 1.0, int(n_points), dtype=float)
    pre_goal = goal - float(approach_distance) * np.asarray(desired_approach, dtype=float)
    out = np.zeros((u.size, 3), dtype=float)
    split = min(0.95, max(0.2, float(approach_fraction)))
    for i, ui in enumerate(u):
        if ui <= split:
            alpha = ui / split
            out[i, :] = (1.0 - alpha) * start + alpha * pre_goal
        else:
            alpha = (ui - split) / (1.0 - split)
            out[i, :] = (1.0 - alpha) * pre_goal + alpha * goal
    return out


@dataclass
class JointSpaceStage1Planner:
    method: str

    def __post_init__(self) -> None:
        self._goal_stage = JointGoalStage()
        self._arm_model = CraneArmCollisionModel()
        planning_limits = (
            Path(__file__).resolve().parents[1] / "trajectory" / "planning_limits.yaml"
        )
        limits, _ = load_planning_limits_yaml(planning_limits)
        self._joint_names = list(self._goal_stage.config.actuated_joints)
        self._bounds = np.asarray(
            [
                [
                    float(limits.get(name, (None, None))[0] if limits.get(name, (None, None))[0] is not None else -10.0),
                    float(limits.get(name, (None, None))[1] if limits.get(name, (None, None))[1] is not None else 10.0),
                ]
                for name in self._joint_names
            ],
            dtype=float,
        )

    def _reduced_q(self, q_map: Dict[str, float]) -> np.ndarray:
        return np.asarray([float(q_map[name]) for name in self._joint_names], dtype=float)

    def _fk_path(
        self,
        q_waypoints: np.ndarray,
        rail_position: float,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
        xyz = np.zeros((q_waypoints.shape[0], 3), dtype=float)
        yaw = np.zeros((q_waypoints.shape[0],), dtype=float)
        q_maps: list[dict[str, float]] = []
        seed: dict[str, float] | None = None
        for i, q_red in enumerate(np.asarray(q_waypoints, dtype=float)):
            q_map = self._arm_model.complete_joint_map(
                q_red,
                q_passive=seed,
                rail_position=rail_position,
            )
            q_maps.append(q_map)
            xyz_i, yaw_i, seed = self._joint_world_pose(q_red, q_map)
            xyz[i, :] = xyz_i
            yaw[i] = yaw_i
        return xyz, yaw, q_maps

    def _joint_world_pose(
        self,
        q_red: np.ndarray,
        q_map: dict[str, float],
    ) -> tuple[np.ndarray, float, dict[str, float]]:
        del q_red
        q_pin = self._goal_stage._kin.pin.neutral(self._goal_stage._kin.model)
        model = self._goal_stage._kin.model
        for jid in range(1, model.njoints):
            name = str(model.names[jid])
            if name not in q_map:
                continue
            joint = model.joints[jid]
            nq = int(joint.nq)
            iq = int(joint.idx_q)
            val = float(q_map[name])
            if nq == 1:
                q_pin[iq] = val
            elif nq == 2 and int(joint.nv) == 1:
                q_pin[iq] = math.cos(val)
                q_pin[iq + 1] = math.sin(val)
        fk = self._goal_stage._kin.forward_kinematics(
            q_pin,
            base_frame="K0_mounting_base",
            end_frame=self._goal_stage.config.target_frame,
        )
        t_mat = np.asarray(fk["base_to_end"]["homogeneous"], dtype=float)
        xyz = np.asarray(t_mat[:3, 3], dtype=float)
        yaw = float(math.atan2(t_mat[1, 0], t_mat[0, 0]))
        return xyz, yaw, q_map

    def _build_waypoints(
        self,
        q_start: np.ndarray,
        flat_vars: np.ndarray,
        n_waypoints: int,
        goal_fixed: bool,
        q_goal_fixed: np.ndarray | None = None,
    ) -> np.ndarray:
        out = np.zeros((int(n_waypoints), q_start.size), dtype=float)
        out[0, :] = q_start
        if goal_fixed:
            if q_goal_fixed is None:
                raise ValueError("q_goal_fixed must be provided when goal_fixed=True")
            out[-1, :] = q_goal_fixed
            n_rows = max(0, int(n_waypoints) - 2)
            if n_rows > 0:
                out[1:-1, :] = np.asarray(flat_vars, dtype=float).reshape(n_rows, q_start.size)
        else:
            n_rows = max(0, int(n_waypoints) - 1)
            if n_rows > 0:
                out[1:, :] = np.asarray(flat_vars, dtype=float).reshape(n_rows, q_start.size)
        return np.clip(out, self._bounds[:, 0], self._bounds[:, 1])

    def _cem_optimize(
        self,
        x0: np.ndarray,
        bounds: np.ndarray,
        objective,
        options: Dict[str, Any],
    ) -> tuple[np.ndarray, float, int]:
        pop_size = int(options.get("cem_population", 48))
        elite_count = int(options.get("cem_elite_count", max(6, pop_size // 4)))
        max_iter = int(options.get("maxiter", 30))
        sigma = np.maximum(
            np.asarray(options.get("cem_sigma_init", 0.18), dtype=float),
            1e-3,
        )
        if sigma.ndim == 0:
            sigma = np.full_like(x0, float(sigma))
        mean = np.asarray(x0, dtype=float).copy()
        best_x = mean.copy()
        best_val = float(objective(best_x))
        rng = np.random.default_rng(int(options.get("seed", 7)))
        for it in range(max_iter):
            samples = rng.normal(loc=mean, scale=sigma, size=(pop_size, mean.size))
            samples = np.clip(samples, bounds[:, 0], bounds[:, 1])
            vals = np.asarray([objective(sample) for sample in samples], dtype=float)
            elite_idx = np.argsort(vals)[:elite_count]
            elite = samples[elite_idx]
            mean = np.mean(elite, axis=0)
            sigma = np.maximum(np.std(elite, axis=0), 1e-3)
            if float(vals[elite_idx[0]]) < best_val:
                best_val = float(vals[elite_idx[0]])
                best_x = elite[0].copy()
        return best_x, best_val, max_iter

    def plan(self, req: PlannerRequest) -> PlannerResult:
        sc = req.scenario
        cfg = dict(req.config)
        options = dict(req.options)

        start_yaw = math.radians(float(sc.start_yaw_deg))
        goal_yaw = math.radians(float(sc.goal_yaw_deg))
        start_sol = self._goal_stage.solve_world_pose(goal_world=sc.start, target_yaw_rad=start_yaw)
        goal_sol = self._goal_stage.solve_world_pose(goal_world=sc.goal, target_yaw_rad=goal_yaw)
        if not start_sol.success:
            return PlannerResult(False, f"start IK failed: {start_sol.message}", BSplinePath(lambda u: np.zeros((np.asarray(u).size, 3))), {}, {})

        q_start = self._reduced_q(start_sol.q_dynamic)
        desired_approach = -_normalize(np.sum(np.asarray(sc.goal_normals, dtype=float), axis=0))
        if not np.any(desired_approach):
            desired_approach = _normalize(np.asarray(sc.goal, dtype=float) - np.asarray(sc.start, dtype=float))
        n_waypoints = int(cfg.get("joint_waypoint_count", 9))
        rail_position = float(cfg.get("fixed_rail_position_m", DEFAULT_PZS100_RAIL_POSITION_M))
        preferred_clearance = float(cfg.get("preferred_safety_margin", cfg.get("safety_margin", 0.05)))
        approach_distance = float(
            cfg.get("approach_distance_m", max(0.4, 0.6 * float(np.linalg.norm(np.asarray(sc.moving_block_size, dtype=float)))))
        )
        approach_fraction = float(cfg.get("goal_approach_window_fraction", 0.75))

        xyz_ref = _piecewise_reference(
            np.asarray(sc.start, dtype=float),
            np.asarray(sc.goal, dtype=float),
            desired_approach,
            n_waypoints,
            approach_distance,
            approach_fraction,
        )
        yaw_ref = np.linspace(start_yaw, goal_yaw, n_waypoints, dtype=float)
        goal_fixed = bool(goal_sol.success)
        if goal_fixed:
            q_goal_seed = self._reduced_q(goal_sol.q_dynamic)
        else:
            pre_family = self._goal_stage.solve_preapproach_family(
                start_world=sc.start,
                target_world=sc.goal,
                target_yaw_rad=goal_yaw,
                offsets_m=[approach_distance, 0.75 * approach_distance, 0.5 * approach_distance],
                q_seed=start_sol.q_dynamic,
            )
            successful = next((item for item in pre_family if item.success), None)
            q_goal_seed = self._reduced_q(successful.q_dynamic) if successful is not None else q_start.copy()

        alpha = np.linspace(0.0, 1.0, n_waypoints, dtype=float).reshape(-1, 1)
        q_init = (1.0 - alpha) * q_start.reshape(1, -1) + alpha * q_goal_seed.reshape(1, -1)
        x0 = q_init[1:-1, :].reshape(-1) if goal_fixed else q_init[1:, :].reshape(-1)
        bounds = np.tile(self._bounds, (max(0, n_waypoints - 2) if goal_fixed else max(0, n_waypoints - 1), 1))

        weights = {
            "track": float(cfg.get("jointspace_track_cost", 60.0)),
            "yaw": float(cfg.get("jointspace_yaw_cost", 20.0)),
            "terminal": float(cfg.get("jointspace_terminal_cost", 180.0)),
            "approach": float(cfg.get("goal_approach_normal_cost", 140.0)),
            "smooth": float(cfg.get("jointspace_smoothness_cost", 3.0)),
            "curvature": float(cfg.get("jointspace_curvature_cost", 6.0)),
            "collision": float(cfg.get("collision_cost", 18.0)),
            "penetration": float(cfg.get("penetration_cost", 240.0)),
        }

        def objective(x_flat: np.ndarray) -> float:
            q_waypoints = self._build_waypoints(
                q_start,
                x_flat,
                n_waypoints,
                goal_fixed=goal_fixed,
                q_goal_fixed=q_goal_seed if goal_fixed else None,
            )
            xyz, yaw, q_maps = self._fk_path(q_waypoints, rail_position)
            pos_err = xyz - xyz_ref
            yaw_err = np.asarray([_wrap_to_pi(float(yaw[i] - yaw_ref[i])) for i in range(yaw.size)], dtype=float)
            cost = 0.0
            cost += weights["track"] * float(np.mean(np.sum(pos_err * pos_err, axis=1)))
            cost += weights["yaw"] * float(np.mean(yaw_err * yaw_err))
            cost += weights["terminal"] * float(np.sum((xyz[-1] - np.asarray(sc.goal, dtype=float)) ** 2))
            cost += weights["terminal"] * float(_wrap_to_pi(float(yaw[-1] - goal_yaw)) ** 2)
            if xyz.shape[0] >= 2:
                final_vec = _normalize(xyz[-1] - xyz[-2])
                align_cos = float(np.clip(np.dot(final_vec, desired_approach), -1.0, 1.0))
                cost += weights["approach"] * float((1.0 - align_cos) ** 2)
            dq = np.diff(q_waypoints, axis=0)
            ddq = np.diff(q_waypoints, axis=0, n=2) if q_waypoints.shape[0] >= 3 else np.zeros((0, q_waypoints.shape[1]))
            cost += weights["smooth"] * float(np.mean(np.sum(dq * dq, axis=1)))
            if ddq.size:
                cost += weights["curvature"] * float(np.mean(np.sum(ddq * ddq, axis=1)))
            for q_map in q_maps:
                dist = float(self._arm_model.clearance(q_map, sc.scene, ignore_ids=["table"]))
                low_clear = max(0.0, preferred_clearance - dist)
                penetration = max(0.0, -dist)
                cost += weights["collision"] * low_clear * low_clear
                cost += weights["penetration"] * penetration * penetration
            return float(cost)

        nit = 0
        if x0.size:
            if str(self.method).upper() == "CEM":
                x_best, best_obj, nit = self._cem_optimize(x0, bounds, objective, options)
            else:
                scipy_method = "Powell" if str(self.method).upper() == "POWELL" else "Nelder-Mead"
                res = minimize(
                    objective,
                    x0,
                    method=scipy_method,
                    options={"maxiter": int(options.get("maxiter", 80)), "disp": False},
                )
                x_best = np.asarray(res.x, dtype=float)
                best_obj = float(res.fun)
                nit = int(getattr(res, "nit", 0))
        else:
            x_best = x0
            best_obj = float(objective(x_best))

        q_waypoints = self._build_waypoints(
            q_start,
            x_best,
            n_waypoints,
            goal_fixed=goal_fixed,
            q_goal_fixed=q_goal_seed if goal_fixed else None,
        )
        u_anchor = np.linspace(0.0, 1.0, n_waypoints, dtype=float)
        u_dense = np.linspace(0.0, 1.0, int(cfg.get("n_samples_curve", 101)), dtype=float)
        q_dense = np.zeros((u_dense.size, q_waypoints.shape[1]), dtype=float)
        for j in range(q_waypoints.shape[1]):
            interp = PchipInterpolator(u_anchor, q_waypoints[:, j], extrapolate=False)
            q_dense[:, j] = np.asarray(interp(u_dense), dtype=float)
            q_dense[0, j] = q_waypoints[0, j]
            q_dense[-1, j] = q_waypoints[-1, j]

        tcp_xyz, tcp_yaw, q_maps_dense = self._fk_path(q_dense, rail_position)
        clearance_report = self._arm_model.check_path(
            q_maps_dense,
            tcp_xyz,
            tcp_yaw,
            sc.moving_block_size,
            sc.scene,
            arm_ignore_ids=["table"],
        )
        final_vec = _normalize(tcp_xyz[-1] - tcp_xyz[-2]) if tcp_xyz.shape[0] >= 2 else np.zeros(3, dtype=float)
        approach_cos = float(np.clip(np.dot(final_vec, desired_approach), -1.0, 1.0))
        approach_angle_deg = float(np.degrees(np.arccos(approach_cos)))

        q_full = np.asarray(
            [
                [
                    q_map["theta1_slewing_joint"],
                    q_map["theta2_boom_joint"],
                    q_map["theta3_arm_joint"],
                    q_map["q4_big_telescope"],
                    q_map["theta6_tip_joint"],
                    q_map["theta7_tilt_joint"],
                    q_map["theta8_rotator_joint"],
                    q_map["q9_left_rail_joint"],
                ]
                for q_map in q_maps_dense
            ],
            dtype=float,
        )

        def xyz_fn(uq: np.ndarray) -> np.ndarray:
            uq_arr = np.asarray(uq, dtype=float).reshape(-1)
            return np.column_stack([np.interp(uq_arr, u_dense, tcp_xyz[:, i]) for i in range(3)])

        def yaw_fn(uq: np.ndarray) -> np.ndarray:
            uq_arr = np.asarray(uq, dtype=float).reshape(-1)
            return np.degrees(np.interp(uq_arr, u_dense, tcp_yaw))

        metrics = {
            "fun": float(best_obj),
            "min_clearance": float(clearance_report.combined_min_clearance_m),
            "mean_clearance": float(
                np.mean(
                    [
                        min(
                            self._arm_model.clearance(q_map, sc.scene, ignore_ids=["table"]),
                            self._arm_model.payload_clearance(
                                tcp_xyz[i],
                                float(tcp_yaw[i]),
                                sc.moving_block_size,
                                sc.scene,
                            ),
                        )
                        for i, q_map in enumerate(q_maps_dense)
                    ]
                )
            ),
            "goal_approach_normal_cost": float((1.0 - approach_cos) ** 2),
            "turn_angle_mean_deg": float(np.degrees(np.mean(np.abs(np.diff(tcp_yaw)))) if tcp_yaw.size > 1 else 0.0),
        }
        diagnostics: Dict[str, Any] = {
            "nit": int(nit),
            "solver_method": str(self.method),
            "q_waypoints_reduced": q_waypoints,
            "q_path_full": q_full,
            "q_maps_path": q_maps_dense,
            "tcp_xyz_path": tcp_xyz,
            "tcp_yaw_path_rad": tcp_yaw,
            "desired_approach_dir": desired_approach,
            "actual_approach_dir": final_vec,
            "approach_alignment_angle_deg": approach_angle_deg,
            "arm_clearance_report": clearance_report,
            "rail_position_m": rail_position,
            "final_position_error_m": float(np.linalg.norm(tcp_xyz[-1] - np.asarray(sc.goal, dtype=float))),
            "final_yaw_error_deg": float(np.degrees(abs(_wrap_to_pi(float(tcp_yaw[-1] - goal_yaw))))),
            "arm_min_clearance_m": float(clearance_report.arm_min_clearance_m),
            "payload_min_clearance_m": float(clearance_report.payload_min_clearance_m),
            "combined_min_clearance_m": float(clearance_report.combined_min_clearance_m),
        }
        return PlannerResult(
            success=bool(clearance_report.combined_min_clearance_m > -0.10),
            message="Joint-space stage-1 path optimized in Cartesian cost space",
            path=BSplinePath(xyz_fn=xyz_fn, yaw_fn=yaw_fn),
            metrics=metrics,
            diagnostics=diagnostics,
        )
