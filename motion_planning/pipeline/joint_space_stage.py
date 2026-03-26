from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator

from motion_planning.pipeline.joint_goal_stage import JointGoalStage


def _wrap_to_pi(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


@dataclass
class JointSpacePlanResult:
    success: bool
    message: str
    q_waypoints: np.ndarray
    xyz_waypoints: np.ndarray
    yaw_waypoints: np.ndarray
    diagnostics: dict[str, float]


class JointSpaceCartesianPlanner:
    """Bounded joint-space planning stage for online commissioning.

    This stage keeps optimization in reduced joint space, but uses the Cartesian
    reference path only to define sparse anchor poses. It intentionally avoids
    nonlinear optimization in the BT service path and instead uses:
    1. sparse Cartesian anchors along the reference path
    2. steady-state joint solves at those anchors
    3. deterministic joint-space spline fitting through solved anchors
    4. TOPP-RA after the path is fixed
    """

    def __init__(
        self,
        *,
        urdf_path: str,
        target_frame: str,
        reduced_joint_names: Sequence[str],
        joint_position_limits: Mapping[str, tuple[float | None, float | None]] | None = None,
        smoothing_passes: int = 0,
        max_projection_waypoints: int = 4,
        maxiter: int | None = None,
    ) -> None:
        del urdf_path, target_frame, maxiter  # resolved inside JointGoalStage today
        self._joint_goal_stage = JointGoalStage()
        self._reduced_joint_names = [str(name) for name in reduced_joint_names]
        self._joint_position_limits = {
            str(k): (None if v[0] is None else float(v[0]), None if v[1] is None else float(v[1]))
            for k, v in (joint_position_limits or {}).items()
        }
        self._smoothing_passes = int(max(0, smoothing_passes))
        self._max_projection_waypoints = int(max(0, max_projection_waypoints))

    def _reduced_q_from_dynamic_map(self, q_map: Mapping[str, float]) -> np.ndarray:
        return np.asarray(
            [float(q_map.get(name, 0.0)) for name in self._reduced_joint_names],
            dtype=float,
        )

    def _reduced_q_from_actuated_map(self, q_map: Mapping[str, float]) -> np.ndarray:
        return np.asarray(
            [float(q_map.get(name, 0.0)) for name in self._reduced_joint_names],
            dtype=float,
        )

    def _complete_dynamic_map(
        self,
        q_map: Mapping[str, float],
        q_seed: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        out = {str(name): float(q_map.get(name, 0.0)) for name in self._reduced_joint_names}
        cfg = self._joint_goal_stage.config
        if set(cfg.actuated_joints).issubset(out.keys()):
            completed = self._joint_goal_stage._steady_state.complete_from_actuated(
                {name: out[name] for name in cfg.actuated_joints},
                q_seed=q_seed if q_seed is not None else out,
            )
            if completed.success:
                out.update(completed.q_dynamic)
        for follower, leader in cfg.tied_joints.items():
            if leader in out:
                out[follower] = float(out[leader])
        return out

    def fk_world_pose(
        self,
        q_red: np.ndarray,
        q_seed: Mapping[str, float] | None = None,
    ) -> tuple[np.ndarray, float, dict[str, float]]:
        q = np.asarray(q_red, dtype=float).reshape(-1)
        q_map = self._complete_dynamic_map(
            {name: float(q[i]) for i, name in enumerate(self._reduced_joint_names)},
            q_seed=q_seed,
        )
        q_pin = self._joint_goal_stage._kin.pin.neutral(self._joint_goal_stage._kin.model)
        model = self._joint_goal_stage._kin.model
        for jid in range(1, model.njoints):
            name = str(model.names[jid])
            if name not in q_map:
                continue
            j = model.joints[jid]
            nq = int(j.nq)
            iq = int(j.idx_q)
            val = float(q_map[name])
            if nq == 1:
                q_pin[iq] = val
            elif nq == 2 and int(j.nv) == 1:
                q_pin[iq] = math.cos(val)
                q_pin[iq + 1] = math.sin(val)
        fk = self._joint_goal_stage._kin.forward_kinematics(
            q_pin,
            base_frame="world",
            end_frame=self._joint_goal_stage.config.target_frame,
        )
        T = np.asarray(fk["base_to_end"]["homogeneous"], dtype=float)
        xyz = np.asarray(T[:3, 3], dtype=float).reshape(3)
        yaw = float(math.atan2(T[1, 0], T[0, 0]))
        return xyz, yaw, q_map

    def _clip_to_joint_limits(self, q_waypoints: np.ndarray) -> np.ndarray:
        out = np.asarray(q_waypoints, dtype=float).copy()
        for j, name in enumerate(self._reduced_joint_names):
            lo, hi = self._joint_position_limits.get(name, (None, None))
            if lo is not None:
                out[:, j] = np.maximum(out[:, j], float(lo))
            if hi is not None:
                out[:, j] = np.minimum(out[:, j], float(hi))
        return out

    def _smooth_waypoints(self, q_waypoints: np.ndarray, locked_mask: np.ndarray | None = None) -> np.ndarray:
        out = np.asarray(q_waypoints, dtype=float).copy()
        if out.shape[0] <= 2:
            return out
        locked = np.zeros(out.shape[0], dtype=bool) if locked_mask is None else np.asarray(locked_mask, dtype=bool)
        for _ in range(self._smoothing_passes):
            prev = out.copy()
            smoothed = 0.25 * prev[:-2, :] + 0.5 * prev[1:-1, :] + 0.25 * prev[2:, :]
            mutable = ~locked[1:-1]
            out[1:-1, :][mutable, :] = smoothed[mutable, :]
            out = self._clip_to_joint_limits(out)
        return out

    @staticmethod
    def _select_anchor_indices(n_wp: int, max_interior: int) -> list[int]:
        if n_wp < 2:
            return [0]
        if n_wp <= 2:
            return [0, n_wp - 1]
        interior = list(range(1, n_wp - 1))
        if max_interior <= 0 or len(interior) <= max_interior:
            return [0, *interior, n_wp - 1]
        sampled = np.linspace(0, len(interior) - 1, max_interior, dtype=int)
        indices = sorted({0, n_wp - 1, *[interior[int(i)] for i in sampled]})
        return indices

    @staticmethod
    def _parameterize_waypoints(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if pts.shape[0] <= 1:
            return np.zeros(pts.shape[0], dtype=float)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)], dtype=float)
        if float(s[-1]) <= 1e-9:
            return np.linspace(0.0, 1.0, pts.shape[0], dtype=float)
        return s / float(s[-1])

    def _fit_joint_spline(
        self,
        *,
        u_anchor: np.ndarray,
        q_anchor: np.ndarray,
        u_query: np.ndarray,
    ) -> np.ndarray:
        u_anchor_arr = np.asarray(u_anchor, dtype=float).reshape(-1)
        q_anchor_arr = np.asarray(q_anchor, dtype=float)
        u_query_arr = np.asarray(u_query, dtype=float).reshape(-1)
        n_anchor, n_joint = q_anchor_arr.shape
        q_out = np.zeros((u_query_arr.size, n_joint), dtype=float)
        if n_anchor == 1:
            q_out[:, :] = q_anchor_arr[0, :]
            return q_out
        for j in range(n_joint):
            interp = PchipInterpolator(u_anchor_arr, q_anchor_arr[:, j], extrapolate=False)
            vals = np.asarray(interp(u_query_arr), dtype=float)
            vals[0] = float(q_anchor_arr[0, j])
            vals[-1] = float(q_anchor_arr[-1, j])
            q_out[:, j] = vals
        return q_out

    @staticmethod
    def _point_to_polyline_distance(point: np.ndarray, polyline: np.ndarray) -> float:
        p = np.asarray(point, dtype=float).reshape(3)
        line = np.asarray(polyline, dtype=float).reshape(-1, 3)
        if line.shape[0] == 0:
            return 0.0
        if line.shape[0] == 1:
            return float(np.linalg.norm(p - line[0]))
        best = float("inf")
        for a, b in zip(line[:-1], line[1:]):
            ab = b - a
            denom = float(np.dot(ab, ab))
            if denom <= 1e-12:
                cand = float(np.linalg.norm(p - a))
            else:
                t = float(np.dot(p - a, ab) / denom)
                t = max(0.0, min(1.0, t))
                proj = a + t * ab
                cand = float(np.linalg.norm(p - proj))
            best = min(best, cand)
        return best

    def _evaluate_cartesian_tracking(
        self,
        q_waypoints: np.ndarray,
        reference_xyz: np.ndarray,
        reference_yaw: np.ndarray,
        anchor_xyz: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        xyz_out = []
        yaw_out = []
        xyz_err = 0.0
        yaw_err = 0.0
        pos_errs = []
        yaw_errs = []
        polyline_errs = []
        q_seed_map = {
            str(name): float(q_waypoints[0, i]) for i, name in enumerate(self._reduced_joint_names)
        }
        for i, q in enumerate(q_waypoints):
            xyz_i, yaw_i, q_seed_map = self.fk_world_pose(q, q_seed=q_seed_map)
            xyz_out.append(xyz_i)
            yaw_out.append(yaw_i)
            xyz_delta = xyz_i - reference_xyz[i, :]
            pos_err = float(np.linalg.norm(xyz_delta))
            pos_errs.append(pos_err)
            xyz_err += float(np.dot(xyz_delta, xyz_delta))
            yaw_delta = _wrap_to_pi(yaw_i - float(reference_yaw[i]))
            yaw_abs = float(abs(yaw_delta))
            yaw_errs.append(yaw_abs)
            yaw_err += float(yaw_delta * yaw_delta)
            if anchor_xyz is not None:
                polyline_errs.append(self._point_to_polyline_distance(xyz_i, anchor_xyz))
        pos_arr = np.asarray(pos_errs, dtype=float)
        yaw_arr = np.asarray(yaw_errs, dtype=float)
        polyline_arr = np.asarray(polyline_errs, dtype=float) if polyline_errs else np.zeros(0, dtype=float)
        return (
            np.asarray(xyz_out, dtype=float),
            np.asarray(yaw_out, dtype=float),
            {
                "cartesian_position_cost": float(xyz_err),
                "cartesian_yaw_cost": float(yaw_err),
                "max_position_error_m": float(np.max(pos_arr)),
                "mean_position_error_m": float(np.mean(pos_arr)),
                "final_position_error_m": float(pos_arr[-1]),
                "max_yaw_error_deg": float(np.degrees(np.max(yaw_arr))),
                "mean_yaw_error_deg": float(np.degrees(np.mean(yaw_arr))),
                "final_yaw_error_deg": float(np.degrees(yaw_arr[-1])),
                "max_anchor_polyline_deviation_m": float(np.max(polyline_arr)) if polyline_arr.size else 0.0,
                "mean_anchor_polyline_deviation_m": float(np.mean(polyline_arr)) if polyline_arr.size else 0.0,
            },
        )

    def plan(
        self,
        *,
        q_start: Sequence[float],
        q_goal: Sequence[float],
        reference_xyz: Sequence[Sequence[float]],
        reference_yaw: Sequence[float],
    ) -> JointSpacePlanResult:
        q_start_arr = np.asarray(q_start, dtype=float).reshape(-1)
        q_goal_arr = np.asarray(q_goal, dtype=float).reshape(-1)
        reference_xyz_arr = np.asarray(reference_xyz, dtype=float).reshape(-1, 3)
        reference_yaw_arr = np.asarray(reference_yaw, dtype=float).reshape(-1)

        if q_start_arr.shape != q_goal_arr.shape:
            return JointSpacePlanResult(
                success=False,
                message="joint-space planner requires matching start/goal dimensions",
                q_waypoints=np.zeros((0, 0), dtype=float),
                xyz_waypoints=np.zeros((0, 3), dtype=float),
                yaw_waypoints=np.zeros(0, dtype=float),
                diagnostics={},
            )
        if reference_xyz_arr.shape[0] != reference_yaw_arr.shape[0] or reference_xyz_arr.shape[0] < 2:
            return JointSpacePlanResult(
                success=False,
                message="joint-space planner requires at least two Cartesian reference waypoints",
                q_waypoints=np.zeros((0, q_start_arr.shape[0]), dtype=float),
                xyz_waypoints=np.zeros((0, 3), dtype=float),
                yaw_waypoints=np.zeros(0, dtype=float),
                diagnostics={},
            )

        n_wp = int(reference_xyz_arr.shape[0])
        alpha = np.linspace(0.0, 1.0, n_wp, dtype=float).reshape(-1, 1)
        q_initial = (1.0 - alpha) * q_start_arr.reshape(1, -1) + alpha * q_goal_arr.reshape(1, -1)

        anchor_indices = self._select_anchor_indices(n_wp, self._max_projection_waypoints)
        anchor_xyz = []
        anchor_yaw = []
        q_anchor = []
        seed_map = {name: float(q_start_arr[i]) for i, name in enumerate(self._reduced_joint_names)}
        solved_anchor_count = 0
        dropped_anchor_count = 0

        for idx, i in enumerate(anchor_indices):
            is_endpoint = i == 0 or i == (n_wp - 1)
            if i == 0:
                q_anchor.append(q_start_arr.copy())
                anchor_xyz.append(reference_xyz_arr[i, :].copy())
                anchor_yaw.append(float(reference_yaw_arr[i]))
                solved_anchor_count += 1
                continue
            if i == (n_wp - 1):
                q_anchor.append(q_goal_arr.copy())
                anchor_xyz.append(reference_xyz_arr[i, :].copy())
                anchor_yaw.append(float(reference_yaw_arr[i]))
                solved_anchor_count += 1
                continue
            solve = self._joint_goal_stage.solve_world_pose(
                goal_world=reference_xyz_arr[i, :],
                target_yaw_rad=float(reference_yaw_arr[i]),
                q_seed=seed_map,
            )
            if solve.success:
                q_anchor.append(self._reduced_q_from_actuated_map(solve.q_actuated))
                anchor_xyz.append(reference_xyz_arr[i, :].copy())
                anchor_yaw.append(float(reference_yaw_arr[i]))
                seed_map = dict(solve.q_dynamic)
                solved_anchor_count += 1
            elif is_endpoint:
                return JointSpacePlanResult(
                    success=False,
                    message=f"anchor solve failed at endpoint index {i}: {solve.message}",
                    q_waypoints=np.zeros((0, q_start_arr.shape[0]), dtype=float),
                    xyz_waypoints=np.zeros((0, 3), dtype=float),
                    yaw_waypoints=np.zeros(0, dtype=float),
                    diagnostics={},
                )
            else:
                dropped_anchor_count += 1

        q_anchor_arr = np.asarray(q_anchor, dtype=float)
        anchor_xyz_arr = np.asarray(anchor_xyz, dtype=float).reshape(-1, 3)
        if q_anchor_arr.shape[0] < 2:
            return JointSpacePlanResult(
                success=False,
                message=(
                    "anchor solve failed: need at least start and goal anchors; "
                    f"got {q_anchor_arr.shape[0]} solved anchors"
                ),
                q_waypoints=np.zeros((0, q_start_arr.shape[0]), dtype=float),
                xyz_waypoints=np.zeros((0, 3), dtype=float),
                yaw_waypoints=np.zeros(0, dtype=float),
                diagnostics={},
            )

        u_reference = self._parameterize_waypoints(reference_xyz_arr)
        u_anchor = self._parameterize_waypoints(anchor_xyz_arr)
        q_waypoints = self._fit_joint_spline(
            u_anchor=u_anchor,
            q_anchor=q_anchor_arr,
            u_query=u_reference,
        )
        q_waypoints = self._clip_to_joint_limits(q_waypoints)
        q_waypoints[0, :] = q_start_arr
        q_waypoints[-1, :] = q_goal_arr
        if self._smoothing_passes > 0:
            q_waypoints = self._smooth_waypoints(q_waypoints)

        initial_xyz_out, initial_yaw_out, initial_diag = self._evaluate_cartesian_tracking(
            q_initial,
            reference_xyz_arr,
            reference_yaw_arr,
            anchor_xyz=anchor_xyz_arr,
        )
        candidate_xyz_out, candidate_yaw_out, candidate_diag = self._evaluate_cartesian_tracking(
            q_waypoints,
            reference_xyz_arr,
            reference_yaw_arr,
            anchor_xyz=anchor_xyz_arr,
        )
        initial_cost = float(
            initial_diag["cartesian_position_cost"] + initial_diag["cartesian_yaw_cost"]
        )
        candidate_cost = float(
            candidate_diag["cartesian_position_cost"] + candidate_diag["cartesian_yaw_cost"]
        )
        if candidate_cost <= initial_cost + 1e-9:
            xyz_out = candidate_xyz_out
            yaw_out = candidate_yaw_out
            final_diag = candidate_diag
            final_cost = candidate_cost
        else:
            q_waypoints = q_initial
            xyz_out = initial_xyz_out
            yaw_out = initial_yaw_out
            final_diag = initial_diag
            final_cost = initial_cost

        fallback_used = bool(q_anchor_arr.shape[0] == 2)
        message = (
            "joint-space anchor spline ANCHOR_V1"
            f" (solved {solved_anchor_count}/{len(anchor_indices)} anchors, "
            f"dropped {dropped_anchor_count} interior anchors"
            + (", direct start-goal fallback" if fallback_used else "")
            + ")"
        )
        return JointSpacePlanResult(
            success=True,
            message=message,
            q_waypoints=q_waypoints,
            xyz_waypoints=xyz_out,
            yaw_waypoints=yaw_out,
            diagnostics={
                **final_diag,
                "initial_cartesian_position_cost": float(initial_diag["cartesian_position_cost"]),
                "initial_cartesian_yaw_cost": float(initial_diag["cartesian_yaw_cost"]),
                "objective_initial": float(initial_cost),
                "objective_final": float(final_cost),
                "anchor_count": float(len(anchor_indices)),
                "solved_anchor_count": float(solved_anchor_count),
                "dropped_anchor_count": float(dropped_anchor_count),
                "direct_anchor_fallback": float(1.0 if fallback_used else 0.0),
                "waypoint_count": float(n_wp),
            },
        )
