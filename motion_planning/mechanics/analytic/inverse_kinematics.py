from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from .config import AnalyticModelConfig
from .crane_geometry import DEFAULT_CRANE_GEOMETRY, CraneGeometryConstants
from .model_description import ModelDescription
from .pinocchio_utils import fk_homogeneous, joint_bounds


def _rotvec_from_R(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    tr = float(np.trace(R))
    c = max(-1.0, min(1.0, 0.5 * (tr - 1.0)))
    theta = float(np.arccos(c))
    if theta < 1e-9:
        return np.zeros(3, dtype=float)
    s = np.sin(theta)
    if abs(s) < 1e-10:
        A = 0.5 * (R + np.eye(3))
        axis = np.array(
            [
                np.sqrt(max(0.0, A[0, 0])),
                np.sqrt(max(0.0, A[1, 1])),
                np.sqrt(max(0.0, A[2, 2])),
            ],
            dtype=float,
        )
        if R[2, 1] - R[1, 2] < 0:
            axis[0] = -axis[0]
        if R[0, 2] - R[2, 0] < 0:
            axis[1] = -axis[1]
        if R[1, 0] - R[0, 1] < 0:
            axis[2] = -axis[2]
        n = np.linalg.norm(axis)
        if n < 1e-10:
            return np.zeros(3, dtype=float)
        return theta * axis / n
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=float)
    return 0.5 * theta / s * w


@dataclass
class IkSolveResult:
    success: bool
    status: str
    message: str
    q_dynamic: Dict[str, float]
    q_actuated: Dict[str, float]
    q_passive: Dict[str, float]
    iterations: int
    cost: float
    pos_error_m: float
    rot_error_rad: float


class _IKBase:
    def __init__(self, desc: ModelDescription, config: AnalyticModelConfig) -> None:
        self._desc = desc
        self._cfg = config
        import pinocchio as pin

        self._pin = pin
        self._pin_model = desc.model
        self._pin_data = self._pin_model.createData()
        self._frame_cache: dict[str, int] = {}

    @staticmethod
    def _wrap_angle(x: float) -> float:
        return float(np.arctan2(np.sin(x), np.cos(x)))

    @staticmethod
    def _midrange_cost(q: Mapping[str, float], bounds: Mapping[str, tuple[float, float]]) -> float:
        vals: list[float] = []
        for jn in ("theta2_boom_joint", "theta3_arm_joint", "q4_big_telescope"):
            if jn not in q or jn not in bounds:
                continue
            lo, hi = bounds[jn]
            if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                continue
            vals.append((float(q[jn]) - lo) / (hi - lo))
        if not vals:
            return 1e6
        v = np.asarray(vals, dtype=float)
        return float(np.linalg.norm(v - 0.5 * np.ones_like(v)))

    def _fk(self, q_values: Mapping[str, float], *, base_frame: str, end_frame: str) -> np.ndarray:
        return fk_homogeneous(
            pin_model=self._pin_model,
            pin_data=self._pin_data,
            pin_module=self._pin,
            q_values=q_values,
            base_frame=base_frame,
            end_frame=end_frame,
            frame_cache=self._frame_cache,
        )

    def _joint_bounds(self, joint_name: str) -> tuple[float, float]:
        lo, hi = joint_bounds(self._pin_model, joint_name)
        ov = self._cfg.joint_position_overrides.get(joint_name)
        if ov is None:
            return lo, hi
        lo_ov, hi_ov = ov
        if lo_ov is not None:
            lo = max(lo, float(lo_ov)) if np.isfinite(lo) else float(lo_ov)
        if hi_ov is not None:
            hi = min(hi, float(hi_ov)) if np.isfinite(hi) else float(hi_ov)
        return float(lo), float(hi)


class AnalyticIKSolver(_IKBase):
    """Analytic IK: fixed joints + 1D independent-joint search."""

    def __init__(self, desc: ModelDescription, config: AnalyticModelConfig, geometry: CraneGeometryConstants) -> None:
        super().__init__(desc, config)
        self._geometry = geometry
        self._base_frame = "K0_mounting_base"
        self._end_frame = "K8_tool_center_point"

    def solve(
        self,
        *,
        target_T_base_to_end: np.ndarray,
        base_frame: str,
        end_frame: str,
        seed: dict[str, float],
        act_names: list[str],
        fixed: dict[str, float],
    ) -> IkSolveResult | None:
        if base_frame != self._base_frame or end_frame != self._end_frame:
            return None

        required_dynamic = {
            "theta1_slewing_joint",
            "theta2_boom_joint",
            "theta3_arm_joint",
            "q4_big_telescope",
            "theta6_tip_joint",
            "theta7_tilt_joint",
            "theta8_rotator_joint",
        }
        required_actuated = {
            "theta1_slewing_joint",
            "theta2_boom_joint",
            "theta3_arm_joint",
            "q4_big_telescope",
            "theta8_rotator_joint",
        }
        if not required_dynamic.issubset(set(self._cfg.dynamic_joints)):
            return None
        if not required_actuated.issubset(set(act_names)):
            return None

        p_target = np.asarray(target_T_base_to_end[:3, 3], dtype=float)
        R_target = np.asarray(target_T_base_to_end[:3, :3], dtype=float)
        x_t, y_t, z_t = float(p_target[0]), float(p_target[1]), float(p_target[2])
        phi_tool = float(np.arctan2(R_target[1, 0], R_target[0, 0]))

        bounds = {
            "theta1_slewing_joint": self._joint_bounds("theta1_slewing_joint"),
            "theta2_boom_joint": self._joint_bounds("theta2_boom_joint"),
            "theta3_arm_joint": self._joint_bounds("theta3_arm_joint"),
            "q4_big_telescope": self._joint_bounds("q4_big_telescope"),
            "theta8_rotator_joint": self._joint_bounds("theta8_rotator_joint"),
        }
        lo1, hi1 = bounds["theta1_slewing_joint"]
        lo2, hi2 = bounds["theta2_boom_joint"]
        lo3, hi3 = bounds["theta3_arm_joint"]
        lo4, hi4 = bounds["q4_big_telescope"]
        lo8, hi8 = bounds["theta8_rotator_joint"]
        if not (np.isfinite(lo2) and np.isfinite(hi2) and hi2 > lo2):
            return None

        q0_1 = float(seed.get("theta1_slewing_joint", 0.0))
        q0_8 = float(seed.get("theta8_rotator_joint", 0.0))
        q6 = float(fixed.get("theta6_tip_joint", seed.get("theta6_tip_joint", 0.0)))
        q7 = float(fixed.get("theta7_tilt_joint", seed.get("theta7_tilt_joint", 0.0)))

        if "theta1_slewing_joint" in fixed:
            theta1 = float(fixed["theta1_slewing_joint"])
        else:
            theta1 = float(np.arctan2(y_t, x_t))
            if abs(q0_1 - theta1) > np.pi:
                if theta1 > 0:
                    theta1_new = theta1 - 2.0 * np.pi
                    if theta1_new > lo1:
                        theta1 = theta1_new
                else:
                    theta1_new = theta1 + 2.0 * np.pi
                    if theta1_new < hi1:
                        theta1 = theta1_new
        if (np.isfinite(lo1) and theta1 < lo1 - 1e-9) or (np.isfinite(hi1) and theta1 > hi1 + 1e-9):
            return None

        if "theta8_rotator_joint" in fixed:
            theta8 = float(fixed["theta8_rotator_joint"])
        else:
            theta8 = self._wrap_angle(theta1 - phi_tool)
            if abs(q0_8 - theta8) > np.pi:
                theta8 -= 2.0 * np.pi * float(np.sign(theta8 - q0_8))
        if (np.isfinite(lo8) and theta8 < lo8 - 1e-9) or (np.isfinite(hi8) and theta8 > hi8 + 1e-9):
            return None

        g = self._geometry
        p2 = g.p2
        p5 = np.array([float(np.hypot(x_t, y_t)), z_t], dtype=float)

        best: dict[str, float] | None = None
        best_pos_err = float("inf")
        best_rot_err = float("inf")
        best_mid = float("inf")
        iters = 0
        step = np.pi / 180.0

        q2_values = [float(fixed["theta2_boom_joint"])] if "theta2_boom_joint" in fixed else np.arange(lo2, hi2 + 0.5 * step, step, dtype=float).tolist()
        for q2 in q2_values:
            iters += 1
            if (np.isfinite(lo2) and q2 < lo2 - 1e-9) or (np.isfinite(hi2) and q2 > hi2 + 1e-9):
                continue
            p3 = np.array([g.a2 * np.cos(q2) - g.a1, g.d1 + g.a2 * np.sin(q2)], dtype=float)
            p32 = p2 - p3
            p35 = p5 - p3
            p32_norm = float(np.linalg.norm(p32))
            p35_norm = float(np.linalg.norm(p35))
            if p32_norm < 1e-12 or p35_norm < g.a3:
                continue

            d45 = float(np.sqrt(max(0.0, p35_norm * p35_norm - g.a3 * g.a3)))
            denom = p35_norm * p32_norm
            if denom < 1e-12:
                continue
            gamma = float(np.arccos(np.clip(float(np.dot(p35, p32)) / denom, -1.0, 1.0)))

            theta3 = float(gamma - 0.5 * np.pi + np.arctan2(g.a3, d45))
            q4 = float(0.5 * (d45 - g.d4))
            if "theta3_arm_joint" in fixed:
                theta3 = float(fixed["theta3_arm_joint"])
            if "q4_big_telescope" in fixed:
                q4 = float(fixed["q4_big_telescope"])
            if (np.isfinite(lo3) and theta3 < lo3 - 1e-6) or (np.isfinite(hi3) and theta3 > hi3 + 1e-6):
                continue
            if theta3 > g.theta3_max:
                continue
            if (np.isfinite(lo4) and q4 < lo4 - 1e-6) or (np.isfinite(hi4) and q4 > hi4 + 1e-6):
                continue

            q_try = dict(seed)
            q_try.update(
                {
                    "theta1_slewing_joint": float(theta1),
                    "theta2_boom_joint": float(q2),
                    "theta3_arm_joint": float(theta3),
                    "q4_big_telescope": float(q4),
                    "theta6_tip_joint": float(q6),
                    "theta7_tilt_joint": float(q7),
                    "theta8_rotator_joint": float(theta8),
                }
            )
            for jn, val in fixed.items():
                q_try[jn] = float(val)
            for follower, leader in self._cfg.tied_joints.items():
                if leader in q_try:
                    q_try[follower] = float(q_try[leader])

            T_try = self._fk(q_try, base_frame=base_frame, end_frame=end_frame)
            pos_err = float(np.linalg.norm(T_try[:3, 3] - p_target))
            rot_err = float(np.linalg.norm(_rotvec_from_R(R_target.T @ T_try[:3, :3])))
            mid = self._midrange_cost(q_try, bounds)

            pose_ok = pos_err <= 5e-3 and rot_err <= 5e-2
            if pose_ok:
                if (mid < best_mid - 1e-12) or (abs(mid - best_mid) <= 1e-12 and pos_err < best_pos_err):
                    best = q_try
                    best_mid = mid
                    best_pos_err = pos_err
                    best_rot_err = rot_err
            elif best is None:
                score_old = best_pos_err + 0.3 * best_rot_err
                score_new = pos_err + 0.3 * rot_err
                if score_new < score_old:
                    best = q_try
                    best_pos_err = pos_err
                    best_rot_err = rot_err

        if best is None or best_pos_err > 5e-3 or best_rot_err > 5e-2:
            return None
        q_act = {jn: float(best.get(jn, 0.0)) for jn in act_names}
        q_pas = {jn: float(best.get(jn, 0.0)) for jn in self._cfg.passive_joints if jn in best}
        return IkSolveResult(
            success=True,
            status="analytic_success",
            message="IK analytic solve converged (1D search)",
            q_dynamic={jn: float(best.get(jn, 0.0)) for jn in self._cfg.dynamic_joints},
            q_actuated=q_act,
            q_passive=q_pas,
            iterations=int(iters),
            cost=float(0.5 * (best_pos_err * best_pos_err + best_rot_err * best_rot_err)),
            pos_error_m=best_pos_err,
            rot_error_rad=best_rot_err,
        )


class NumericIKSolver(_IKBase):
    def solve(
        self,
        *,
        target_T_base_to_end: np.ndarray,
        base_frame: str,
        end_frame: str,
        seed: dict[str, float],
        act_names: list[str],
        fixed: dict[str, float],
        pos_weight: float,
        rot_weight: float,
        reg_seed_weight: float,
        max_nfev: int,
        ftol: float,
        xtol: float,
        gtol: float,
    ) -> IkSolveResult:
        T_target = np.asarray(target_T_base_to_end, dtype=float).reshape(4, 4)
        p_target = T_target[:3, 3].copy()
        R_target = T_target[:3, :3].copy()
        dynamic_names = list(self._cfg.dynamic_joints)
        q_opt_names = [jn for jn in act_names if jn not in fixed]

        q_lb = []
        q_ub = []
        x0 = []
        for jn in q_opt_names:
            lo, hi = self._joint_bounds(jn)
            q_lb.append(lo)
            q_ub.append(hi)
            x0.append(float(seed.get(jn, 0.0)))
        x0_arr = np.asarray(x0, dtype=float)
        lb_arr = np.asarray(q_lb, dtype=float)
        ub_arr = np.asarray(q_ub, dtype=float)
        x0_arr = np.clip(x0_arr, lb_arr, ub_arr)

        def build_full_q(x: np.ndarray) -> dict[str, float]:
            q = dict(seed)
            for jn, v in zip(q_opt_names, x):
                q[jn] = float(v)
            for jn, v in fixed.items():
                q[jn] = float(v)
            for follower, leader in self._cfg.tied_joints.items():
                if leader in q:
                    q[follower] = float(q[leader])
            return q

        def residual(x: np.ndarray) -> np.ndarray:
            q = build_full_q(x)
            T_cur = self._fk(q, base_frame=base_frame, end_frame=end_frame)
            p_cur = T_cur[:3, 3]
            R_cur = T_cur[:3, :3]
            r_pos = pos_weight * (p_cur - p_target)
            r_rot = rot_weight * _rotvec_from_R(R_target.T @ R_cur)
            r_reg = np.sqrt(reg_seed_weight) * (x - x0_arr)
            return np.concatenate([r_pos, r_rot, r_reg])

        opt = least_squares(
            residual,
            x0_arr,
            bounds=(lb_arr, ub_arr),
            max_nfev=max_nfev,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
        )
        q_sol = build_full_q(opt.x)
        T_fin = self._fk(q_sol, base_frame=base_frame, end_frame=end_frame)
        p_fin = T_fin[:3, 3]
        R_fin = T_fin[:3, :3]
        pos_err = float(np.linalg.norm(p_fin - p_target))
        rot_err = float(np.linalg.norm(_rotvec_from_R(R_target.T @ R_fin)))

        success = bool(opt.success) or (pos_err <= 1e-5 and rot_err <= 1e-5)
        status = "numeric_success" if bool(opt.success) else ("numeric_residual_success" if success else "failed")
        msg = "IK solve converged (numeric fallback)" if success else f"IK solve failed: {opt.message}"
        q_act = {jn: float(q_sol.get(jn, 0.0)) for jn in act_names}
        q_pas = {jn: float(q_sol.get(jn, 0.0)) for jn in self._cfg.passive_joints if jn in q_sol}
        return IkSolveResult(
            success=success,
            status=status,
            message=msg,
            q_dynamic={jn: float(q_sol.get(jn, 0.0)) for jn in dynamic_names},
            q_actuated=q_act,
            q_passive=q_pas,
            iterations=int(opt.nfev),
            cost=float(opt.cost),
            pos_error_m=pos_err,
            rot_error_rad=rot_err,
        )


class AnalyticInverseKinematics:
    """Facade selecting analytic IK first, then numeric fallback."""

    def __init__(self, desc: ModelDescription, config: AnalyticModelConfig) -> None:
        self._desc = desc
        self._cfg = config
        self._analytic = AnalyticIKSolver(desc, config, DEFAULT_CRANE_GEOMETRY)
        self._numeric = NumericIKSolver(desc, config)

    def solve_pose(
        self,
        *,
        target_T_base_to_end: np.ndarray,
        base_frame: str = "K0_mounting_base",
        end_frame: str = "K8_tool_center_point",
        q_seed: Mapping[str, float] | None = None,
        actuated_joint_names: Sequence[str] | None = None,
        fixed_joints: Mapping[str, float] | None = None,
        pos_weight: float = 1.0,
        rot_weight: float = 1.0,
        reg_seed_weight: float = 1e-6,
        max_nfev: int = 200,
        ftol: float = 1e-9,
        xtol: float = 1e-9,
        gtol: float = 1e-9,
        force_analytic: bool = False,
    ) -> IkSolveResult:
        T_target = np.asarray(target_T_base_to_end, dtype=float).reshape(4, 4)
        dynamic_names = list(self._cfg.dynamic_joints)
        act_default = list(self._cfg.actuated_joints)
        act_names = list(actuated_joint_names) if actuated_joint_names is not None else act_default
        fixed = {str(k): float(v) for k, v in (fixed_joints or {}).items() if str(k) in dynamic_names}

        seed = {jn: 0.0 for jn in dynamic_names}
        if q_seed is not None:
            for jn in dynamic_names:
                if jn in q_seed:
                    seed[jn] = float(q_seed[jn])
        for jn, v in fixed.items():
            seed[jn] = float(v)

        analytic_res = self._analytic.solve(
            target_T_base_to_end=T_target,
            base_frame=base_frame,
            end_frame=end_frame,
            seed=seed,
            act_names=act_names,
            fixed=fixed,
        )
        if analytic_res is not None:
            return analytic_res
        if force_analytic:
            q_fail = dict(seed)
            for jn, val in fixed.items():
                q_fail[jn] = float(val)
            for follower, leader in self._cfg.tied_joints.items():
                if leader in q_fail:
                    q_fail[follower] = float(q_fail[leader])
            q_act = {jn: float(q_fail.get(jn, 0.0)) for jn in act_names}
            q_pas = {jn: float(q_fail.get(jn, 0.0)) for jn in self._cfg.passive_joints if jn in q_fail}
            return IkSolveResult(
                success=False,
                status="analytic_failed",
                message="IK analytic solve failed and numeric fallback disabled (force_analytic=True).",
                q_dynamic={jn: float(q_fail.get(jn, 0.0)) for jn in dynamic_names},
                q_actuated=q_act,
                q_passive=q_pas,
                iterations=0,
                cost=float("inf"),
                pos_error_m=float("inf"),
                rot_error_rad=float("inf"),
            )

        return self._numeric.solve(
            target_T_base_to_end=T_target,
            base_frame=base_frame,
            end_frame=end_frame,
            seed=seed,
            act_names=act_names,
            fixed=fixed,
            pos_weight=pos_weight,
            rot_weight=rot_weight,
            reg_seed_weight=reg_seed_weight,
            max_nfev=max_nfev,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
        )
