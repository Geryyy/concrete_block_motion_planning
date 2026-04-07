from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares

from .config import AnalyticModelConfig
from .inverse_kinematics import AnalyticInverseKinematics, IkSolveResult
from .model_description import ModelDescription
from .pinocchio_utils import q_map_to_pin_q
from .pose_conventions import phi_tool_from_transform, pose_from_pos_yaw
from .reference_states import merge_reference_seed

@dataclass
class SteadyStateResult:
    success: bool
    message: str
    q_actuated: Dict[str, float]
    q_passive: Dict[str, float]
    q_dynamic: Dict[str, float]
    ik_result: IkSolveResult
    passive_residual: float
    fk_position_error_m: float
    fk_yaw_error_rad: float
    fk_xyz: np.ndarray
    fk_yaw_rad: float

class CraneSteadyState:
    _DEFAULT_BASE = "K0_mounting_base"
    _DEFAULT_END = "K8_tool_center_point"

    def __init__(
        self,
        desc: ModelDescription,
        config: AnalyticModelConfig,
        *,
        ik_max_nfev: int = 1000,
        passive_tol: float = 1e-10,
        passive_residual_tol: float = 1e-6,
        fk_position_tol_m: float = 2e-2,
        fk_yaw_tol_rad: float = 5e-2,
        passive_max_iter: int = 200,
        base_frame: str = _DEFAULT_BASE,
        end_frame: str = _DEFAULT_END,
    ) -> None:
        self._desc = desc
        self._cfg = config
        self._base_frame = base_frame
        self._end_frame = end_frame
        self._passive_tol = passive_tol
        self._passive_residual_tol = passive_residual_tol
        self._fk_position_tol_m = float(fk_position_tol_m)
        self._fk_yaw_tol_rad = float(fk_yaw_tol_rad)
        self._passive_max_iter = passive_max_iter
        self._ik = AnalyticInverseKinematics(desc, config)
        self._ik_max_nfev = ik_max_nfev

        self._pin_model: pin.Model = desc.model
        self._pin_data = self._pin_model.createData()

        self._pas_names: list[str] = list(config.passive_joints)
        self._pas_v_idx: list[int] = []
        self._pas_q0: list[float] = []
        self._pas_lb: list[float] = []
        self._pas_ub: list[float] = []
        for jname in self._pas_names:
            jid = int(self._pin_model.getJointId(jname))
            j = self._pin_model.joints[jid]
            vi = int(j.idx_v)
            iq = int(j.idx_q)
            self._pas_v_idx.append(vi)
            lo = float(self._pin_model.lowerPositionLimit[iq])
            hi = float(self._pin_model.upperPositionLimit[iq])
            ov = self._cfg.joint_position_overrides.get(jname)
            if ov is not None:
                lo_ov, hi_ov = ov
                if lo_ov is not None:
                    lo = max(lo, float(lo_ov)) if np.isfinite(lo) else float(lo_ov)
                if hi_ov is not None:
                    hi = min(hi, float(hi_ov)) if np.isfinite(hi) else float(hi_ov)
            self._pas_lb.append(lo)
            self._pas_ub.append(hi)
            self._pas_q0.append(0.5 * (lo + hi) if (np.isfinite(lo) and np.isfinite(hi)) else 0.0)

        self._eq_params = self._extract_equilibrium_params()

    def _extract_equilibrium_params(self) -> dict[str, float]:
        model = self._pin_model
        j6_id = int(model.getJointId("theta6_tip_joint"))
        j7_id = int(model.getJointId("theta7_tilt_joint"))
        j8_id = int(model.getJointId("theta8_rotator_joint"))
        i6 = model.inertias[j6_id]
        i7 = model.inertias[j7_id]
        i8 = model.inertias[j8_id]
        data = model.createData()
        q_zero = pin.neutral(model)
        pin.forwardKinematics(model, data, q_zero)
        pin.updateFramePlacements(model, data)
        k5_fid = model.getFrameId("K5_inner_telescope")
        k6_fid = model.getFrameId("K6_double_joint_link")
        k8_fid = model.getFrameId("K8_tool_center_point")
        k5_pos = data.oMf[k5_fid].translation
        k6_pos = data.oMf[k6_fid].translation
        k8_pos = data.oMf[k8_fid].translation
        a6 = float(np.linalg.norm(k6_pos - k5_pos))
        d8 = float(np.linalg.norm(k8_pos - k6_pos))
        return {
            "m6": float(i6.mass), "s6x": float(i6.lever[0]), "s6y": float(i6.lever[1]), "s6z": float(i6.lever[2]),
            "m7": float(i7.mass), "s7x": float(i7.lever[0]), "s7y": float(i7.lever[1]), "s7z": float(i7.lever[2]),
            "m8": float(i8.mass), "s8x": float(i8.lever[0]), "s8y": float(i8.lever[1]), "s8z": float(i8.lever[2]),
            "a6": a6, "d8": d8,
        }

    def analytic_equilibrium(
        self,
        theta2: float,
        theta3: float,
        theta8: float,
    ) -> tuple[float, float]:
        p = self._eq_params
        m6, m7, m8 = p["m6"], p["m7"], p["m8"]
        s6x, s6z = p["s6x"], p["s6z"]
        s7x, s7y, s7z = p["s7x"], p["s7y"], p["s7z"]
        s8x, s8y, s8z = p["s8x"], p["s8y"], p["s8z"]
        a6, d8 = p["a6"], p["d8"]

        theta7 = -np.arctan2(
            d8 * m8 + s7z * m7 + s8z * m8,
            np.cos(theta8) * m8 * s8x - np.sin(theta8) * m8 * s8y + m7 * s7x,
        ) + np.pi

        c2, s2 = np.cos(theta2), np.sin(theta2)
        c3, s3 = np.cos(theta3), np.sin(theta3)
        c7, s7 = np.cos(theta7), np.sin(theta7)
        c8, s8 = np.cos(theta8), np.sin(theta8)

        t3 = s2 * c3
        t14 = s3 * c2
        t17 = s3 * s2
        t32 = c2 * c3

        t6 = s8y * m8 * c8
        t10 = s8x * m8 * s8
        t20 = m8 * d8 * s7
        t23 = s7z * m7 * s7
        t26 = s8z * m8 * s7
        t30 = s7x * m7 * c7
        t39 = s8x * m8 * c7
        t42 = s8y * m8 * s8

        num_a = (t42 * c7 * t32 - t39 * c8 * t32 - t10 * t14 - t10 * t3
                 - t6 * t14 - t20 * t17 - t23 * t17 - t26 * t17
                 + t30 * t17 + t20 * t32 + t23 * t32 + t26 * t32
                 - t6 * t3 - t30 * t32)

        t49 = a6 * m6
        t51 = a6 * m7
        t53 = m8 * a6
        t55 = m6 * s6x
        t57 = m6 * s6z
        t59 = s7y * m7

        num_b = (-t42 * c7 * t17 + t39 * c8 * t17 + t57 * t14 - t59 * t14
                 + t49 * t17 + t51 * t17 + t53 * t17 + t55 * t17
                 + t57 * t3 - t59 * t3 - t49 * t32 - t51 * t32
                 - t53 * t32 - t55 * t32)

        t70_c12 = m8 * c2
        t71 = s8y * t70_c12
        t74_s12 = m8 * s2
        t75 = s8y * t74_s12
        t78 = s8x * t70_c12
        t81 = s8x * t74_s12
        t84 = m7 * s2
        t88 = m7 * c2
        t91 = c3 * s7
        t99 = s3 * s7
        t107 = c7 * c8
        t113 = c7 * s8

        den_a = (m8 * d8 * c2 * t99 + m8 * d8 * s2 * t91
                 - s7x * t84 * c3 * c7 - s7x * t88 * s3 * c7
                 + s7z * t84 * t91 + s7z * t88 * t99
                 + s8z * t70_c12 * t99 + s8z * t74_s12 * t91
                 - t81 * c3 * t107 + t71 * c3 * c8 + t78 * c3 * s8
                 - t78 * s3 * t107 - t75 * s3 * c8 - t81 * s3 * s8)

        den_b = (t75 * c3 * t113 + t71 * s3 * t113
                 - t49 * t14 - t51 * t14 - t53 * t14 - t55 * t14
                 + t57 * t17 - t59 * t17 - t49 * t3 - t51 * t3
                 - t53 * t3 - t55 * t3 - t57 * t32 + t59 * t32)

        denom = den_a + den_b
        if abs(denom) < 1e-15:
            theta6 = 0.0
        else:
            theta6 = float(np.arctan((num_a + num_b) / denom))

        return float(theta6), float(theta7)

    @classmethod
    def default(
        cls,
        *,
        ik_max_nfev: int = 1000,
        passive_tol: float = 1e-10,
        passive_residual_tol: float = 1e-6,
        fk_position_tol_m: float = 2e-2,
        fk_yaw_tol_rad: float = 5e-2,
        passive_max_iter: int = 200,
    ) -> "CraneSteadyState":
        cfg = AnalyticModelConfig.default()
        desc = ModelDescription(cfg)
        return cls(
            desc,
            cfg,
            ik_max_nfev=ik_max_nfev,
            passive_tol=passive_tol,
            passive_residual_tol=passive_residual_tol,
            fk_position_tol_m=fk_position_tol_m,
            fk_yaw_tol_rad=fk_yaw_tol_rad,
            passive_max_iter=passive_max_iter,
        )

    def _result(
        self,
        *,
        success: bool,
        message: str,
        q_actuated: Dict[str, float],
        q_passive: Dict[str, float],
        q_dynamic: Dict[str, float],
        ik_result: IkSolveResult,
        passive_residual: float,
        fk_position_error_m: float,
        fk_yaw_error_rad: float,
        fk_xyz: np.ndarray | None = None,
        fk_yaw_rad: float = float("nan"),
    ) -> SteadyStateResult:
        return SteadyStateResult(
            success=success,
            message=message,
            q_actuated=q_actuated,
            q_passive=q_passive,
            q_dynamic=q_dynamic,
            ik_result=ik_result,
            passive_residual=passive_residual,
            fk_position_error_m=fk_position_error_m,
            fk_yaw_error_rad=fk_yaw_error_rad,
            fk_xyz=np.zeros(3, dtype=float) if fk_xyz is None else np.asarray(fk_xyz, dtype=float),
            fk_yaw_rad=fk_yaw_rad,
        )

    def _maps_from_dynamic(self, q_map: Mapping[str, float]) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        q_full = dict(q_map)
        for follower, leader in self._cfg.tied_joints.items():
            if leader in q_full:
                q_full[follower] = float(q_full[leader])
        return (
            q_full,
            {jn: float(q_full.get(jn, 0.0)) for jn in self._cfg.dynamic_joints},
            {jn: float(q_full.get(jn, 0.0)) for jn in self._cfg.actuated_joints},
        )

    @staticmethod
    def _empty_ik_result(status: str, message: str) -> IkSolveResult:
        return IkSolveResult(
            success=False,
            status=status,
            message=message,
            q_dynamic={},
            q_actuated={},
            q_passive={},
            iterations=0,
            cost=float("inf"),
            pos_error_m=float("inf"),
            rot_error_rad=float("inf"),
        )

    def _passive_only_ik_result(
        self,
        *,
        success: bool,
        q_dynamic: Dict[str, float],
        q_actuated: Dict[str, float],
        q_passive: Dict[str, float],
    ) -> IkSolveResult:
        return IkSolveResult(
            success=success,
            status="passive_equilibrium_only" if success else "passive_equilibrium_failed",
            message="actuated-only completion",
            q_dynamic=q_dynamic,
            q_actuated=q_actuated,
            q_passive=q_passive,
            iterations=0,
            cost=0.0 if success else float("inf"),
            pos_error_m=0.0 if success else float("inf"),
            rot_error_rad=0.0 if success else float("inf"),
        )

    def _failed_result(
        self,
        *,
        message: str,
        q_actuated: Dict[str, float],
        q_passive: Dict[str, float],
        q_dynamic: Dict[str, float],
        ik_result: IkSolveResult,
        passive_residual: float = float("inf"),
    ) -> SteadyStateResult:
        return self._result(
            success=False,
            message=message,
            q_actuated=q_actuated,
            q_passive=q_passive,
            q_dynamic=q_dynamic,
            ik_result=ik_result,
            passive_residual=passive_residual,
            fk_position_error_m=float("inf"),
            fk_yaw_error_rad=float("inf"),
        )

    def _seed_target_joints(
        self,
        target_pos: np.ndarray,
        target_yaw: float,
        q_seed: Mapping[str, float] | None,
    ) -> dict[str, float]:
        q_seed_map = merge_reference_seed(q_seed)
        q_seed_map.setdefault("theta1_slewing_joint", float(np.arctan2(float(target_pos[1]), float(target_pos[0]))))
        theta1 = float(q_seed_map["theta1_slewing_joint"])
        q_seed_map.setdefault(
            "theta8_rotator_joint",
            float(np.clip(np.arctan2(np.sin(theta1 - float(target_yaw)), np.cos(theta1 - float(target_yaw))), -1.01, 1.01)),
        )
        return q_seed_map

    def _initial_passive_seed(self, q_seed_map: Mapping[str, float]) -> Dict[str, float]:
        eq6, eq7 = self.analytic_equilibrium(
            float(q_seed_map.get("theta2_boom_joint", 0.0)),
            float(q_seed_map.get("theta3_arm_joint", 0.0)),
            float(q_seed_map.get("theta8_rotator_joint", 0.0)),
        )
        eq_seed = {"theta6_tip_joint": eq6, "theta7_tilt_joint": eq7}
        return {
            jn: float(q_seed_map.get(jn, eq_seed.get(jn, self._pas_q0[i])))
            for i, jn in enumerate(self._pas_names)
        }

    def _fk_check_message(self, converged: bool, pos_err: float, yaw_err: float) -> str:
        if not converged:
            return "Steady state iteration reached max iterations before convergence."
        return (
            "Steady state rejected by FK truth check "
            f"(pos_err={pos_err:.4f}m, yaw_err={yaw_err:.4f}rad)."
        )

    def compute(
        self,
        target_pos: np.ndarray,
        target_yaw: float,
        *,
        q_seed: Mapping[str, float] | None = None,
    ) -> SteadyStateResult:
        p = np.asarray(target_pos, dtype=float).ravel()
        if p.shape != (3,):
            raise ValueError(f"target_pos must have 3 elements, got shape {p.shape}.")

        T_target = pose_from_pos_yaw(p, float(target_yaw))
        q_seed_map = self._seed_target_joints(p, float(target_yaw), q_seed)
        q_p_eq = self._initial_passive_seed(q_seed_map)
        residual = float("inf")
        ik_res = None
        converged = False
        max_passive_ik_iters = 12
        passive_delta_tol = 1e-5

        for _ in range(max_passive_ik_iters):
            ik_res = self._ik.solve_pose(
                target_T_base_to_end=T_target,
                base_frame=self._base_frame,
                end_frame=self._end_frame,
                q_seed=q_seed_map,
                fixed_joints=q_p_eq,
                max_nfev=self._ik_max_nfev,
            )

            if not ik_res.success:
                return self._failed_result(
                    message=f"IK failed: {ik_res.message}",
                    q_actuated=ik_res.q_actuated,
                    q_passive=q_p_eq,
                    q_dynamic=ik_res.q_dynamic,
                    ik_result=ik_res,
                )

            q_full_map, _, _ = self._maps_from_dynamic(ik_res.q_dynamic)
            q_p_next, residual, passive_ok, passive_msg = self._passive_equilibrium(q_full_map, q_seed_map)
            if not passive_ok:
                return self._failed_result(
                    message=f"Passive equilibrium failed: {passive_msg} (residual={residual:.3e})",
                    q_actuated=ik_res.q_actuated,
                    q_passive=q_p_next,
                    q_dynamic=ik_res.q_dynamic,
                    ik_result=ik_res,
                    passive_residual=residual,
                )

            delta = max(abs(float(q_p_next.get(jn, 0.0)) - float(q_p_eq.get(jn, 0.0))) for jn in self._pas_names)
            q_p_eq = q_p_next
            q_seed_map.update(ik_res.q_dynamic)
            q_seed_map.update(q_p_eq)
            if delta <= passive_delta_tol:
                converged = True
                break

        if ik_res is None:
            return self._failed_result(
                message="Steady-state iteration failed to initialize IK.",
                q_actuated={},
                q_passive={},
                q_dynamic={},
                ik_result=self._empty_ik_result(
                    "steady_state_init_failed",
                    "steady-state iteration failed to initialize IK",
                ),
            )

        q_full_map = dict(ik_res.q_dynamic)
        q_full_map.update(q_p_eq)
        q_full_map, q_dyn_out, q_act_out = self._maps_from_dynamic(q_full_map)
        fk_xyz, fk_yaw_rad, fk_position_error_m, fk_yaw_error_rad = self._evaluate_fk_target_error(
            q_full_map,
            target_pos=p,
            target_yaw=float(target_yaw),
        )
        fk_ok = (
            np.isfinite(fk_position_error_m)
            and np.isfinite(fk_yaw_error_rad)
            and fk_position_error_m <= self._fk_position_tol_m
            and abs(fk_yaw_error_rad) <= self._fk_yaw_tol_rad
        )
        success = bool(converged and fk_ok)
        message = "Steady state computed successfully." if success else self._fk_check_message(
            converged,
            fk_position_error_m,
            fk_yaw_error_rad,
        )

        return self._result(
            success=success,
            message=message,
            q_actuated=q_act_out,
            q_passive=q_p_eq,
            q_dynamic=q_dyn_out,
            ik_result=ik_res,
            passive_residual=residual,
            fk_position_error_m=fk_position_error_m,
            fk_yaw_error_rad=fk_yaw_error_rad,
            fk_xyz=fk_xyz,
            fk_yaw_rad=fk_yaw_rad,
        )

    def complete_from_actuated(
        self,
        q_actuated: Mapping[str, float],
        *,
        q_seed: Mapping[str, float] | None = None,
    ) -> SteadyStateResult:
        q_seed_map = merge_reference_seed(q_seed, q_actuated=q_actuated)
        q_full_map, q_dyn_out, q_act_out = self._maps_from_dynamic(
            {jn: float(q_actuated.get(jn, 0.0)) for jn in self._cfg.actuated_joints}
        )

        q_p_eq, residual, passive_ok, passive_msg = self._passive_equilibrium(q_full_map, q_seed_map)
        if not passive_ok:
            return self._result(
                success=False,
                message=f"Passive equilibrium failed: {passive_msg} (residual={residual:.3e})",
                q_actuated=q_act_out,
                q_passive=q_p_eq,
                q_dynamic=q_dyn_out,
                ik_result=self._passive_only_ik_result(
                    success=False,
                    q_dynamic=q_dyn_out,
                    q_actuated=q_act_out,
                    q_passive=q_p_eq,
                ),
                passive_residual=residual,
                fk_position_error_m=float("nan"),
                fk_yaw_error_rad=float("nan"),
            )

        q_full_map.update(q_p_eq)
        _, q_dyn_out, q_act_out = self._maps_from_dynamic(q_full_map)
        return self._result(
            success=True,
            message="Passive equilibrium completed successfully.",
            q_actuated=q_act_out,
            q_passive=q_p_eq,
            q_dynamic=q_dyn_out,
            ik_result=self._passive_only_ik_result(
                success=True,
                q_dynamic=q_dyn_out,
                q_actuated=q_act_out,
                q_passive=q_p_eq,
            ),
            passive_residual=residual,
            fk_position_error_m=0.0,
            fk_yaw_error_rad=0.0,
            fk_xyz=np.zeros(3, dtype=float),
            fk_yaw_rad=float("nan"),
        )

    def _passive_equilibrium(
        self,
        q_act_map: Mapping[str, float],
        q_seed: Mapping[str, float] | None,
    ) -> tuple[Dict[str, float], float, bool, str]:
        q_p0 = list(self._pas_q0)
        if q_seed is not None:
            for i, jname in enumerate(self._pas_names):
                if jname in q_seed:
                    q_p0[i] = float(q_seed[jname])
        q_p0_arr = np.asarray(q_p0, dtype=float)

        q_base_map = dict(q_act_map)

        def gravity_passive(q_p_vals: np.ndarray) -> np.ndarray:
            q_cur = dict(q_base_map)
            for jname, val in zip(self._pas_names, q_p_vals):
                q_cur[jname] = float(val)
            q_pin = q_map_to_pin_q(self._pin_model, q_cur, pin)
            pin.computeGeneralizedGravity(self._pin_model, self._pin_data, q_pin)
            return np.array(
                [float(self._pin_data.g[vi]) for vi in self._pas_v_idx],
                dtype=float,
            )

        lb = np.asarray(self._pas_lb, dtype=float)
        ub = np.asarray(self._pas_ub, dtype=float)
        q0 = np.asarray(q_p0_arr, dtype=float)
        finite_bounds = np.isfinite(lb) & np.isfinite(ub)
        if np.any(finite_bounds):
            q0[finite_bounds] = np.clip(q0[finite_bounds], lb[finite_bounds], ub[finite_bounds])

        lsq = least_squares(
            gravity_passive,
            q0,
            bounds=(lb, ub),
            xtol=self._passive_tol,
            ftol=self._passive_tol,
            gtol=self._passive_tol,
            max_nfev=self._passive_max_iter,
        )

        q_p_eq = np.asarray(lsq.x, dtype=float)
        residual = float(np.linalg.norm(gravity_passive(q_p_eq)))
        converged = bool(lsq.success) and residual <= float(self._passive_residual_tol)
        msg = str(lsq.message)
        if not converged:
            print(f"[CraneSteadyState] passive equilibrium did not converge: {msg}; residual={residual:.3e}")

        return (
            {jname: float(q_p_eq[i]) for i, jname in enumerate(self._pas_names)},
            residual,
            converged,
            msg,
        )

    def _evaluate_fk_target_error(
        self,
        q_map: Mapping[str, float],
        *,
        target_pos: np.ndarray,
        target_yaw: float,
    ) -> tuple[np.ndarray, float, float, float]:
        q_full = dict(q_map)
        for follower, leader in self._cfg.tied_joints.items():
            if leader in q_full:
                q_full[follower] = float(q_full[leader])
        T = self._ik._analytic._fk(
            q_full,
            base_frame=self._base_frame,
            end_frame=self._end_frame,
        )
        fk_xyz = np.asarray(T[:3, 3], dtype=float).reshape(3)
        fk_yaw = phi_tool_from_transform(T)
        pos_err = float(np.linalg.norm(fk_xyz - np.asarray(target_pos, dtype=float).reshape(3)))
        yaw_err = float(np.arctan2(np.sin(fk_yaw - float(target_yaw)), np.cos(fk_yaw - float(target_yaw))))
        return fk_xyz, fk_yaw, pos_err, yaw_err
