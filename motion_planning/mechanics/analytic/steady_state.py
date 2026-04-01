"""Crane steady-state computation: workspace target → actuated + passive equilibrium.

Given a desired end-effector position and yaw angle this module:

1. Solves the 5-DoF IK for the actuated joints
   (theta1, theta2, theta3, q4, theta8) using :class:`AnalyticInverseKinematics`.
2. Finds the static-equilibrium values of the passive joints
   (theta6_tip_joint, theta7_tilt_joint) by solving g_p(q_act, q_p) = 0
   via ``scipy.fsolve``.

Quick-start
-----------
>>> from motion_planning.mechanics.analytic import CraneSteadyState
>>> ss = CraneSteadyState.default()
>>> result = ss.compute(target_pos=np.array([5.0, 2.0, 3.0]), target_yaw=0.5)
>>> print(result.q_actuated)
>>> print(result.q_passive)
>>> print(f"passive residual: {result.passive_residual:.2e}")
"""

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


# ------------------------------------------------------------------ #
# Result dataclass
# ------------------------------------------------------------------ #

@dataclass
class SteadyStateResult:
    """Result of :meth:`CraneSteadyState.compute`.

    Attributes
    ----------
    success:
        ``True`` if both the IK and the passive equilibrium solve converged.
    message:
        Human-readable status message.
    q_actuated:
        Equilibrium values for the 5 actuated joints.
    q_passive:
        Equilibrium values for the 2 passive joints (g_p = 0 solution).
    q_dynamic:
        Combined dict of all 7 dynamic joint values (actuated + passive).
    ik_result:
        The full :class:`~.inverse_kinematics.IkSolveResult` for inspection.
    passive_residual:
        ``||g_p(q)||`` at the equilibrium solution (should be ~1e-11).
    fk_position_error_m:
        Final TCP position error of the returned state against the requested target.
    fk_yaw_error_rad:
        Final ``phiTool`` error of the returned state against the requested target.
    fk_xyz:
        Final TCP position of the returned state in the configured base frame.
    fk_yaw_rad:
        Final ``phiTool`` of the returned state in the configured base frame.
    """

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


# ------------------------------------------------------------------ #
# Main class
# ------------------------------------------------------------------ #

class CraneSteadyState:
    """Compute crane steady state from workspace target.

    Parameters
    ----------
    desc:
        Analytic model description (wraps the pinocchio model).
    config:
        Analytic model config (joint categorisation, tied joints, etc.).
    ik_max_nfev:
        Maximum function evaluations for the numeric IK fallback.
    passive_tol:
        Convergence tolerance passed to ``scipy.fsolve`` for the passive
        equilibrium solve.
    passive_max_iter:
        Maximum iterations for the passive equilibrium fsolve.
    base_frame:
        Pinocchio frame name used as the IK reference base.
    end_frame:
        Pinocchio frame name used as the IK end-effector target.
    """

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

        # Pre-compute passive joint v-indices and URDF-midpoint initial guesses.
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

        # ---- Pre-compute mass parameters for analytic equilibrium --------
        # Port of timber comp_equilibrium.hpp — needs masses and COM of
        # bodies at theta6, theta7, theta8 joints plus link lengths a6, d8.
        self._eq_params = self._extract_equilibrium_params()

    def _extract_equilibrium_params(self) -> dict[str, float]:
        """Extract mass/COM/link params from pinocchio for analytic equilibrium."""
        model = self._pin_model
        # Joint IDs for the wrist chain
        j6_id = int(model.getJointId("theta6_tip_joint"))
        j7_id = int(model.getJointId("theta7_tilt_joint"))
        j8_id = int(model.getJointId("theta8_rotator_joint"))
        i6 = model.inertias[j6_id]
        i7 = model.inertias[j7_id]
        i8 = model.inertias[j8_id]
        # a6: K5→K6 offset (from frame placement)
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
        # d8: distance from K7/K6 to K8 along the rotator axis (at zero config)
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
        """Analytic passive-joint equilibrium (ported from timber comp_equilibrium).

        Given actuated arm joints, returns (theta6, theta7) at static
        equilibrium under gravity.  This is a closed-form formula using
        mass/COM parameters extracted from the pinocchio model.

        Parameters
        ----------
        theta2, theta3, theta8 : float
            Actuated joint positions.

        Returns
        -------
        (theta6, theta7) : tuple[float, float]
        """
        p = self._eq_params
        m6, m7, m8 = p["m6"], p["m7"], p["m8"]
        s6x, s6z = p["s6x"], p["s6z"]
        s7x, s7y, s7z = p["s7x"], p["s7y"], p["s7z"]
        s8x, s8y, s8z = p["s8x"], p["s8y"], p["s8z"]
        a6, d8 = p["a6"], p["d8"]

        # theta7: gravity balance of K7+K8 pendulum (independent of theta2/theta3)
        theta7 = -np.arctan2(
            d8 * m8 + s7z * m7 + s8z * m8,
            np.cos(theta8) * m8 * s8x - np.sin(theta8) * m8 * s8y + m7 * s7x,
        ) + np.pi

        # theta6: gravity balance of K6+K7+K8 assembly
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

    # ------------------------------------------------------------------ #
    # Constructor helpers
    # ------------------------------------------------------------------ #

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
        """Construct using the bundled crane config (``crane_config.yaml``)."""
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

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def compute(
        self,
        target_pos: np.ndarray,
        target_yaw: float,
        *,
        q_seed: Mapping[str, float] | None = None,
    ) -> SteadyStateResult:
        """Compute the steady state for a given workspace target.

        Parameters
        ----------
        target_pos:
            Desired end-effector position ``[x, y, z]`` in the base frame
            (metres).
        target_yaw:
            Desired timber ``phiTool`` angle in radians. This matches the
            projected world-yaw of the K8 frame's local Y axis:
            ``phiTool = atan2(R[1,1], R[0,1])``.
        q_seed:
            Optional joint-name → value seed.  For passive joints the seed
            is used only as the fsolve initial guess; it is overwritten by
            the equilibrium solution.

        Returns
        -------
        SteadyStateResult
        """
        p = np.asarray(target_pos, dtype=float).ravel()
        if p.shape != (3,):
            raise ValueError(f"target_pos must have 3 elements, got shape {p.shape}.")

        T_target = _pose_from_pos_yaw(p, float(target_yaw))

        q_seed_map = dict(q_seed or {})

        # Auto-seed theta1 and theta8 if not provided.
        if "theta1_slewing_joint" not in q_seed_map:
            theta1_guess = float(np.arctan2(float(p[1]), float(p[0])))
            q_seed_map["theta1_slewing_joint"] = theta1_guess
        if "theta8_rotator_joint" not in q_seed_map:
            theta1_for_t8 = float(q_seed_map["theta1_slewing_joint"])
            theta8_guess = float(np.arctan2(np.sin(theta1_for_t8 - float(target_yaw)), np.cos(theta1_for_t8 - float(target_yaw))))
            q_seed_map["theta8_rotator_joint"] = float(np.clip(theta8_guess, -1.01, 1.01))

        # Use analytic equilibrium (ported from timber comp_equilibrium) as
        # initial passive joint estimate — much better than midrange defaults.
        _theta2_seed = float(q_seed_map.get("theta2_boom_joint", 0.0))
        _theta3_seed = float(q_seed_map.get("theta3_arm_joint", 0.0))
        _theta8_seed = float(q_seed_map.get("theta8_rotator_joint", 0.0))
        _eq6, _eq7 = self.analytic_equilibrium(_theta2_seed, _theta3_seed, _theta8_seed)
        _eq_seed = {"theta6_tip_joint": _eq6, "theta7_tilt_joint": _eq7}

        q_p_eq = {}
        for i, jn in enumerate(self._pas_names):
            if jn in q_seed_map:
                q_p_eq[jn] = float(q_seed_map[jn])
            elif jn in _eq_seed:
                q_p_eq[jn] = float(_eq_seed[jn])
            else:
                q_p_eq[jn] = float(self._pas_q0[i])
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
                return SteadyStateResult(
                    success=False,
                    message=f"IK failed: {ik_res.message}",
                    q_actuated=ik_res.q_actuated,
                    q_passive=q_p_eq,
                q_dynamic=ik_res.q_dynamic,
                ik_result=ik_res,
                passive_residual=float("inf"),
                fk_position_error_m=float("inf"),
                fk_yaw_error_rad=float("inf"),
                fk_xyz=np.zeros(3, dtype=float),
                fk_yaw_rad=float("nan"),
            )

            q_full_map = dict(ik_res.q_dynamic)
            for follower, leader in self._cfg.tied_joints.items():
                if leader in q_full_map:
                    q_full_map[follower] = q_full_map[leader]

            q_p_next, residual, passive_ok, passive_msg = self._passive_equilibrium(q_full_map, q_seed_map)
            if not passive_ok:
                return SteadyStateResult(
                    success=False,
                    message=f"Passive equilibrium failed: {passive_msg} (residual={residual:.3e})",
                    q_actuated=ik_res.q_actuated,
                    q_passive=q_p_next,
                    q_dynamic=ik_res.q_dynamic,
                    ik_result=ik_res,
                    passive_residual=residual,
                    fk_position_error_m=float("inf"),
                    fk_yaw_error_rad=float("inf"),
                    fk_xyz=np.zeros(3, dtype=float),
                    fk_yaw_rad=float("nan"),
                )

            delta = 0.0
            if q_p_eq:
                delta = max(abs(float(q_p_next.get(jn, 0.0)) - float(q_p_eq.get(jn, 0.0))) for jn in self._pas_names)
            q_p_eq = q_p_next
            q_seed_map.update(ik_res.q_dynamic)
            q_seed_map.update(q_p_eq)
            if delta <= passive_delta_tol:
                converged = True
                break

        if ik_res is None:
            return SteadyStateResult(
                success=False,
                message="Steady-state iteration failed to initialize IK.",
                q_actuated={},
                q_passive={},
                q_dynamic={},
                ik_result=IkSolveResult(
                    success=False,
                    status="steady_state_init_failed",
                    message="steady-state iteration failed to initialize IK",
                    q_dynamic={},
                    q_actuated={},
                    q_passive={},
                    iterations=0,
                    cost=float("inf"),
                    pos_error_m=float("inf"),
                    rot_error_rad=float("inf"),
                ),
                passive_residual=float("inf"),
                fk_position_error_m=float("inf"),
                fk_yaw_error_rad=float("inf"),
                fk_xyz=np.zeros(3, dtype=float),
                fk_yaw_rad=float("nan"),
            )

        q_full_map = dict(ik_res.q_dynamic)
        q_full_map.update(q_p_eq)
        for follower, leader in self._cfg.tied_joints.items():
            if leader in q_full_map:
                q_full_map[follower] = q_full_map[leader]

        q_dyn_out = {jn: float(q_full_map.get(jn, 0.0)) for jn in self._cfg.dynamic_joints}
        q_act_out = {jn: float(q_full_map.get(jn, 0.0)) for jn in self._cfg.actuated_joints}
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
        if success:
            message = "Steady state computed successfully."
        elif not converged:
            message = "Steady state iteration reached max iterations before convergence."
        else:
            message = (
                "Steady state rejected by FK truth check "
                f"(pos_err={fk_position_error_m:.4f}m, yaw_err={fk_yaw_error_rad:.4f}rad)."
            )

        return SteadyStateResult(
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
        """Complete passive equilibrium for a known actuated joint state."""
        q_full_map: dict[str, float] = {
            jn: float(q_actuated.get(jn, 0.0)) for jn in self._cfg.actuated_joints
        }
        for follower, leader in self._cfg.tied_joints.items():
            if leader in q_full_map:
                q_full_map[follower] = q_full_map[leader]

        q_p_eq, residual, passive_ok, passive_msg = self._passive_equilibrium(q_full_map, q_seed)
        if not passive_ok:
            q_dyn_out = {jn: float(q_full_map.get(jn, 0.0)) for jn in self._cfg.dynamic_joints}
            q_act_out = {jn: float(q_full_map.get(jn, 0.0)) for jn in self._cfg.actuated_joints}
            return SteadyStateResult(
                success=False,
                message=f"Passive equilibrium failed: {passive_msg} (residual={residual:.3e})",
                q_actuated=q_act_out,
                q_passive=q_p_eq,
                q_dynamic=q_dyn_out,
                ik_result=IkSolveResult(
                    success=False,
                    status="passive_equilibrium_failed",
                    message="actuated-only completion",
                    q_dynamic=q_dyn_out,
                    q_actuated=q_act_out,
                    q_passive=q_p_eq,
                    iterations=0,
                    cost=float("inf"),
                    pos_error_m=float("inf"),
                    rot_error_rad=float("inf"),
                ),
                passive_residual=residual,
                fk_position_error_m=float("nan"),
                fk_yaw_error_rad=float("nan"),
                fk_xyz=np.zeros(3, dtype=float),
                fk_yaw_rad=float("nan"),
            )

        q_full_map.update(q_p_eq)
        for follower, leader in self._cfg.tied_joints.items():
            if leader in q_full_map:
                q_full_map[follower] = q_full_map[leader]

        q_dyn_out = {jn: float(q_full_map.get(jn, 0.0)) for jn in self._cfg.dynamic_joints}
        q_act_out = {jn: float(q_full_map.get(jn, 0.0)) for jn in self._cfg.actuated_joints}
        return SteadyStateResult(
            success=True,
            message="Passive equilibrium completed successfully.",
            q_actuated=q_act_out,
            q_passive=q_p_eq,
            q_dynamic=q_dyn_out,
            ik_result=IkSolveResult(
                success=True,
                status="passive_equilibrium_only",
                message="actuated-only completion",
                q_dynamic=q_dyn_out,
                q_actuated=q_act_out,
                q_passive=q_p_eq,
                iterations=0,
                cost=0.0,
                pos_error_m=0.0,
                rot_error_rad=0.0,
            ),
            passive_residual=residual,
            fk_position_error_m=0.0,
            fk_yaw_error_rad=0.0,
            fk_xyz=np.zeros(3, dtype=float),
            fk_yaw_rad=float("nan"),
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _passive_equilibrium(
        self,
        q_act_map: Mapping[str, float],
        q_seed: Mapping[str, float] | None,
    ) -> tuple[Dict[str, float], float, bool, str]:
        """Solve g_p(q_act, q_p) = 0 for the passive joints.

        At static equilibrium (dq = 0) the passive-joint equations of motion
        reduce to the gravity term:  g_p(q_act, q_p) = 0.  We solve this
        small nonlinear system via ``scipy.fsolve``.

        Returns
        -------
        (passive_dict, residual_norm)
            passive_dict:  joint-name → equilibrium value for each passive joint.
            residual_norm: ||g_p|| at the solution (ideal: ~1e-11).
        """
        # Initial guess: prefer seed values, fall back to URDF midpoints.
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
        fk_yaw = _phi_tool_from_transform(T)
        pos_err = float(np.linalg.norm(fk_xyz - np.asarray(target_pos, dtype=float).reshape(3)))
        yaw_err = float(np.arctan2(np.sin(fk_yaw - float(target_yaw)), np.cos(fk_yaw - float(target_yaw))))
        return fk_xyz, fk_yaw, pos_err, yaw_err


# ------------------------------------------------------------------ #
# Module-level helpers
# ------------------------------------------------------------------ #

def _phi_tool_from_transform(T: np.ndarray) -> float:
    """Extract timber ``phiTool`` from a homogeneous transform.

    In the timber stack, ``phiTool`` is the world-yaw of the K8 frame's local
    Y axis projected into the XY plane.
    """
    T_arr = np.asarray(T, dtype=float).reshape(4, 4)
    return float(np.arctan2(T_arr[1, 1], T_arr[0, 1]))


def _pose_from_pos_yaw(pos: np.ndarray, yaw: float) -> np.ndarray:
    """Build a 4x4 homogeneous transform from position + timber ``phiTool``.

    For a pure Z rotation, the K8 frame's local Y axis has yaw ``phiTool``.
    That corresponds to a body rotation of ``phiTool - pi/2`` about world Z.
    """
    rot_z = float(yaw) - 0.5 * np.pi
    cy, sy = float(np.cos(rot_z)), float(np.sin(rot_z))
    T = np.eye(4, dtype=float)
    T[0, 0] = cy;  T[0, 1] = -sy
    T[1, 0] = sy;  T[1, 1] =  cy
    T[:3, 3] = pos
    return T
