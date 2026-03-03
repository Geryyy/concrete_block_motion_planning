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
    """

    success: bool
    message: str
    q_actuated: Dict[str, float]
    q_passive: Dict[str, float]
    q_dynamic: Dict[str, float]
    ik_result: IkSolveResult
    passive_residual: float


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
        ik_max_nfev: int = 200,
        passive_tol: float = 1e-10,
        passive_residual_tol: float = 1e-6,
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

    # ------------------------------------------------------------------ #
    # Constructor helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def default(
        cls,
        *,
        ik_max_nfev: int = 200,
        passive_tol: float = 1e-10,
        passive_residual_tol: float = 1e-6,
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
            Desired yaw angle of the tool frame (radians) around the world
            Z-axis.  This is ``phi_tool = arctan2(R[1,0], R[0,0])`` as used
            by the analytic IK.
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

        ik_res = self._ik.solve_pose(
            target_T_base_to_end=T_target,
            base_frame=self._base_frame,
            end_frame=self._end_frame,
            q_seed=q_seed,
            max_nfev=self._ik_max_nfev,
        )

        if not ik_res.success:
            return SteadyStateResult(
                success=False,
                message=f"IK failed: {ik_res.message}",
                q_actuated=ik_res.q_actuated,
                q_passive={jn: 0.0 for jn in self._pas_names},
                q_dynamic=ik_res.q_dynamic,
                ik_result=ik_res,
                passive_residual=float("inf"),
            )

        # Build full q map from the IK solution (includes passive at seed/0).
        q_full_map: dict[str, float] = dict(ik_res.q_dynamic)
        for follower, leader in self._cfg.tied_joints.items():
            if leader in q_full_map:
                q_full_map[follower] = q_full_map[leader]

        q_p_eq, residual, passive_ok, passive_msg = self._passive_equilibrium(q_full_map, q_seed)
        if not passive_ok:
            return SteadyStateResult(
                success=False,
                message=f"Passive equilibrium failed: {passive_msg} (residual={residual:.3e})",
                q_actuated=ik_res.q_actuated,
                q_passive=q_p_eq,
                q_dynamic=ik_res.q_dynamic,
                ik_result=ik_res,
                passive_residual=residual,
            )

        # Merge equilibrium passive values back into full map.
        q_full_map.update(q_p_eq)
        for follower, leader in self._cfg.tied_joints.items():
            if leader in q_full_map:
                q_full_map[follower] = q_full_map[leader]

        q_dyn_out = {jn: float(q_full_map.get(jn, 0.0)) for jn in self._cfg.dynamic_joints}
        q_act_out = {jn: float(q_full_map.get(jn, 0.0)) for jn in self._cfg.actuated_joints}

        return SteadyStateResult(
            success=True,
            message="Steady state computed successfully.",
            q_actuated=q_act_out,
            q_passive=q_p_eq,
            q_dynamic=q_dyn_out,
            ik_result=ik_res,
            passive_residual=residual,
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


# ------------------------------------------------------------------ #
# Module-level helpers
# ------------------------------------------------------------------ #

def _pose_from_pos_yaw(pos: np.ndarray, yaw: float) -> np.ndarray:
    """Build a 4x4 homogeneous transform from position + yaw angle.

    The rotation is a pure Z-axis rotation so that:
        phi_tool = arctan2(R[1,0], R[0,0]) == yaw

    This matches the ``phi_tool`` convention used by
    :class:`~.inverse_kinematics.AnalyticInverseKinematics`.
    """
    cy, sy = float(np.cos(yaw)), float(np.sin(yaw))
    T = np.eye(4, dtype=float)
    T[0, 0] = cy;  T[0, 1] = -sy
    T[1, 0] = sy;  T[1, 1] =  cy
    T[:3, 3] = pos
    return T
