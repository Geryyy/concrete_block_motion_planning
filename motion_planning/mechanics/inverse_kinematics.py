from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np

from .config import AnalyticModelConfig
from .crane_geometry import DEFAULT_CRANE_GEOMETRY, CraneGeometryConstants
from .model_description import ModelDescription
from .pinocchio_utils import fk_homogeneous, joint_bounds
from .pose_conventions import phi_tool_from_rotation


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

    def _sanitize_seed_value(self, joint_name: str, value: float) -> float:
        jid = int(self._pin_model.getJointId(joint_name))
        joint = self._pin_model.joints[jid]
        if int(joint.nq) == 1:
            lo, hi = self._joint_bounds(joint_name)
            out = float(value)
            if np.isfinite(lo):
                out = max(out, lo)
            if np.isfinite(hi):
                out = min(out, hi)
            return out
        if int(joint.nq) == 2 and int(joint.nv) == 1:
            return self._wrap_angle(float(value))
        return float(value)

    def _use_seed_value(self, joint_name: str, value: float) -> bool:
        jid = int(self._pin_model.getJointId(joint_name))
        joint = self._pin_model.joints[jid]
        if int(joint.nq) == 2 and int(joint.nv) == 1:
            return np.isfinite(value) and abs(float(value)) <= 10.0 * np.pi
        return np.isfinite(value)


class AnalyticIKSolver(_IKBase):
    """Analytic IK: fixed joints + 1D independent-joint search.

    Geometry note
    -------------
    The 2-link arm model (boom a2, effective forearm sqrt(a3^2+d45^2)) correctly
    describes the arm from K0 to K5_inner_telescope — the inner telescope endpoint.
    The remaining chain (theta6_tip_joint, theta7_tilt_joint, theta8_rotator_joint)
    carries K5 to K8_tool_center_point (TCP).  Because the end-effector chain is
    not part of the 2-link geometry, we must subtract its contribution from the
    TCP target before running the geometry, then verify the full FK against K8
    afterwards.
    """

    # Frame that marks the end of the 2-link arm geometry.
    _GEOMETRY_FRAME = "K5_inner_telescope"

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
        # Timber IK uses phiTool, i.e. the world-yaw of the K8 frame's local Y axis.
        phi_tool = phi_tool_from_rotation(R_target)

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

        # -----------------------------------------------------------------------
        # Compute K8_in_K5 — the constant K8 position in the K5 local frame.
        #
        # The 2-link geometry reaches K5_inner_telescope, not K8_tool_center_point.
        # The K5→K8 chain (theta6/theta7/theta8) rotates the K8 position vector
        # with the arm orientation.  K8 in K5 frame is constant given theta6/7/8.
        #
        # Arm-plane correction for a given theta2:
        #   K5_x_arm = [cos(theta2), sin(theta2)]   (K5 x-axis ≈ boom direction)
        #   K5_y_arm = [-sin(theta2), cos(theta2)]  (K5 y-axis = perpendicular)
        #   correction_r = k8_in_k5[0]*cos(theta2) - k8_in_k5[1]*sin(theta2)
        #   correction_z = k8_in_k5[0]*sin(theta2) + k8_in_k5[1]*cos(theta2)
        #   K5_target = K8_target - [correction_r, correction_z]
        #
        # Two-pass approach: pass-1 targets K8 to get approximate theta2, then
        # pass-2 uses the corrected K5 target for the precise result.
        # -----------------------------------------------------------------------
        q_chain_ref: dict[str, float] = {jn: 0.0 for jn in self._cfg.dynamic_joints}
        q_chain_ref["theta1_slewing_joint"] = float(theta1)
        q_chain_ref["theta6_tip_joint"] = float(q6)
        q_chain_ref["theta7_tilt_joint"] = float(q7)
        q_chain_ref["theta8_rotator_joint"] = float(theta8)
        for follower, leader in self._cfg.tied_joints.items():
            if leader in q_chain_ref:
                q_chain_ref[follower] = float(q_chain_ref[leader])

        T_k5_ref = self._fk(q_chain_ref, base_frame=self._base_frame, end_frame=self._GEOMETRY_FRAME)
        T_k8_ref = self._fk(q_chain_ref, base_frame=self._base_frame, end_frame=self._end_frame)
        # K8 position in K5 local frame (constant given theta6/theta7/theta8).
        T_k5_inv = np.linalg.inv(T_k5_ref)
        k8_in_k5 = np.asarray((T_k5_inv @ T_k8_ref)[:3, 3], dtype=float)
        k8lx = float(k8_in_k5[0])  # K8 x-component in K5 frame
        k8ly = float(k8_in_k5[1])  # K8 y-component in K5 frame

        g = self._geometry
        r_t = float(np.hypot(x_t, y_t))

        # d45 bounds from q4 limits (constant for both passes).
        d45_q4_lo = g.d4 + 2.0 * lo4
        d45_q4_hi = g.d4 + 2.0 * hi4

        # Cost reference values (midrange + seed deviation, as in timber).
        mid2 = 0.5 * (lo2 + hi2)
        mid3 = 0.5 * (lo3 + hi3)
        mid4 = 0.5 * (lo4 + hi4)
        rng2 = max(hi2 - lo2, 1e-9)
        rng3 = max(hi3 - lo3, 1e-9)
        rng4 = max(hi4 - lo4, 1e-9)
        seed_q2 = float(seed.get("theta2_boom_joint", mid2))
        seed_q3 = float(seed.get("theta3_arm_joint", mid3))
        seed_q4 = float(seed.get("q4_big_telescope", mid4))

        # Helper: corrected arm-plane K5 target from K8 target.
        # K5 x-axis in the arm plane = [cos(theta2+theta3), sin(theta2+theta3)],
        # so the K5→K8 offset projection uses the combined angle alpha.
        def _k5_target(alpha: float) -> np.ndarray:
            ca, sa = float(np.cos(alpha)), float(np.sin(alpha))
            corr_r = k8lx * ca - k8ly * sa
            corr_z = k8lx * sa + k8ly * ca
            return np.array([r_t - corr_r, z_t - corr_z], dtype=float)

        # -----------------------------------------------------------------------
        # d45-parameterized analytic IK  (ported from timber invKin2DoF /
        # calcInverseKinCraneColAvoid golden-section approach).
        #
        # For the 2-link arm (boom a2, effective forearm link2=sqrt(a3^2+d45^2)):
        #   theta2 = base_angle ± arccos(law-of-cosines at joint-2)
        #   theta3 = arccos(law-of-cosines at joint-3) + atan2(a3,d45) - π/2
        # Both are purely geometric given d45 — no FK needed in the inner loop.
        # A single FK is run at the end to verify the best candidate.
        #
        # Iterative K8→K5 correction (timber-style geometry adaptation):
        #   The 2-link geometry reaches K5, not K8.  The K5→K8 offset (~1m)
        #   depends on theta2 through the arm-plane projection.  We iterate:
        #   start with K8 target, compute theta2, apply correction, repeat
        #   until theta2 converges (typically 3-5 iterations).
        # -----------------------------------------------------------------------

        last_search_n_pts = 0

        def _run_d45_search(
            p5_arm: np.ndarray,
        ) -> tuple[float | None, float | None, float | None]:
            """Vectorised d45 search targeting p5_arm (K5 position in arm plane)."""
            nonlocal last_search_n_pts
            dp = p5_arm - g.p2
            d_p2p5 = float(np.linalg.norm(dp))
            if d_p2p5 < 1e-9:
                return None, None, None

            # Triangle feasibility bounds for d45.
            tri_hi_sq = (d_p2p5 + g.a2) ** 2 - g.a3 ** 2
            if tri_hi_sq <= 0.0:
                return None, None, None
            d45_tri_lo = float(np.sqrt(max(0.0, (d_p2p5 - g.a2) ** 2 - g.a3 ** 2))) + 1e-9
            d45_tri_hi = float(np.sqrt(tri_hi_sq)) - 1e-9

            d45_lo_eff = max(d45_q4_lo, d45_tri_lo)
            d45_hi_eff = min(d45_q4_hi, d45_tri_hi)
            if d45_lo_eff >= d45_hi_eff:
                return None, None, None

            # If q4 is fixed, collapse search to single d45 value.
            if "q4_big_telescope" in fixed:
                q4_fixed = float(fixed["q4_big_telescope"])
                d45_lo_eff = g.d4 + 2.0 * q4_fixed
                d45_hi_eff = d45_lo_eff + 1e-12

            n_pts = 1 if d45_hi_eff - d45_lo_eff < 1e-11 else 120
            last_search_n_pts = n_pts
            d45_vec = np.linspace(d45_lo_eff, d45_hi_eff, n_pts, dtype=float)
            link2_vec = np.sqrt(g.a3 ** 2 + d45_vec ** 2)
            q4_vec = 0.5 * (d45_vec - g.d4)

            # Law of cosines — angle at joint-2 (beta) and joint-3 (gamma).
            cos_beta = np.clip(
                (g.a2 ** 2 + d_p2p5 ** 2 - link2_vec ** 2) / (2.0 * g.a2 * d_p2p5),
                -1.0, 1.0,
            )
            beta_vec = np.arccos(cos_beta)
            base_angle = float(np.arctan2(dp[1], dp[0]))

            cos_gamma = np.clip(
                (g.a2 ** 2 + link2_vec ** 2 - d_p2p5 ** 2) / (2.0 * g.a2 * link2_vec),
                -1.0, 1.0,
            )
            gamma_vec = np.arccos(cos_gamma)
            theta3_vec = np.arctan2(g.a3, d45_vec) - 0.5 * np.pi + gamma_vec

            best_q2_: float | None = None
            best_q3_: float | None = None
            best_q4_: float | None = None
            best_cost_ = float("inf")

            for sign in (+1.0, -1.0):
                theta2_vec = base_angle + sign * beta_vec

                if "theta2_boom_joint" in fixed:
                    theta2_fix = float(fixed["theta2_boom_joint"])
                    valid = np.abs(theta2_vec - theta2_fix) < 1e-3
                else:
                    valid = (theta2_vec >= lo2 - 1e-6) & (theta2_vec <= hi2 + 1e-6)

                if "theta3_arm_joint" in fixed:
                    theta3_use = np.full_like(theta3_vec, float(fixed["theta3_arm_joint"]))
                else:
                    theta3_use = theta3_vec
                    valid &= (
                        (theta3_use >= lo3 - 1e-6)
                        & (theta3_use <= np.minimum(hi3, g.theta3_max) + 1e-6)
                    )

                valid &= (q4_vec >= lo4 - 1e-6) & (q4_vec <= hi4 + 1e-6)
                valid &= np.isfinite(theta2_vec) & np.isfinite(theta3_use)

                if not np.any(valid):
                    continue

                cost_vec = (
                    ((theta2_vec - mid2) / rng2) ** 2
                    + ((theta3_use - mid3) / rng3) ** 2
                    + ((q4_vec - mid4) / rng4) ** 2
                    + 0.1 * (
                        (theta2_vec - seed_q2) ** 2
                        + (theta3_use - seed_q3) ** 2
                        + (q4_vec - seed_q4) ** 2
                    )
                )
                cost_vec[~valid] = 1e10

                idx = int(np.argmin(cost_vec))
                if cost_vec[idx] < best_cost_:
                    best_cost_ = float(cost_vec[idx])
                    best_q2_ = float(theta2_vec[idx])
                    best_q3_ = float(theta3_use[idx])
                    best_q4_ = float(q4_vec[idx])

            return best_q2_, best_q3_, best_q4_

        # Iterative K5 correction: start with alpha=0 (no correction),
        # then refine until (theta2+theta3) converges.
        alpha_prev = 0.0
        best_q2: float | None = None
        best_q3: float | None = None
        best_q4: float | None = None
        for _iter in range(8):
            p5_corrected = _k5_target(alpha_prev)
            q2_i, q3_i, q4_i = _run_d45_search(p5_corrected)
            if q2_i is None:
                break
            best_q2, best_q3, best_q4 = q2_i, q3_i, q4_i
            alpha_i = q2_i + q3_i
            if abs(alpha_i - alpha_prev) < 1e-4:
                break
            alpha_prev = alpha_i
        if best_q2 is None:
            return None

        # Single FK call to verify the best geometric solution.
        q_try = dict(seed)
        q_try.update(
            {
                "theta1_slewing_joint": float(theta1),
                "theta2_boom_joint": best_q2,
                "theta3_arm_joint": best_q3,
                "q4_big_telescope": best_q4,
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
        phi_try = phi_tool_from_rotation(T_try[:3, :3])
        rot_err = float(np.arctan2(np.sin(phi_try - phi_tool), np.cos(phi_try - phi_tool)))
        rot_err = float(abs(rot_err))

        if pos_err > 2e-2 or rot_err > 5e-2:
            return None

        q_act = {jn: float(q_try.get(jn, 0.0)) for jn in act_names}
        q_pas = {jn: float(q_try.get(jn, 0.0)) for jn in self._cfg.passive_joints if jn in q_try}
        return IkSolveResult(
            success=True,
            status="analytic_success",
            message="IK analytic solve converged (d45 vectorised search)",
            q_dynamic={jn: float(q_try.get(jn, 0.0)) for jn in self._cfg.dynamic_joints},
            q_actuated=q_act,
            q_passive=q_pas,
            iterations=max(1, last_search_n_pts * 2),
            cost=float(0.5 * (pos_err * pos_err + rot_err * rot_err)),
            pos_error_m=pos_err,
            rot_error_rad=rot_err,
        )


