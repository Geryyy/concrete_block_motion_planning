"""VP-STO crane model — Python port of timber crane_vpsto_model.cpp.

This implements the crane-specific cost model for VP-STO trajectory
optimization.  The 5 actuated joints are partitioned as:

- Fixed (computed from target): theta1 (slewing), theta8 (rotator)
- Independent (optimized by VP-STO): theta2 (boom)
- Dependent (geometry-constrained): theta3 (arm), q4 (telescope)

The geometry uses the 2-link arm model (boom a2 + effective forearm
sqrt(a3^2 + d45^2)) ported from timber's compute_dependent_joints().
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CraneVpstoConfig:
    """Crane geometry and joint limits for VP-STO."""

    # Crane geometry (from CraneGeometryConstants / timber parStruct)
    a1: float = 0.18
    d1: float = 2.425
    a2: float = 3.49288333
    a3: float = 0.3925
    d4: float = 3.157001602823

    # Actuated joint order: [theta1, theta2, theta3, q4, theta8]
    q_min: np.ndarray = field(default_factory=lambda: np.array([-3.71, -1.57, -0.35, 0.0, -3.14]))
    q_max: np.ndarray = field(default_factory=lambda: np.array([3.71, 0.35, 1.40, 2.236, 3.14]))
    q_dot_min: np.ndarray = field(default_factory=lambda: np.array([-0.5, -0.3, -0.3, -0.15, -1.0]))
    q_dot_max: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.3, 0.3, 0.15, 1.0]))
    q_ddot_min: np.ndarray = field(default_factory=lambda: np.array([-1.0, -1.0, -1.0, -1.0, -1.0]))
    q_ddot_max: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0, 1.0, 1.0]))

    # CBS theta8 gearing (timber uses theta8 = theta1 - phiTool directly)
    theta8_gain: float = 0.4192
    theta8_offset: float = np.pi / 2.0


class CraneVpstoModel:
    """Crane cost model for VP-STO (ported from timber crane_vpsto_model.cpp).

    Joint partition for 5 actuated DoF:
      fixed      = [0, 4]  → theta1, theta8  (from target)
      independent = [1]    → theta2           (optimized)
      dependent  = [2, 3]  → theta3, q4      (geometry-constrained)
    """

    ndof = 5

    def __init__(self, config: CraneVpstoConfig | None = None, n_eval: int = 20) -> None:
        self.config = config or CraneVpstoConfig()
        self.n_eval = n_eval

        c = self.config
        self._p2 = np.array([-c.a1, c.d1], dtype=float)
        self._p5 = np.zeros(2, dtype=float)  # set by set_yd()
        self._yd = np.zeros(4, dtype=float)

    # ------------------------------------------------------------------ #
    # Target setup
    # ------------------------------------------------------------------ #

    def set_yd(self, yd: np.ndarray) -> None:
        """Set desired target: [x, y, z, yaw].

        This sets p5 (arm-plane target) from the TCP target position,
        same as timber's set_yd().  The arm-plane target is the radial
        distance and height, ignoring the K5→K8 offset (the VP-STO
        optimizer compensates via theta2 optimization).
        """
        self._yd = np.asarray(yd, dtype=float).ravel()
        self._p5 = np.array(
            [np.sqrt(self._yd[0] ** 2 + self._yd[1] ** 2), self._yd[2]],
            dtype=float,
        )

    # ------------------------------------------------------------------ #
    # Joint partition (VP-STO interface)
    # ------------------------------------------------------------------ #

    @staticmethod
    def fixed_joint_indices() -> list[int]:
        return [0, 4]

    @staticmethod
    def independent_joint_indices() -> list[int]:
        return [1]

    @staticmethod
    def dependent_joint_indices() -> list[int]:
        return [2, 3]

    # ------------------------------------------------------------------ #
    # Fixed joints: theta1 and theta8 from target
    # ------------------------------------------------------------------ #

    def compute_fixed_joints(self, q0: np.ndarray, yd: np.ndarray) -> np.ndarray:
        """Compute theta1, theta8 from target (ported from timber).

        Returns [theta1, theta8].
        """
        c = self.config

        # theta1 = atan2(y, x) with wrap handling (same as timber)
        theta1 = float(np.arctan2(yd[1], yd[0]))
        if abs(q0[0] - theta1) > np.pi:
            if theta1 > 0:
                theta1_new = theta1 - 2.0 * np.pi
                if theta1_new > c.q_min[0]:
                    theta1 = theta1_new
            else:
                theta1_new = theta1 + 2.0 * np.pi
                if theta1_new < c.q_max[0]:
                    theta1 = theta1_new

        # theta8 from yaw — CBS gearing (timber: theta8 = theta1 - phiTool)
        phi_tool = float(yd[3])
        theta8 = float((phi_tool - c.theta8_offset - theta1) / c.theta8_gain)
        # Wrap to [-pi, pi]
        theta8 = float(np.arctan2(np.sin(theta8), np.cos(theta8)))

        if abs(q0[4] - theta8) > np.pi:
            theta8 -= 2.0 * np.pi * float(np.sign(theta8 - q0[4]))

        return np.array([theta1, theta8], dtype=float)

    # ------------------------------------------------------------------ #
    # Dependent joints: theta3, q4 from theta2 + target geometry
    # ------------------------------------------------------------------ #

    def compute_dependent_joints(
        self, yd: np.ndarray, q_fixed: np.ndarray, q_independent: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Compute theta3, q4 from theta2 (ported from timber).

        Parameters
        ----------
        yd : target [x, y, z, yaw] (unused here, p5 was set via set_yd)
        q_fixed : [theta1, theta8]
        q_independent : [theta2]

        Returns
        -------
        (q_dependent, penalty)
            q_dependent = [theta3, q4]
            penalty = 0 on success, >0 on infeasibility
        """
        c = self.config
        theta2 = float(q_independent[0])

        # K3 position in arm plane (end of boom link)
        p3 = np.array(
            [c.a2 * np.cos(theta2) - c.a1, c.d1 + c.a2 * np.sin(theta2)],
            dtype=float,
        )

        p32 = self._p2 - p3
        p35 = self._p5 - p3

        p32_norm = float(np.linalg.norm(p32))
        p35_norm = float(np.linalg.norm(p35))

        if p35_norm < c.a3:
            return np.array([np.nan, np.nan], dtype=float), 1e2 * (c.a3 - p35_norm + 1.0)

        # Telescope extension from Pythagoras
        d45 = float(np.sqrt(p35_norm * p35_norm - c.a3 * c.a3))

        # Angle at K3 between p32 and p35
        dot = float(np.dot(p35, p32))
        cos_gamma = np.clip(dot / (p35_norm * p32_norm), -1.0, 1.0)
        gamma = float(np.arccos(cos_gamma))

        theta3 = gamma - 0.5 * np.pi + float(np.arctan2(c.a3, d45))
        q4 = 0.5 * (d45 - c.d4)

        return np.array([theta3, q4], dtype=float), 0.0

    # ------------------------------------------------------------------ #
    # Cost functions
    # ------------------------------------------------------------------ #

    def compute_final_time(
        self,
        q_vec: np.ndarray,
        q_prime_vec: np.ndarray,
        q_pprime_vec: np.ndarray,
    ) -> float:
        """Compute minimum final time from velocity/acceleration limits.

        Ported from timber crane_vpsto_model.cpp:compute_final_time().
        """
        c = self.config
        n = self.ndof
        T_vals: list[float] = []

        for k in range(self.n_eval):
            for i in range(n):
                qp = float(q_prime_vec[k * n + i])
                qpp = float(q_pprime_vec[k * n + i])

                # Velocity bottleneck
                if qp >= 0:
                    T_vals.append(qp / c.q_dot_max[i])
                else:
                    T_vals.append(qp / c.q_dot_min[i])

                # Acceleration bottleneck
                if qpp >= 0:
                    T_vals.append(np.sqrt(qpp / c.q_ddot_max[i]))
                else:
                    T_vals.append(np.sqrt(qpp / c.q_ddot_min[i]))

        return float(max(T_vals)) if T_vals else 1.0

    def compute_joint_limit_penalty(self, q_vec: np.ndarray) -> float:
        """Soft penalty for joint limit violations.

        Ported from timber crane_vpsto_model.cpp:compute_joint_limit_penalty().
        """
        c = self.config
        n = self.ndof
        penalty = 0.0

        for k in range(1, self.n_eval - 1):
            for i in range(n):
                q_ki = float(q_vec[k * n + i])
                if q_ki < c.q_min[i]:
                    penalty += 1e2 * (1.0 + c.q_min[i] - q_ki)
                elif q_ki > c.q_max[i]:
                    penalty += 1e2 * (1.0 + q_ki - c.q_max[i])

        return penalty

    def compute_collision_cost(self, q_vec: np.ndarray, T: float) -> float:
        """Collision cost — stub, returns 0. Override for FCL integration."""
        return 0.0

    def compute_trajectory_cost(
        self,
        q_vec: np.ndarray,
        q_prime_vec: np.ndarray,
        q_pprime_vec: np.ndarray,
    ) -> float:
        """Total trajectory cost (ported from timber).

        cost = T + collision_cost + joint_limit_penalty
        """
        jl_penalty = self.compute_joint_limit_penalty(q_vec)
        T = 0.0
        if jl_penalty < 1e1:
            T = self.compute_final_time(q_vec, q_prime_vec, q_pprime_vec)
        return T + self.compute_collision_cost(q_vec, T) + jl_penalty
