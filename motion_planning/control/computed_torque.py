from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ComputedTorqueController:
    """Computed-torque helper with acceleration feedback and actuator clipping."""

    kp: np.ndarray
    kd: np.ndarray
    u_min: np.ndarray | None = None
    u_max: np.ndarray | None = None

    def __post_init__(self) -> None:
        kp = np.asarray(self.kp, dtype=float).reshape(-1)
        kd = np.asarray(self.kd, dtype=float).reshape(-1)
        if kp.shape != kd.shape:
            raise ValueError(f"kp and kd must have identical shapes, got {kp.shape} vs {kd.shape}.")
        object.__setattr__(self, "kp", kp)
        object.__setattr__(self, "kd", kd)

        if self.u_min is not None:
            u_min = np.asarray(self.u_min, dtype=float).reshape(-1)
            if u_min.shape != kp.shape:
                raise ValueError(f"u_min shape {u_min.shape} must match kp shape {kp.shape}.")
            object.__setattr__(self, "u_min", u_min)
        if self.u_max is not None:
            u_max = np.asarray(self.u_max, dtype=float).reshape(-1)
            if u_max.shape != kp.shape:
                raise ValueError(f"u_max shape {u_max.shape} must match kp shape {kp.shape}.")
            object.__setattr__(self, "u_max", u_max)

    def compute_acceleration(
        self,
        *,
        q_des: np.ndarray,
        dq_des: np.ndarray,
        q: np.ndarray,
        dq: np.ndarray,
        qdd_ff: np.ndarray | None = None,
    ) -> np.ndarray:
        q_des = np.asarray(q_des, dtype=float).reshape(-1)
        dq_des = np.asarray(dq_des, dtype=float).reshape(-1)
        q = np.asarray(q, dtype=float).reshape(-1)
        dq = np.asarray(dq, dtype=float).reshape(-1)
        if q_des.shape != self.kp.shape or dq_des.shape != self.kp.shape:
            raise ValueError("q_des and dq_des must match gain vector size.")
        if q.shape != self.kp.shape or dq.shape != self.kp.shape:
            raise ValueError("q and dq must match gain vector size.")
        qdd_cmd = self.kp * (q_des - q) + self.kd * (dq_des - dq)
        if qdd_ff is not None:
            qdd_ff = np.asarray(qdd_ff, dtype=float).reshape(-1)
            if qdd_ff.shape != self.kp.shape:
                raise ValueError("qdd_ff must match gain vector size.")
            qdd_cmd = qdd_ff + qdd_cmd
        return qdd_cmd

    def torque_to_control(self, *, tau: np.ndarray, gear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tau = np.asarray(tau, dtype=float).reshape(-1)
        gear = np.asarray(gear, dtype=float).reshape(-1)
        if tau.shape != self.kp.shape or gear.shape != self.kp.shape:
            raise ValueError("tau and gear must match gain vector size.")
        safe_gear = np.where(np.abs(gear) > 1e-12, gear, 1.0)
        u = tau / safe_gear
        clipped = np.zeros_like(u, dtype=bool)
        if self.u_min is not None:
            clipped = clipped | (u < self.u_min)
            u = np.maximum(u, self.u_min)
        if self.u_max is not None:
            clipped = clipped | (u > self.u_max)
            u = np.minimum(u, self.u_max)
        return u, clipped
