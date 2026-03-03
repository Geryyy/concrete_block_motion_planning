from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PDController:
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

    def compute(self, q_des: np.ndarray, dq_des: np.ndarray, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        q_des = np.asarray(q_des, dtype=float).reshape(-1)
        dq_des = np.asarray(dq_des, dtype=float).reshape(-1)
        q = np.asarray(q, dtype=float).reshape(-1)
        dq = np.asarray(dq, dtype=float).reshape(-1)
        if q_des.shape != self.kp.shape or dq_des.shape != self.kp.shape:
            raise ValueError("q_des and dq_des must match controller gain vector size.")
        if q.shape != self.kp.shape or dq.shape != self.kp.shape:
            raise ValueError("q and dq must match controller gain vector size.")
        u = self.kp * (q_des - q) + self.kd * (dq_des - dq)
        if self.u_min is not None:
            u = np.maximum(u, self.u_min)
        if self.u_max is not None:
            u = np.minimum(u, self.u_max)
        return u
