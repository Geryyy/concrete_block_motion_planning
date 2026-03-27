"""Joint-space spline reference built from VP-STO waypoints."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


class JointSplineReference:
    """Cubic spline through VP-STO waypoints, parameterised by time.

    Parameters
    ----------
    q_waypoints:
        Shape (M, n_q) — actuated joint positions from VP-STO.
    T:
        Total trajectory duration [s].
    """

    def __init__(self, q_waypoints: np.ndarray, T: float) -> None:
        q = np.asarray(q_waypoints, dtype=float)
        M, self.n_q = q.shape
        self.T = float(T)
        t = np.linspace(0.0, self.T, M)
        self._cs = CubicSpline(t, q, bc_type="clamped")

    # ------------------------------------------------------------------

    def query(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (q_ref, dq_ref) at time *t*."""
        t_clip = float(np.clip(t, 0.0, self.T))
        return (
            np.asarray(self._cs(t_clip), dtype=float),
            np.asarray(self._cs(t_clip, 1), dtype=float),
        )

    def sample_uniform(self, N: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (q_traj, dq_traj) sampled at *N* uniformly-spaced times."""
        times = np.linspace(0.0, self.T, N)
        q = np.asarray(self._cs(times), dtype=float)
        dq = np.asarray(self._cs(times, 1), dtype=float)
        return q, dq
