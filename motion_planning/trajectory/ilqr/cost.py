"""Time-varying LQR cost for joint-space tracking.

Running cost (step k):
    l(x, u) = 0.5*(x - x_ref[k])^T Q_run (x - x_ref[k]) + 0.5*u^T R u

Terminal cost:
    lf(x) = 0.5*(x - x_ref[N])^T Qf x

where x_ref[k] = [q_ref[k], dq_ref[k]].
"""

from __future__ import annotations

import numpy as np


class TrackingCost:
    """LQR tracking cost for double-integrator crane dynamics.

    Parameters
    ----------
    n_q:
        Number of joints.
    Q_q, Q_dq:
        Per-joint diagonal weights for position and velocity tracking.
    R:
        Per-joint diagonal control (acceleration) cost.
    Qf_q, Qf_dq:
        Terminal cost weights (position, velocity).
    """

    def __init__(
        self,
        n_q: int = 5,
        Q_q: float | np.ndarray = 10.0,
        Q_dq: float | np.ndarray = 0.1,
        R: float | np.ndarray = 0.01,
        Qf_q: float | np.ndarray = 100.0,
        Qf_dq: float | np.ndarray = 1.0,
    ) -> None:
        self.n_q = n_q
        self.nx = 2 * n_q
        self.nu = n_q

        def _diag(w: float | np.ndarray, n: int) -> np.ndarray:
            w = np.asarray(w, dtype=float)
            return np.diag(np.broadcast_to(w, (n,)))

        self._Qq = _diag(Q_q, n_q)
        self._Qdq = _diag(Q_dq, n_q)
        self._R = _diag(R, n_q)
        self._Qfq = _diag(Qf_q, n_q)
        self._Qfdq = _diag(Qf_dq, n_q)

        self._Q_run = np.block([
            [self._Qq, np.zeros((n_q, n_q))],
            [np.zeros((n_q, n_q)), self._Qdq],
        ])
        self._Qf = np.block([
            [self._Qfq, np.zeros((n_q, n_q))],
            [np.zeros((n_q, n_q)), self._Qfdq],
        ])

    # ------------------------------------------------------------------
    # Gradient / Hessian helpers (analytic, constant)
    # ------------------------------------------------------------------

    def running_lxx(self) -> np.ndarray:
        return self._Q_run.copy()

    def running_luu(self) -> np.ndarray:
        return self._R.copy()

    def terminal_lxx(self) -> np.ndarray:
        return self._Qf.copy()

    def running_lx(self, x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        return self._Q_run @ (x - x_ref)

    def running_lu(self, u: np.ndarray) -> np.ndarray:
        return self._R @ u

    def terminal_lx(self, x: np.ndarray, x_ref_N: np.ndarray) -> np.ndarray:
        return self._Qf @ (x - x_ref_N)

    # ------------------------------------------------------------------

    def running(self, x: np.ndarray, u: np.ndarray, x_ref: np.ndarray) -> float:
        dx = x - x_ref
        return float(0.5 * dx @ self._Q_run @ dx + 0.5 * u @ self._R @ u)

    def terminal(self, x: np.ndarray, x_ref_N: np.ndarray) -> float:
        dx = x - x_ref_N
        return float(0.5 * dx @ self._Qf @ dx)
