"""Linear double-integrator dynamics for iLQR.

State:  x = [q (n_q), dq (n_q)]          shape n_x = 2*n_q
Control: u = qdd (n_q)                    shape n_u = n_q

Dynamics:
    q_new  = q  + Ts*dq + 0.5*Ts²*u
    dq_new = dq + Ts*u

Jacobians are exact (time-invariant linear system).
"""

from __future__ import annotations

import numpy as np


class DoubleIntegratorDynamics:
    def __init__(self, n_q: int = 5, Ts: float = 0.1) -> None:
        self.n_q = n_q
        self.Ts = float(Ts)
        self.nx = 2 * n_q
        self.nu = n_q

        I = np.eye(n_q)
        Z = np.zeros((n_q, n_q))
        self._A = np.block([[I, Ts * I], [Z, I]])
        self._B = np.block([[0.5 * Ts ** 2 * I], [Ts * I]])

    # ------------------------------------------------------------------

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self._A @ x + self._B @ u

    def jacobians(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (A, B) — exact, constant."""
        return self._A, self._B
