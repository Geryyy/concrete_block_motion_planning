"""iLQR solver for linear double-integrator crane dynamics.

Because the dynamics are linear, the backward pass Jacobians are exact
and constant — there is no need to re-linearise on each outer iteration.
The algorithm converges in a single backward pass (classical finite-horizon
LQR with time-varying cost).

Line search uses Armijo backtracking to handle the (potentially non-convex)
running cost when reference deviations are large.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cost import TrackingCost
from .dynamics import DoubleIntegratorDynamics


@dataclass
class ILQRResult:
    success: bool
    message: str
    q_traj: np.ndarray          # (N+1, n_q)
    dq_traj: np.ndarray         # (N+1, n_q)
    qdd_traj: np.ndarray        # (N, n_q)
    time_s: np.ndarray          # (N+1,)
    cost: float
    iterations: int
    diagnostics: dict = field(default_factory=dict)


class ILQRSolver:
    """Finite-horizon iLQR for time-varying LQR with double-integrator dynamics.

    Parameters
    ----------
    dynamics:
        ``DoubleIntegratorDynamics`` instance.
    cost:
        ``TrackingCost`` instance.
    N:
        Number of time steps.
    max_iter:
        Maximum outer iterations (typically 1 for linear systems).
    armijo_c, armijo_beta:
        Line-search parameters.
    reg_init:
        Initial regularisation added to Quu diagonal.
    """

    def __init__(
        self,
        dynamics: DoubleIntegratorDynamics,
        cost: TrackingCost,
        N: int = 60,
        max_iter: int = 20,
        armijo_c: float = 1e-4,
        armijo_beta: float = 0.5,
        reg_init: float = 1e-6,
    ) -> None:
        self.dyn = dynamics
        self.cost = cost
        self.N = N
        self.max_iter = max_iter
        self.armijo_c = armijo_c
        self.armijo_beta = armijo_beta
        self.reg_init = reg_init

    # ------------------------------------------------------------------

    def solve(
        self,
        x0: np.ndarray,
        x_refs: np.ndarray,
    ) -> ILQRResult:
        """Run iLQR from *x0*.

        Parameters
        ----------
        x0:
            Initial state (n_x,).
        x_refs:
            Reference states (N+1, n_x) at each step k=0..N.

        Returns
        -------
        ILQRResult
        """
        N = self.N
        n_x = self.dyn.nx
        n_u = self.dyn.nu
        n_q = self.dyn.n_q
        Ts = self.dyn.Ts
        A, B = self.dyn.jacobians()

        # ---- Initialise trajectory with zero control ----
        x_bar = np.zeros((N + 1, n_x))
        u_bar = np.zeros((N, n_u))
        x_bar[0] = np.asarray(x0, dtype=float)
        for k in range(N):
            x_bar[k + 1] = self.dyn.step(x_bar[k], u_bar[k])

        J_prev = self._total_cost(x_bar, u_bar, x_refs)
        reg = self.reg_init
        iters_done = 0

        for it in range(self.max_iter):
            # ---- Backward pass ----
            K, d = self._backward(x_bar, u_bar, x_refs, A, B, reg)

            # ---- Line search ----
            alpha = 1.0
            for _ in range(16):
                x_new, u_new = self._forward(x_bar, u_bar, K, d, alpha, x0)
                J_new = self._total_cost(x_new, u_new, x_refs)
                expected_improvement = alpha * self._expected_improvement(d, K, x_bar, u_bar, x_refs, A, B)
                if J_new < J_prev - self.armijo_c * max(expected_improvement, 0.0):
                    break
                alpha *= self.armijo_beta
            else:
                # No improvement found — keep current trajectory
                iters_done = it + 1
                break

            x_bar = x_new
            u_bar = u_new
            J_prev = J_new
            iters_done = it + 1

            if np.max(np.abs(d)) < 1e-8:
                break

        q_traj = x_bar[:, :n_q]
        dq_traj = x_bar[:, n_q:]
        qdd_traj = u_bar
        time_s = np.linspace(0.0, N * Ts, N + 1)

        return ILQRResult(
            success=True,
            message=f"iLQR: {iters_done} iters, cost={J_prev:.4f}",
            q_traj=q_traj,
            dq_traj=dq_traj,
            qdd_traj=qdd_traj,
            time_s=time_s,
            cost=float(J_prev),
            iterations=iters_done,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _backward(
        self,
        x_bar: np.ndarray,
        u_bar: np.ndarray,
        x_refs: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        reg: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        N = self.N
        n_x = self.dyn.nx
        n_u = self.dyn.nu

        K = np.zeros((N, n_u, n_x))
        d = np.zeros((N, n_u))

        Vxx = self.cost.terminal_lxx()
        Vx = self.cost.terminal_lx(x_bar[N], x_refs[N])

        lxx = self.cost.running_lxx()
        luu = self.cost.running_luu()
        R_reg = luu + reg * np.eye(n_u)

        for k in range(N - 1, -1, -1):
            lx = self.cost.running_lx(x_bar[k], x_refs[k])
            lu = self.cost.running_lu(u_bar[k])

            Qxx = lxx + A.T @ Vxx @ A
            Quu = R_reg + B.T @ Vxx @ B
            Qux = B.T @ Vxx @ A
            Qx = lx + A.T @ Vx
            Qu = lu + B.T @ Vx

            try:
                L = np.linalg.cholesky(Quu)
            except np.linalg.LinAlgError:
                # Fallback: add more regularisation
                Quu = Quu + 1e-3 * np.eye(n_u)
                L = np.linalg.cholesky(Quu)

            K[k] = -np.linalg.solve(Quu, Qux)
            d[k] = -np.linalg.solve(Quu, Qu)

            Vxx = Qxx + Qux.T @ K[k]
            Vx = Qx + Qux.T @ d[k]

        return K, d

    def _forward(
        self,
        x_bar: np.ndarray,
        u_bar: np.ndarray,
        K: np.ndarray,
        d: np.ndarray,
        alpha: float,
        x0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        N = self.N
        n_x = self.dyn.nx
        n_u = self.dyn.nu

        x_new = np.zeros((N + 1, n_x))
        u_new = np.zeros((N, n_u))
        x_new[0] = np.asarray(x0, dtype=float)

        for k in range(N):
            dx = x_new[k] - x_bar[k]
            u_new[k] = u_bar[k] + K[k] @ dx + alpha * d[k]
            x_new[k + 1] = self.dyn.step(x_new[k], u_new[k])

        return x_new, u_new

    def _total_cost(
        self,
        x_traj: np.ndarray,
        u_traj: np.ndarray,
        x_refs: np.ndarray,
    ) -> float:
        total = 0.0
        for k in range(self.N):
            total += self.cost.running(x_traj[k], u_traj[k], x_refs[k])
        total += self.cost.terminal(x_traj[self.N], x_refs[self.N])
        return total

    def _expected_improvement(
        self,
        d: np.ndarray,
        K: np.ndarray,
        x_bar: np.ndarray,
        u_bar: np.ndarray,
        x_refs: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
    ) -> float:
        total = 0.0
        luu = self.cost.running_luu()
        for k in range(self.N):
            lu = self.cost.running_lu(u_bar[k])
            total += float(lu @ d[k] + 0.5 * d[k] @ luu @ d[k])
        return total
