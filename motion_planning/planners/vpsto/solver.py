"""VP-STO solver — Python port of timber tsc_vpsto.cpp.

Time-Space-Correlated VP-STO (Velocity Profile - Space Time Optimization)
uses CMA-ES to optimize via-point weights for a B-spline trajectory.
The trajectory cost is dominated by the minimum feasible final time T,
computed from velocity/acceleration limits.

Reference: tsc_vpsto.cpp (timber_crane_cpp/vpsto/ros2_vpsto/src/)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .crane_model import CraneVpstoModel


def _get_lambda(ds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Lambda matrices for cubic spline basis (port of getLambda)."""
    Lambda = np.array([
        [ds ** 2 / 2.0, 0.0],
        [-ds ** 3 / 6.0, ds ** 2 / 2.0],
    ], dtype=float)
    Lambda1 = np.array([
        [ds, 0.0],
        [-ds ** 2 / 2.0, ds],
    ], dtype=float)
    Lambda2 = np.array([
        [1.0, 0.0],
        [-ds, 1.0],
    ], dtype=float)
    return Lambda, Lambda1, Lambda2


def get_basis_matrices(
    ndof: int, n_via: int, n_eval: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute B-spline basis matrices Phi, dPhi, ddPhi.

    Ported from tsc_vpsto.cpp::get_basis_matrices().

    Parameters
    ----------
    ndof : number of degrees of freedom
    n_via : number of via-points (including start and end)
    n_eval : number of evaluation points along the path

    Returns
    -------
    Phi, dPhi, ddPhi : basis matrices for position, velocity, acceleration
        Shape: (ndof * n_eval, ndof * (n_via + 2))
        Columns: [q0 | via_1..via_{n_via-2} | qd | dq0 | dqd]
    """
    ds_via = 1.0 / (n_via - 1)
    n_w = n_via + 2  # total weight count: n_via positions + 2 velocities
    n_v = n_via

    b = np.array([0.0, 1.0], dtype=float)

    _, _, Lambda2_ref = _get_lambda(ds_via)

    # M matrix
    M = np.array([
        [12.0 / ds_via ** 3, 6.0 / ds_via ** 2],
        [6.0 / ds_via ** 2, 4.0 / ds_via],
    ], dtype=float)

    # Selection matrices
    Sw = [np.zeros((2, n_w), dtype=float) for _ in range(n_via)]
    Sv = [np.zeros((2, n_v), dtype=float) for _ in range(n_via)]
    for i in range(n_via):
        Sw[i][0, i] = 1.0
        Sv[i][1, i] = 1.0

    Lw = [Lambda2_ref.T @ Sw[i + 1] - Sw[i] for i in range(n_via - 1)]
    Lv = [Lambda2_ref.T @ Sv[i + 1] - Sv[i] for i in range(n_via - 1)]

    # Constraint matrices
    Pw = np.zeros((n_v, n_w), dtype=float)
    Pv = np.zeros((n_v, n_v), dtype=float)
    for i in range(n_via - 2):
        Pw[i, :] = b @ (Lambda2_ref @ M @ Lw[i] - M @ Lw[i + 1])
        Pv[i, :] = b @ (M @ Lv[i + 1] - Lambda2_ref @ M @ Lv[i])
    # Boundary conditions
    Pv[n_v - 2, 0] = 1.0
    Pw[n_v - 2, n_via] = 1.0
    Pv[n_v - 1, n_via - 1] = 1.0
    Pw[n_v - 1, n_via + 1] = 1.0

    P = np.linalg.solve(Pv, Pw)

    # Omega matrices
    Omega = [M @ (Lw[i] + Lv[i] @ P) for i in range(n_via - 1)]

    # Build Phi, dPhi, ddPhi
    Phi = np.zeros((ndof * n_eval, ndof * n_w), dtype=float)
    dPhi = np.zeros((ndof * n_eval, ndof * n_w), dtype=float)
    ddPhi = np.zeros((ndof * n_eval, ndof * n_w), dtype=float)

    I_ndof = np.eye(ndof, dtype=float)
    ds_eval = 1.0 / max(n_eval - 1, 1)

    for i in range(n_eval):
        s_eval = i * ds_eval
        s_start = 0.0
        for j in range(n_via - 1):
            if s_eval <= s_start + ds_via + 1e-8:
                dsk = s_eval - s_start
                cq = np.zeros(n_w, dtype=float)
                cq[j] = 1.0
                cv = np.zeros(n_v, dtype=float)
                cv[j] = 1.0

                Lambda_k, Lambda1_k, Lambda2_k = _get_lambda(dsk)

                phi = cq + dsk * cv @ P + b @ Lambda_k @ Omega[j]
                dphi = cv @ P + b @ Lambda1_k @ Omega[j]
                ddphi = b @ Lambda2_k @ Omega[j]

                Phi[i * ndof:(i + 1) * ndof, :] = np.kron(phi, I_ndof)
                dPhi[i * ndof:(i + 1) * ndof, :] = np.kron(dphi, I_ndof)
                ddPhi[i * ndof:(i + 1) * ndof, :] = np.kron(ddphi, I_ndof)
                break
            s_start += ds_via

    return Phi, dPhi, ddPhi


@dataclass
class VpstoResult:
    """Result of VP-STO solve."""
    success: bool
    q_via: np.ndarray       # via-point weights: (n_via-2)*ndof
    q_goal: np.ndarray      # goal joint config: ndof
    T: float                # final time
    cost: float             # best cost
    iterations: int         # CMA-ES generations
    q_traj: np.ndarray      # sampled trajectory: (N, ndof)
    dq_traj: np.ndarray     # sampled velocities: (N, ndof)
    t_traj: np.ndarray      # time stamps: (N,)


class VpstoSolver:
    """VP-STO trajectory optimizer using CMA-ES.

    Ported from timber tsc_vpsto.cpp.
    """

    def __init__(
        self,
        model: CraneVpstoModel,
        n_via: int = 5,
        n_eval: int = 20,
        n_samples: int = 64,
        sigma_init: float = 0.5,
        cma_tol_fun_hist: float = 1e-2,
    ) -> None:
        self.model = model
        self.ndof = model.ndof
        self.n_via = n_via
        self.n_eval = n_eval
        self.n_samples = n_samples
        self.sigma_init = sigma_init
        self.cma_tol_fun_hist = cma_tol_fun_hist

        # Build basis matrices
        self.Phi, self.dPhi, self.ddPhi = get_basis_matrices(
            self.ndof, n_via, n_eval,
        )

        # Joint partition indices
        self.fixed_idx = model.fixed_joint_indices()
        self.indep_idx = model.independent_joint_indices()
        self.dep_idx = model.dependent_joint_indices()

        # Compute prior (from tsc_vpsto.cpp constructor lines 34-58)
        n_w = n_via + 2
        n_interior = (n_via - 2) * self.ndof + len(self.indep_idx)

        # ddPhi_p: columns for interior via-points + independent terminal joints
        ddPhi_p = np.zeros((self.ddPhi.shape[0], n_interior), dtype=float)
        # Interior via-point columns: ddPhi[:, ndof : ndof + (n_via-2)*ndof]
        n_via_cols = (n_via - 2) * self.ndof
        ddPhi_p[:, :n_via_cols] = self.ddPhi[:, self.ndof:self.ndof + n_via_cols]
        # Independent terminal joint columns
        p_idx = n_via_cols
        for ind_i in self.indep_idx:
            col = self.ddPhi.shape[1] - 3 * self.ndof + ind_i
            ddPhi_p[:, p_idx] = self.ddPhi[:, col]
            p_idx += 1

        # ddPhi_b: columns for q0 + fixed terminal joints + velocity boundaries
        n_boundary = 3 * self.ndof + len(self.fixed_idx)
        ddPhi_b = np.zeros((self.ddPhi.shape[0], n_boundary), dtype=float)
        # q0 columns
        ddPhi_b[:, :self.ndof] = self.ddPhi[:, :self.ndof]
        # Fixed terminal joint columns
        b_idx = self.ndof
        for fix_i in self.fixed_idx:
            col = self.ddPhi.shape[1] - 3 * self.ndof + fix_i
            ddPhi_b[:, b_idx] = self.ddPhi[:, col]
            b_idx += 1
        # Velocity boundary columns (last 2*ndof cols of ddPhi)
        ddPhi_b[:, b_idx:] = self.ddPhi[:, -2 * self.ndof:]

        # Prior covariance
        S = np.linalg.inv(ddPhi_p.T @ ddPhi_p / n_eval)
        self.ddPhi_p_pinv = S @ ddPhi_p.T / n_eval
        self.S_chol = np.linalg.cholesky(S)
        self.ddPhi_b = ddPhi_b

    def _make_via_prior(
        self, q0: np.ndarray, qd: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute prior mean and Cholesky for via-point-only search.

        Used when the full goal joint config ``qd`` is known in advance,
        so CMA-ES only optimises the interior via-points (not the terminal
        theta2).  Returns (mu_p, S_chol) with search dimension
        ``(n_via - 2) * ndof``.
        """
        ndof = self.ndof
        n_via = self.n_via
        n_via_cols = (n_via - 2) * ndof
        qd_col_start = ndof + n_via_cols

        # Optimized columns: interior via-points only
        ddPhi_p = self.ddPhi[:, ndof:ndof + n_via_cols]

        # Boundary columns: q0 + all qd + dq0=0 + dqd=0
        ddPhi_b = np.zeros((self.ddPhi.shape[0], 4 * ndof), dtype=float)
        ddPhi_b[:, :ndof] = self.ddPhi[:, :ndof]
        ddPhi_b[:, ndof:2 * ndof] = self.ddPhi[:, qd_col_start:qd_col_start + ndof]
        ddPhi_b[:, 2 * ndof:] = self.ddPhi[:, -2 * ndof:]

        base = np.zeros(4 * ndof, dtype=float)
        base[:ndof] = q0
        base[ndof:2 * ndof] = qd
        # dq0 = dqd = 0 (already)

        S = np.linalg.inv(ddPhi_p.T @ ddPhi_p / self.n_eval)
        ddPhi_p_pinv = S @ ddPhi_p.T / self.n_eval
        S_chol = np.linalg.cholesky(S)
        mu_p = -ddPhi_p_pinv @ ddPhi_b @ base
        return mu_p, S_chol

    def solve(
        self,
        q0: np.ndarray,
        yd: np.ndarray,
        q_goal: np.ndarray | None = None,
    ) -> VpstoResult:
        """Run VP-STO optimization.

        Parameters
        ----------
        q0 : initial actuated joint config (ndof,)
        yd : target [x, y, z, yaw] (4,)
        q_goal : optional validated goal joint config (ndof,).  When provided
            the full goal config is fixed and CMA-ES only optimises interior
            via-points, bypassing the geometric goal computation.  This avoids
            the CBS K5→K8 offset error in ``compute_dependent_joints``.

        Returns
        -------
        VpstoResult
        """
        try:
            import cma
        except ImportError:
            raise ImportError("VP-STO requires the 'cma' package: pip install cma")

        q0 = np.asarray(q0, dtype=float).ravel()
        yd = np.asarray(yd, dtype=float).ravel()

        # Pre-compute boundary contributions (q0 part, same regardless of mode)
        self._q0 = q0.copy()
        self._q_boundary = self.Phi[:, :self.ndof] @ q0
        self._qp_boundary = self.dPhi[:, :self.ndof] @ q0
        self._qpp_boundary = self.ddPhi[:, :self.ndof] @ q0

        if q_goal is not None:
            # --- Fixed-goal mode: bypass geometry, optimise via-points only ---
            qd = np.asarray(q_goal, dtype=float).ravel()
            mu_p, S_chol_run = self._make_via_prior(q0, qd)
            N = (self.n_via - 2) * self.ndof
            self._mu_p = mu_p
            self._qd_fixed = qd
        else:
            # --- Original mode: geometry-based goal, optimise theta2 + via-pts ---
            self.model.set_yd(yd)
            qF = self.model.compute_fixed_joints(q0, yd)
            self._qF = qF.copy()
            self._yd = yd.copy()
            self._qd_fixed = None

            base = np.zeros(q0.size + qF.size + 2 * self.ndof, dtype=float)
            base[:q0.size] = q0
            base[q0.size:q0.size + qF.size] = qF
            self._mu_p = -self.ddPhi_p_pinv @ self.ddPhi_b @ base
            N = (self.n_via - 2) * self.ndof + len(self.indep_idx)
            S_chol_run = self.S_chol

        x0 = np.zeros(N, dtype=float)
        opts = cma.CMAOptions()
        opts["popsize"] = self.n_samples
        opts["tolfunhist"] = self.cma_tol_fun_hist
        opts["verbose"] = -9
        opts["seed"] = 42

        self._S_chol_run = S_chol_run
        es = cma.CMAEvolutionStrategy(x0, self.sigma_init, opts)
        while not es.stop():
            solutions = es.ask()
            costs = [self._comp_cost(np.asarray(x, dtype=float)) for x in solutions]
            es.tell(solutions, costs)

        x_best = np.asarray(es.result.xbest, dtype=float)
        x_vec = self._mu_p + S_chol_run @ x_best

        if q_goal is not None:
            # Fixed-goal: x_vec is purely via-point weights
            q_via = x_vec
            qd = self._qd_fixed
        else:
            # Original: extract via-points and terminal theta2
            n_via_weights = (self.n_via - 2) * self.ndof
            q_via = x_vec[:n_via_weights]
            q_indep = x_vec[n_via_weights:]

            qD, dep_cost = self.model.compute_dependent_joints(yd, self._qF, q_indep)
            if dep_cost > 0:
                return VpstoResult(
                    success=False, q_via=q_via, q_goal=np.zeros(self.ndof),
                    T=0.0, cost=float("inf"), iterations=es.result.iterations,
                    q_traj=np.empty((0, self.ndof)), dq_traj=np.empty((0, self.ndof)),
                    t_traj=np.empty(0),
                )
            qd = np.zeros(self.ndof, dtype=float)
            for i, fi in enumerate(self.fixed_idx):
                qd[fi] = self._qF[i]
            for i, ii in enumerate(self.indep_idx):
                qd[ii] = q_indep[i]
            for i, di in enumerate(self.dep_idx):
                qd[di] = qD[i]

        # Compute final time and sample trajectory
        q_vec, qp_vec, qpp_vec = self._get_path(q_via, qd)
        T = self.model.compute_final_time(q_vec, qp_vec, qpp_vec)
        dt = 0.02  # 50 Hz
        q_traj, dq_traj, t_traj = self.sample_trajectory(q_via, qd, T, dt)

        return VpstoResult(
            success=True,
            q_via=q_via,
            q_goal=qd,
            T=T,
            cost=float(es.result.fbest),
            iterations=int(es.result.iterations),
            q_traj=q_traj,
            dq_traj=dq_traj,
            t_traj=t_traj,
        )

    def _comp_cost(self, x: np.ndarray) -> float:
        """Evaluate trajectory cost for a CMA-ES sample."""
        x_vec = self._mu_p + self._S_chol_run @ x

        if self._qd_fixed is not None:
            # Fixed-goal mode: x_vec contains only via-point weights
            q_via = x_vec
            qd = self._qd_fixed
        else:
            # Original mode: via-points + terminal theta2
            n_via_weights = (self.n_via - 2) * self.ndof
            q_via = x_vec[:n_via_weights]
            q_indep = x_vec[n_via_weights:]

            qD, dep_cost = self.model.compute_dependent_joints(self._yd, self._qF, q_indep)
            if dep_cost > 0:
                return dep_cost

            qd = np.zeros(self.ndof, dtype=float)
            for i, fi in enumerate(self.fixed_idx):
                qd[fi] = self._qF[i]
            for i, ii in enumerate(self.indep_idx):
                qd[ii] = q_indep[i]
            for i, di in enumerate(self.dep_idx):
                qd[di] = qD[i]

        q_vec, qp_vec, qpp_vec = self._get_path(q_via, qd)
        return self.model.compute_trajectory_cost(q_vec, qp_vec, qpp_vec)

    def _get_path(
        self, q_via: np.ndarray, qd: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute path from via-points and goal (port of getPath)."""
        n_via_weights = (self.n_via - 2) * self.ndof
        Phi_via = self.Phi[:, self.ndof:self.ndof + n_via_weights]
        Phi_qd = self.Phi[:, self.ndof + n_via_weights:self.ndof + n_via_weights + self.ndof]

        q_vec = self._q_boundary + Phi_via @ q_via + Phi_qd @ qd
        qp_vec = self._qp_boundary + self.dPhi[:, self.ndof:self.ndof + n_via_weights] @ q_via + self.dPhi[:, self.ndof + n_via_weights:self.ndof + n_via_weights + self.ndof] @ qd
        qpp_vec = self._qpp_boundary + self.ddPhi[:, self.ndof:self.ndof + n_via_weights] @ q_via + self.ddPhi[:, self.ndof + n_via_weights:self.ndof + n_via_weights + self.ndof] @ qd

        return q_vec, qp_vec, qpp_vec

    def sample_trajectory(
        self, q_via: np.ndarray, qd: np.ndarray, T: float, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample trajectory at regular time intervals.

        Ported from tsc_vpsto.cpp::sampleTrajectory().

        Returns
        -------
        q_traj : (N, ndof) joint positions
        dq_traj : (N, ndof) joint velocities
        t_traj : (N,) time stamps
        """
        N = int(np.ceil(T / dt)) + 1
        T_sampled = (N - 1) * dt

        Phi_s, dPhi_s, ddPhi_s = get_basis_matrices(self.ndof, self.n_via, N)

        # Weight vector: [q0, via_points, qd, dq0=0, dqd=0]
        n_w = self.n_via + 2
        w = np.zeros(self.ndof * n_w, dtype=float)
        w[:self.ndof] = self._q0
        n_via_weights = (self.n_via - 2) * self.ndof
        w[self.ndof:self.ndof + n_via_weights] = q_via
        w[self.ndof + n_via_weights:self.ndof + n_via_weights + self.ndof] = qd
        # dq0 and dqd are zeros (already)

        q_flat = Phi_s @ w
        dq_flat = dPhi_s @ w / T_sampled
        # ddq_flat = ddPhi_s @ w / (T_sampled ** 2)

        q_traj = q_flat.reshape(N, self.ndof)
        dq_traj = dq_flat.reshape(N, self.ndof)
        t_traj = np.linspace(0.0, T_sampled, N, dtype=float)

        # Enforce exact goal at end
        q_traj[-1] = qd
        dq_traj[-1] = 0.0

        return q_traj, dq_traj, t_traj
