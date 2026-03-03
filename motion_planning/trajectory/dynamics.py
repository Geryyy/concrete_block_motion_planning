from __future__ import annotations

from typing import Sequence

import numpy as np


def _submatrix(ca, mat, rows: Sequence[int], cols: Sequence[int]):
    return ca.vertcat(*[ca.hcat([mat[r, c] for c in cols]) for r in rows])


def build_underactuated_qdd_symbolic(
    *,
    ca,
    cpin,
    cmodel,
    cdata,
    model,
    q_pin,
    dq,
    u_qdd,
    act_v_idx: Sequence[int],
    passive_v_idx: Sequence[int],
    dynamics_mode: str,
    passive_solve_damping: float,
    extra_generalized_forces=None,
):
    if dynamics_mode not in {"split", "projected"}:
        raise ValueError(f"Unsupported dynamics_mode '{dynamics_mode}'. Expected 'split' or 'projected'.")

    nv = int(model.nv)
    if extra_generalized_forces is None:
        tau_extra = ca.SX.zeros(nv, 1)
    else:
        tau_extra = extra_generalized_forces

    if dynamics_mode == "split" and len(passive_v_idx) > 0:
        M = cpin.crba(cmodel, cdata, q_pin)
        M = 0.5 * (M + M.T)
        G = cpin.computeGeneralizedGravity(cmodel, cdata, q_pin)
        G = G + tau_extra

        act_idx = list(act_v_idx)
        pas_idx = list(passive_v_idx)
        Mpa = _submatrix(ca, M, pas_idx, act_idx)
        Mpp = _submatrix(ca, M, pas_idx, pas_idx)
        G_p = ca.vertcat(*[G[r] for r in pas_idx])
        d_pas = np.asarray(model.damping, dtype=float)[pas_idx]
        dq_p = ca.vertcat(*[dq[r] for r in pas_idx])
        rhs_p = -(Mpa @ u_qdd + G_p + ca.diag(ca.DM(d_pas)) @ dq_p)
        if passive_solve_damping > 0.0:
            Mpp = Mpp + passive_solve_damping * ca.SX.eye(len(pas_idx))
        qdd_p = ca.solve(Mpp, rhs_p)

        qdd = ca.SX.zeros(nv, 1)
        for i, vi in enumerate(act_idx):
            qdd[vi] = u_qdd[i]
        for i, vi in enumerate(pas_idx):
            qdd[vi] = qdd_p[i]
        return qdd

    # Projected full-model passive acceleration:
    # solve passive accelerations from coupled dynamics while treating
    # non-passive accelerations as known (actuated from input, others 0).
    qdd = ca.SX.zeros(nv, 1)
    for i, vi in enumerate(act_v_idx):
        qdd[vi] = u_qdd[i]
    if len(passive_v_idx) == 0:
        return qdd

    M = cpin.crba(cmodel, cdata, q_pin)
    M = 0.5 * (M + M.T)
    G = cpin.computeGeneralizedGravity(cmodel, cdata, q_pin)
    G = G + tau_extra
    all_idx = list(range(nv))
    pas_idx = list(passive_v_idx)
    non_pas_idx = [i for i in all_idx if i not in set(pas_idx)]

    Mpp = _submatrix(ca, M, pas_idx, pas_idx)
    Mpn = _submatrix(ca, M, pas_idx, non_pas_idx)
    G_p = ca.vertcat(*[G[r] for r in pas_idx])
    d_pas = np.asarray(model.damping, dtype=float)[pas_idx]
    dq_p = ca.vertcat(*[dq[r] for r in pas_idx])
    qdd_known = ca.vertcat(*[qdd[i] for i in non_pas_idx])
    rhs_p = -(Mpn @ qdd_known + G_p + ca.diag(ca.DM(d_pas)) @ dq_p)
    if passive_solve_damping > 0.0:
        Mpp = Mpp + passive_solve_damping * ca.SX.eye(len(pas_idx))
    qdd_p = ca.solve(Mpp, rhs_p)
    for i, vi in enumerate(pas_idx):
        qdd[vi] = qdd_p[i]
    return qdd
