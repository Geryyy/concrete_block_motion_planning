from __future__ import annotations

import casadi as ca
import numpy as np


def clamped_uniform_knots(n_ctrl: int, degree: int) -> np.ndarray:
    if n_ctrl <= degree:
        raise ValueError(f"n_ctrl must be > degree, got n_ctrl={n_ctrl}, degree={degree}.")
    n = n_ctrl - 1
    n_knots = n + degree + 2
    knots = np.zeros(n_knots, dtype=float)
    knots[-(degree + 1) :] = 1.0
    interior_count = n - degree
    if interior_count > 0:
        interior = np.linspace(0.0, 1.0, interior_count + 2, dtype=float)[1:-1]
        knots[degree + 1 : degree + 1 + interior_count] = interior
    return knots


def bspline_basis_all_symbolic(s: ca.SX, knots: np.ndarray, degree: int, n_ctrl: int) -> list[ca.SX]:
    if n_ctrl <= 0:
        raise ValueError("n_ctrl must be positive.")
    if degree < 0:
        raise ValueError("degree must be non-negative.")

    s_is_one = ca.fabs(s - 1.0) <= 1e-12
    basis = []
    for i in range(n_ctrl):
        in_left = knots[i] <= s
        in_right = s < knots[i + 1]
        base = ca.if_else(ca.logic_and(in_left, in_right), 1.0, 0.0)
        base = ca.if_else(ca.logic_and(s_is_one, i == (n_ctrl - 1)), 1.0, base)
        basis.append(base)

    for k in range(1, degree + 1):
        next_basis: list[ca.SX] = []
        for i in range(n_ctrl):
            left = 0.0
            right = 0.0
            den_left = float(knots[i + k] - knots[i])
            den_right = float(knots[i + k + 1] - knots[i + 1]) if (i + 1) < n_ctrl else 0.0
            if den_left > 1e-12:
                left = ((s - knots[i]) / den_left) * basis[i]
            if den_right > 1e-12 and (i + 1) < n_ctrl:
                right = ((knots[i + k + 1] - s) / den_right) * basis[i + 1]
            next_basis.append(left + right)
        basis = next_basis
    return basis


def bspline_eval_symbolic(s: ca.SX, control_points: np.ndarray, degree: int) -> ca.SX:
    ctrl = np.asarray(control_points, dtype=float)
    if ctrl.ndim != 2:
        raise ValueError(f"control_points must be 2D, got shape {ctrl.shape}.")
    n_ctrl, dim = ctrl.shape
    # Clamped cubic B-spline with 4 control points equals cubic Bezier.
    # This branch avoids piecewise basis logic and gives smooth derivatives.
    if n_ctrl == 4 and degree == 3:
        t = s
        omt = 1.0 - t
        b0 = omt * omt * omt
        b1 = 3.0 * omt * omt * t
        b2 = 3.0 * omt * t * t
        b3 = t * t * t
        y = (
            b0 * ca.DM(ctrl[0].reshape(dim, 1))
            + b1 * ca.DM(ctrl[1].reshape(dim, 1))
            + b2 * ca.DM(ctrl[2].reshape(dim, 1))
            + b3 * ca.DM(ctrl[3].reshape(dim, 1))
        )
        return y

    knots = clamped_uniform_knots(n_ctrl=n_ctrl, degree=degree)
    basis = bspline_basis_all_symbolic(s=s, knots=knots, degree=degree, n_ctrl=n_ctrl)
    y = ca.SX.zeros(dim, 1)
    for i in range(n_ctrl):
        y += basis[i] * ca.DM(ctrl[i].reshape(dim, 1))
    return y
