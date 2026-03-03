# Cartesian Path-Following OCP: Solver Design Notes

## Background

The Cartesian path-following optimizer (`CartesianPathFollowingOptimizer`) solves a
nonlinear optimal control problem (OCP) in which a robotic crane tracks a reference
path expressed as a B-spline in task space. The core difficulty is that the tracking
cost involves symbolic forward kinematics (FK):

```
cost ∝ ‖FK(q) − xyz_ref(s)‖²
```

where `FK(q)` is the TCP (tool-center-point) position obtained from symbolic Pinocchio
CasADi kinematics, and `xyz_ref(s)` is a B-spline evaluated at the progress state `s`.
The OCP is transcribed and solved with acados using an SQP outer loop and HPIPM as the
inner QP solver.

---

## Problem: MINSTEP Failures with EXTERNAL Cost + EXACT Hessian

### Initial formulation

The first implementation expressed the full running and terminal costs as scalar CasADi
expressions assigned to `model.cost_expr_ext_cost` / `model.cost_expr_ext_cost_e`, with
`ocp.cost.cost_type = "EXTERNAL"` and `ocp.solver_options.hessian_approx = "EXACT"`.

```python
l_cost = xyz_w * ca.dot(xyz_err, xyz_err) + s_w * (1-s)**2 + ...
ac_model.cost_expr_ext_cost = l_cost
ocp.cost.cost_type = "EXTERNAL"
ocp.solver_options.hessian_approx = "EXACT"
```

### Observed failure

For small joint motions (Δθ₁ ≈ 0.3 rad), this formulation converged reliably.
For larger motions (Δθ₁ ≥ 1.5 rad), every solve failed immediately:

```
QP solver returned error status 3 (ACADOS_MINSTEP) in SQP iteration 1, QP iteration 8
```

Status 4 (`MINSTEP`) means the HPIPM line search could not find a valid descent step,
even at minimum step size.

Attempts that did **not** resolve the issue:
- Increasing horizon length (`N=80`, `Tf=12 s`)
- Setting `hessian_approx = "GAUSS_NEWTON"` — acados emits a warning that with
  `EXTERNAL` cost type the exact Hessian is always computed regardless of this flag
- Enabling `globalization = "MERIT_BACKTRACKING"` (was already set)
- Relaxing QP tolerances

### Root cause analysis

For an EXTERNAL scalar cost `L(x, u)`, acados computes the exact second-order Hessian:

```
H = d²L/d(x,u)²
```

For the Cartesian tracking cost `L = w · ‖FK(q) − r(s)‖²`, expanding the Hessian gives:

```
H = 2w · Jᵀ J  +  2w · Σᵢ eᵢ · d²FKᵢ/dq²
```

where `J = dFK/dq` is the geometric Jacobian and `eᵢ = FKᵢ(q) − rᵢ(s)` is the i-th
component of the Cartesian error.

The first term `2w Jᵀ J` is always positive semi-definite (PSD). The second term
`2w Σᵢ eᵢ · d²FKᵢ/dq²` is the contraction of the FK Hessian with the residual vector.
For a nonlinear kinematic chain, `d²FK/dq²` is non-zero, and when the residual `e` is
large — as it is for the linear joint-space warm-start far from the Cartesian path — this
term dominates and makes `H` **indefinite** (negative eigenvalues).

HPIPM requires the Hessian to be at least positive semi-definite to guarantee a valid QP
solution. An indefinite Hessian means the QP is non-convex, the interior-point method
loses its guaranteed descent direction, and the line search fails with MINSTEP.

For small motions, the initial guess is close to the solution, the residuals are small,
and the second-order term stays negligible. For large motions, the initial linear
interpolation places the trajectory far from the Cartesian path, the residuals are large,
and the Hessian becomes indefinite at the very first SQP iteration.

---

## Fix: NONLINEAR_LS Cost + GAUSS_NEWTON Hessian

### Change

The cost is reformulated as a least-squares residual vector rather than a scalar:

```python
# Running residuals
y_expr = ca.vertcat(
    xyz_err,            # (3,)  — Cartesian tracking error
    1.0 - s,           # (1,)  — progress-to-goal
    sdot - sdot_ref,   # (1,)  — reference speed tracking
    u_qdd,             # (n_act,) — actuator effort
    v,                 # (1,)  — progress acceleration
    q_passive_err,     # (n_pas,) — passive joint anti-sway position
    dq_passive,        # (n_pas,) — passive joint anti-sway velocity
)
W = diag([xyz_w (×3), s_w, sdot_w, u_w (×n_act), v_w, passive_q_w (×n_pas), ...])

ac_model.cost_y_expr    = y_expr      # running cost residual
ac_model.cost_y_expr_e  = y_e_expr    # terminal cost residual
ocp.cost.cost_type      = "NONLINEAR_LS"
ocp.cost.cost_type_e    = "NONLINEAR_LS"
ocp.cost.W              = W           # diagonal weight matrix
ocp.cost.yref           = zeros(ny)   # reference embedded in y_expr itself
ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
```

The reference `xyz_ref(s)` is embedded directly into the residual expression
`y_expr = xyz_tcp − xyz_ref_s`, so `yref = 0` and no per-node reference updates are
needed even though the reference depends on the state `s`.

### Why this works

With `NONLINEAR_LS` + `GAUSS_NEWTON`, acados approximates the Hessian as:

```
H_GN = Jᵀ W J
```

where `J = dy_expr/d(x, u)` is the Jacobian of the residual vector. `Jᵀ W J` is
always **positive semi-definite** by construction (for any `W ≥ 0`), because:

```
vᵀ (Jᵀ W J) v = (Jv)ᵀ W (Jv) ≥ 0  ∀v
```

This drops the second-order FK terms entirely. The approximation is exact at the
solution (where residuals vanish), which preserves local convergence guarantees.
Far from the solution — exactly where the large-motion warm-start starts — the
Gauss-Newton approximation remains well-conditioned, giving HPIPM a valid convex QP
at every SQP iteration.

### Effect on `GAUSS_NEWTON` flag under `EXTERNAL` cost

Importantly, setting `hessian_approx = "GAUSS_NEWTON"` has **no effect** when
`cost_type = "EXTERNAL"` — acados always computes the exact Hessian for scalar external
costs and emits a deprecation warning. The NONLINEAR_LS reformulation is what actually
enables the Gauss-Newton approximation.

---

## Terminal cost structure

```python
y_e_expr = ca.vertcat(
    xyz_err,         # (3,)    — Cartesian error at final node
    dq,              # (nv,)   — joint velocity (penalises residual motion)
    1.0 - s,         # (1,)    — incomplete progress penalty
    sdot,            # (1,)    — residual speed at terminal node
    q_passive_err,   # (n_pas,) — anti-sway position at terminal
    dq_passive,      # (n_pas,) — anti-sway velocity at terminal
)
```

Terminal hard equality constraints (`s=1`, `ṡ=0`, `dq=0`) are enforced separately via
`ocp.constraints.idxbx_e`. The soft terminal cost terms are retained for robustness
during intermediate SQP iterations before the constraints become feasible.

---

## Validation

| Test case | Δθ₁ (slewing) | Horizon | Result (EXTERNAL+EXACT) | Result (NONLINEAR_LS+GN) |
|-----------|---------------|---------|-------------------------|--------------------------|
| Small motion | 0.3 rad | N=40, Tf=4 s | ✓ status=0 | ✓ status=0 |
| Explicit ctrl pts | 0.3 rad | N=40, Tf=4 s | ✓ status=0 | ✓ status=0 |
| Large motion | 1.5 rad | N=80, Tf=8 s | ✗ MINSTEP (status=4) | ✓ status=0 |

All three solver tests are in `motion_planning/tests/test_cartesian_path_following.py`.

---

## Summary of acados options

| Option | Old value | New value | Reason for change |
|--------|-----------|-----------|-------------------|
| `cost_type` / `cost_type_e` | `"EXTERNAL"` | `"NONLINEAR_LS"` | Enable Gauss-Newton Hessian |
| `hessian_approx` | `"EXACT"` | `"GAUSS_NEWTON"` | PSD Hessian `Jᵀ W J` for robustness |
| `model.cost_expr_ext_cost` | scalar CasADi expr | *(removed)* | — |
| `model.cost_y_expr` | *(absent)* | residual vector | NONLINEAR_LS interface |
| `ocp.cost.W` | *(absent)* | `np.diag(w_diag)` | Weight matrix for NONLINEAR_LS |
| `ocp.cost.yref` | *(absent)* | `np.zeros(ny)` | Reference embedded in residual |
| `globalization` | conditional | always `"MERIT_BACKTRACKING"` | Robust line search unconditionally |
