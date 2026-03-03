# Underactuated Passive-Acceleration Models

This note summarizes the two underactuated models currently available in
`motion_planning.mechanics.analytic` and how to reproduce their comparison.

## Scope

We consider passive-joint acceleration prediction for:

- passive joints: `theta6_tip_joint`, `theta7_tilt_joint`
- actuated acceleration input: `qdd_a`
- state: `(q, dq)`

The target equation is:

`M(q) qdd + h(q, dq) = tau`

with passive components solved from the coupled dynamics.

## Implementations

### 1) Split Model

Class: `SplitUnderactuatedDynamics`  
File: `motion_planning/mechanics/analytic/split_dynamics.py`

Approach:

- Build a reduced Pinocchio model by locking selected joints.
- Partition reduced dynamics into actuated/passive blocks.
- Solve:
  - `qdd_p = - (Mpp)^-1 (Mpa qdd_a + h_p)`

Pros:

- Compact model, lower dimension.
- Fast and simple.

Cons:

- Approximation error from reduction/locking and tied-joint treatment.
- Sensitive when omitted couplings are relevant.

### 2) Projected Full Model

Class: `ProjectedUnderactuatedDynamics`  
File: `motion_planning/mechanics/analytic/projected_dynamics.py`

Approach:

- Use full model dynamics (URDF or MJCF source).
- Enforce tied and locked joints in full state construction.
- Solve passive accelerations from full-system partition:
  - `qdd_p = - (Mpp)^-1 (Mpn qdd_known + h_p)`

Pros:

- Highest fidelity without data-driven calibration.
- When loaded from the same MJCF as MuJoCo, matches MuJoCo passive
  acceleration numerically (up to solver tolerance).

Cons:

- Higher computational cost than the reduced split model.

## Reproducible Comparison

The previous trajectory comparison script was removed as part of cleanup of
intermediate trajectory examples. The model interpretation below remains valid.

It compared:

1. Split model vs MuJoCo
2. Projected full model vs MuJoCo
3. Pinocchio-from-MJCF baseline vs MuJoCo

Output includes:

- RMSE and normalized RMSE for passive `qdd`
- model-term consistency diagnostics (`Mpp`, `Mpa`, `h_p`)
- plot: `passive_qdd_split_vs_mujoco.png`

## Current Interpretation

With synchronized MJCF (`crane_urdf/crane.xml`) used consistently:

- projected full model ≈ MuJoCo (near numerical precision),
- split model shows residual error due to intentional model reduction.

So the remaining split-model error is structural (reduction choice), not
Pinocchio/MuJoCo core dynamics inconsistency.
