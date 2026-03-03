# Mechanics Package

This package provides crane mechanics utilities focused on:

- model/config loading from URDF,
- inverse kinematics (analytic-first, numeric fallback),
- steady-state computation (workspace target -> joint state),
- MuJoCo-based visualization/examples.

## Structure

- `analytic/config.py`
  - `AnalyticModelConfig`: loads joint sets, tied joints, gravity, URDF path from YAML.
- `analytic/model_description.py`
  - `ModelDescription`: Pinocchio model wrapper and model/frame introspection.
- `analytic/inverse_kinematics.py`
  - `AnalyticIKSolver`: analytic IK for crane native frame pair.
  - `NumericIKSolver`: least-squares IK fallback.
  - `AnalyticInverseKinematics`: facade that tries analytic first, then numeric.
  - `IkSolveResult`: unified result format (`success`, `status`, errors, joint maps).
- `analytic/steady_state.py`
  - `CraneSteadyState`: computes actuated IK solution + passive static equilibrium.
- `analytic/split_dynamics.py`
  - `SplitUnderactuatedDynamics`: reduced split passive-acceleration model.
- `analytic/projected_dynamics.py`
  - `ProjectedUnderactuatedDynamics`: full projected passive-acceleration model.
- `analytic/pinocchio_utils.py`
  - shared Pinocchio helpers (joint limits, q map conversion, FK transform).
- `analytic/crane_geometry.py`
  - centralized geometry constants used by analytic IK.
- Runtime package does not include MuJoCo example scripts.
  - keep examples in top-level `examples/` to avoid coupling core mechanics to simulation tooling.

## Public API

Primary imports:

- `AnalyticModelConfig`
- `ModelDescription`
- `AnalyticIKSolver`
- `NumericIKSolver`
- `AnalyticInverseKinematics`
- `IkSolveResult`
- `CraneSteadyState`
- `SteadyStateResult`
- `SplitUnderactuatedDynamics`
- `SplitPassiveAccelResult`
- `ProjectedUnderactuatedDynamics`
- `PassiveAccelResult`

## IK Behavior

`AnalyticInverseKinematics.solve_pose(...)`:

1. tries `AnalyticIKSolver`,
2. if unavailable/invalid and `force_analytic=False`, uses `NumericIKSolver`,
3. if `force_analytic=True`, returns failure instead of fallback.

`IkSolveResult.status` values:

- `analytic_success`
- `numeric_success`
- `numeric_residual_success`
- `analytic_failed`
- `failed`

## When Analytic IK Fails (Fallback Needed)

Typical reasons:

1. Frame pair mismatch
   - Analytic solver is implemented for:
     - base: `K0_mounting_base`
     - end: `K8_tool_center_point`
2. Actuated set mismatch
   - Analytic solver expects the crane 5 actuated joints:
     `theta1`, `theta2`, `theta3`, `q4`, `theta8`.
3. Geometric infeasibility
   - closed-form dependent-joint geometry has no valid solution.
4. Joint-limit rejection
   - all analytic candidates violate limits.
5. Search resolution miss
   - independent joint (`theta2`) is searched at 1-degree resolution.
6. Full SE(3) mismatch vs reduced geometric assumptions
   - analytic branch can be valid geometrically but fail strict full-pose residual checks.

## Notes

- `solve_passive` is intentionally removed from IK API.
  - passive equilibrium is handled in `CraneSteadyState` (gravity-balance solve).
- Forward kinematics in IK evaluation uses Pinocchio (`pinocchio_utils.fk_homogeneous`).
- Legacy symbolic kinematics/dynamics code has been removed from active mechanics API.

## Underactuated Model Note

Detailed split-vs-projected model summary (equations, assumptions, and
comparison context):

- `analytic/UNDERACTUATED_MODELS.md`

Reproducible comparison script: removed from `motion_planning/trajectory` as part
of trajectory-example cleanup.

## Minimal Usage

```python
from motion_planning.mechanics.analytic import (
    AnalyticModelConfig,
    ModelDescription,
    AnalyticInverseKinematics,
)

cfg = AnalyticModelConfig.default()
desc = ModelDescription(cfg)
ik = AnalyticInverseKinematics(desc, cfg)

res = ik.solve_pose(
    target_T_base_to_end=T_target,
    base_frame="K0_mounting_base",
    end_frame="K8_tool_center_point",
    q_seed=q_seed_map,
    force_analytic=False,
)
```
