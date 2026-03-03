# Path-Following OCP: Hessian and Dynamics Study Notes

## Problem Statement

We want to understand the tradeoff between:

1. `hessian_approx = GAUSS_NEWTON` vs `hessian_approx = EXACT` in acados
2. `dynamics_mode = split` vs `dynamics_mode = projected` for passive-joint dynamics

for a configuration-space path-following OCP of the crane.

Primary publication question:

- How do these choices affect optimization speed, numerical robustness, and final trajectory quality (including passive-joint swing suppression and replay tracking)?

## Current OCP Formulation (Implemented)

### State and Input

- State:
  - `x = [q, dq, s, sdot]`
- Input:
  - `u = [qdd_actuated, v]`
  - with `v = s_ddot` (path progress acceleration)

### Path Representation

- Configuration-space B-spline path `q_ref(s)`:
  - Implemented in `motion_planning/trajectory/path_following/spline.py`
  - Current default uses clamped cubic with 4 control points (Bezier-equivalent branch)

### Cost Structure (EXTERNAL cost in acados)

- Stage terms include:
  - path tracking in `q`
  - velocity tracking w.r.t. path progression
  - progress terms in `s`, `sdot`
  - control regularization
  - passive-sway penalties (`q_passive`, `dq_passive`)
- Terminal terms include:
  - path tracking
  - velocity penalties
  - progress penalties
  - stronger passive-sway penalties

### Hard Terminal Constraints

- `dq(T) = 0` for all joints
- `sdot(T) = 0`
- `s(T) = 1`

These are currently active and numerically satisfied.

## Relevant Code Locations

- Path-following optimizer:
  - `motion_planning/trajectory/path_following/optimizer.py`
- Spline model:
  - `motion_planning/trajectory/path_following/spline.py`
- Replay:
  - `motion_planning/simulation/mujoco_pd_replay.py`

## Observed Behavior (Current Workspace)

### Hessian Approximation

- With `hessian_approx = EXACT`:
  - no Gauss-Newton warning
  - significantly higher optimization time
  - observed example run:
    - `optimization_solver_time_s` around `47.49`
    - `optimization_wall_time_s` around `51.22`

- With `hessian_approx = GAUSS_NEWTON`:
  - acados warning appears for `EXTERNAL` cost
  - much faster solves
  - observed example run:
    - `optimization_solver_time_s` around `0.263`
    - `optimization_wall_time_s` around `3.39`

### Dynamics Mode

- `dynamics_mode = projected` is now default (requested for accuracy).
- `split` remains available as fallback/ablation mode.

### Terminal Velocity Constraints

- Verified from saved trajectory artifact:
  - `max(abs(dq(T)))` approximately `1e-18` to `1e-13` range
  - `sdot(T)` approximately `1e-18` to `1e-14`
  - `s(T) = 1.0`

## Why GAUSS_NEWTON Warning Appears

The warning is expected when:

- `cost_type = EXTERNAL`
- `hessian_approx = GAUSS_NEWTON`

acados states that in this setup it effectively computes exact cost Hessian contributions (and not full GN treatment for constraints/dynamics). This is acceptable for experiments but must be explicitly documented in publication methodology.

## Reproducibility Commands

### Run OCP

```bash
conda run -n mp_env python motion_planning/trajectory/run_crane_acados_ocp_example.py \
  --traj-out ./crane_acados_ocp_trajectory.npz \
  --plot-out ./crane_acados_ocp_example.png
```

### Replay

```bash
conda run -n mp_env python motion_planning/simulation/mujoco_pd_replay.py \
  --traj ./crane_acados_ocp_trajectory.npz \
  --kp 20 --kd 5 --tail-s 0.2 --no-view
```

### Inspect timing and terminal conditions

```bash
python - <<'PY'
import numpy as np
d = np.load("crane_acados_ocp_trajectory.npz")
print("optimization_wall_time_s", float(d["optimization_wall_time_s"][0]))
print("optimization_solver_time_s", float(d["optimization_solver_time_s"][0]))
print("terminal_dq_abs_max", float(np.max(np.abs(d["dq_trajectory"][-1]))))
print("terminal_s", float(d["s_trajectory"][-1]))
print("terminal_sdot", float(d["sdot_trajectory"][-1]))
PY
```

## Suggested Publication Experiment Matrix

Compare all combinations:

1. `hessian_approx in {GAUSS_NEWTON, EXACT}`
2. `dynamics_mode in {split, projected}`

For each condition, report:

- solve time:
  - wall time
  - solver-reported time
- convergence:
  - status code
  - iteration stats
- trajectory quality:
  - terminal path error
  - terminal velocity norm
  - passive-joint sway metrics
- replay quality:
  - joint RMSE
  - actuator clipping fraction
  - post-horizon residual swing (especially passive joints/gripper)

## Open Questions for Later Investigation

1. Is `GAUSS_NEWTON + EXTERNAL` effectively close enough to EXACT in trajectory quality for this OCP?
2. Which terms dominate the runtime increase with EXACT under projected dynamics?
3. Can an equivalent nonlinear least-squares reformulation avoid the warning while preserving speed?
4. How sensitive are passive-sway results to:
   - `sdot_ref`
   - passive terminal weights
   - horizon length
   - dynamics mode

## Notes

- These notes capture the current implementation state and observed behavior in this repository and environment.
- Keep this file updated as ablation experiments are added.
