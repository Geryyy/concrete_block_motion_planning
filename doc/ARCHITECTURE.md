# concrete_block_motion_planning Architecture

This package is organized as a planning-only ROS 2 node with explicit separation between:
- API layer (ROS service handlers)
- Planning/runtime integration
- Configuration loading
- In-memory planning state

## Module Layout

- `scripts/motion_planning_node.py`
  - Thin executable entrypoint.
  - Delegates to `cbmp.node.main()`.

- `scripts/cbmp/node.py`
  - Node composition and lifecycle orchestration.
  - Loads config, initializes runtime/data, registers services, logs startup.

- `scripts/cbmp/config.py`
  - Centralized parameter declaration and typed config loading (`NodeConfig`).
  - Single source of truth for runtime tunables.

- `scripts/cbmp/state.py`
  - In-memory state container (`MotionPlanningState`).
  - Tracks plans, trajectories, wall-plan progress, runtime caches.
  - Contains dedicated `RuntimeStatus` for backend readiness and reasons.

- `scripts/cbmp/services.py`
  - Service request/response handling logic.
  - Implements orchestration of geometric planning + trajectory generation.

- `scripts/cbmp/runtime.py`
  - Runtime helpers and planner backend integration.
  - Geometry planning, IK/steady-state, optimizer setup, yaml loading, frame/math utils.

- `scripts/cbmp/types.py`
  - Shared dataclasses for stored plans/trajectories and wall-plan tasks.

## Design Intent

1. Keep ROS plumbing separate from planning internals.
2. Keep parameters explicit and typed.
3. Keep mutable runtime state centralized.
4. Keep service APIs stable while allowing backend evolution.

## Current Planning Pipeline

1. Task-space geometric planning via `~/plan_geometric_path`.
2. Configuration-space trajectory optimization via `~/compute_trajectory`.
3. Optional dry-run validation via execute services (execution itself is external).

## Robustness Notes

- Runtime capability checks are explicit (`_check_geometric_runtime`, `_check_trajectory_runtime`).
- Backend status is published on `~/trajectory_backend_status`.
- YAML loaders validate structure and numeric constraints before loading runtime state.

## Recommended Next Refactors

1. Add focused unit tests around:
   - `config.declare_and_load_config`
   - wall-plan progression and reset behavior
   - service success/failure behavior for core APIs
2. Introduce a pure-Python planner facade class to further reduce mixin surface in `node.py`.
