# `concrete_block_motion_planning` Services

Node name by default:

- `concrete_block_motion_planning_node`

Private services therefore resolve to paths like:

- `/concrete_block_motion_planning_node/plan_and_compute_trajectory`

## Canonical operator-facing services

### `~/plan_and_compute_trajectory`

Purpose:

- main backend-neutral planning entrypoint
- used by the current commissioning BTs

Inputs:

- `start_pose`, `goal_pose`
- `target_block_id`, `reference_block_id`
- `use_world_model`
- `geometric_method`, `geometric_timeout_s`
- `trajectory_method`, `trajectory_timeout_s`
- `validate_dynamics`

Behavior:

- resolves planning context from the world model
- dispatches to the active planner backend
- stores the resulting trajectory
- publishes planned path to RViz when available

Current note:

- for the CBS/concrete path, this is the preferred shared-shell entrypoint
- the concrete online default is `TOPPRA_PATH_FOLLOWING`
- for the timber path, this remains the working reference service

### `~/execute_trajectory`

Purpose:

- execute a stored trajectory by `trajectory_id`

Inputs:

- `trajectory_id`
- `dry_run`

Behavior:

- `dry_run=true` validates only
- `dry_run=false` dispatches through the configured execution backend

### `~/execute_named_configuration`

Purpose:

- resolve and optionally execute a named configuration

Inputs:

- `configuration_name`
- `dry_run`

Outputs:

- `trajectory_id`
- `success`
- `message`

### `~/get_next_assembly_task`

Purpose:

- retrieve the next task from the active wall plan

Outputs:

- `task_id`
- `target_block_id`
- `reference_block_id`
- `has_task`

This is the task-selection seam used by `Single block plan` and `Single block execute`.

## Lower-level staged services

These remain useful for debugging and CBS commissioning:

### `~/plan_geometric_path`

- geometric path only
- most relevant for `planner.backend:=concrete`

### `~/compute_trajectory`

- trajectory stage only
- useful when commissioning the concrete IK + TOPP-RA stage separately from geometric planning

## World-model context used by the planner

The shared planner shell currently talks to:

- `/world_model_node/get_planning_scene`

The planning-scene response includes:

- block objects from the persistent world model
- static crane/environment obstacles

CBS/FCL scene building should consume that centralized scene rather than reconstructing obstacles locally.

Compatibility note:

- `/world_model_node/get_coarse_blocks` still exists for block-only consumers, but it is no longer the intended scene source for CBS obstacle-aware planning.
