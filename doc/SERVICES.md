# concrete_block_motion_planning Services

This package exposes ROS 2 services from `motion_planning_node.py`.

By default (from `launch/motion_planning.launch.py`), the node name is:
- `concrete_block_motion_planning_node`

So private service names like `~/plan_geometric_path` resolve to:
- `/concrete_block_motion_planning_node/plan_geometric_path`

## Service Catalog

### `~/plan_geometric_path` (`PlanGeometricPath`)
Purpose:
- Computes a geometric Cartesian path between `start_pose` and `goal_pose`.

Inputs (main):
- `start_pose`, `goal_pose`
- `method` (optional, falls back to `default_geometric_method`)

Outputs:
- `success`
- `geometric_plan_id`
- `cartesian_path` (`nav_msgs/Path`)
- `message`

Capability notes:
- This is the geometric stage only (task-space path generation).
- Stores result internally by `geometric_plan_id` for later trajectory computation.

### `~/compute_trajectory` (`ComputeTrajectory`)
Purpose:
- Converts a geometric path into a time-parameterized joint trajectory.

Inputs (main):
- Either `geometric_plan_id` (stored from geometric stage) or `direct_path` with `use_direct_path=true`
- `method` (trajectory optimizer profile)
- `timeout_s`, `validate_dynamics`

Outputs:
- `success`
- `trajectory_id`
- `trajectory` (`trajectory_msgs/JointTrajectory`)
- `message`

Capability notes:
- Solves start/goal IK/steady-state from Cartesian poses.
- Runs trajectory optimization in joint/configuration space.
- Stores trajectory internally by `trajectory_id`.

### `~/execute_trajectory` (`ExecuteTrajectory`)
Purpose:
- Handles execution requests for a stored `trajectory_id`.

Inputs:
- `trajectory_id`
- `dry_run`

Outputs:
- `success`
- `message`

Capability notes:
- `dry_run=true`: validates and returns success.
- Non-dry-run execution is intentionally disabled in this node (planning-only design).
- Real execution must be done by an external execution server/backend.

### `~/execute_named_configuration` (`ExecuteNamedConfiguration`)
Purpose:
- Resolves a named joint configuration to a stored trajectory entry.

Inputs:
- `configuration_name`
- `timeout_s` (present in interface; not used for local execution here)
- `dry_run`

Outputs:
- `success`
- `trajectory_id`
- `message`

Capability notes:
- Creates a trajectory from preloaded named configurations.
- Non-dry-run execution remains disabled here (same planning-only behavior).

### `~/get_next_assembly_task` (`GetNextAssemblyTask`)
Purpose:
- Provides sequential wall-assembly tasks from a selected wall plan.

Inputs:
- `wall_plan_name`
- `reset_plan`

Outputs:
- `success`, `has_task`
- `task_id`
- `target_block_id`, `reference_block_id`
- `target_pose`, `reference_pose`
- `message`

Capability notes:
- Tracks progress per wall plan.
- Returns `has_task=false` when plan is complete.

## Actions

This package does not define or host ROS 2 actions in this node.

Notes:
- No `.action` files are defined in the package.

## Planning Space: Task Space or Configuration Space?

Short answer:
- Both.

How it is split:
- Stage 1 (`~/plan_geometric_path`): task-space planning (Cartesian path between poses).
- Stage 2 (`~/compute_trajectory`): configuration-space trajectory generation (joint trajectory) using IK/steady-state mapping and optimization.

Operationally, this node is a two-stage planner:
- Cartesian geometric planning followed by joint-space trajectory optimization.
