# `concrete_block_motion_planning` Architecture

## Current role

This package is the shared motion-planning service layer for the concrete-block stack.
Behavior trees and operator tooling should talk only to these shared services:

- `plan_and_compute_trajectory`
- `execute_trajectory`
- `execute_named_configuration`
- `get_next_assembly_task`
- `plan_geometric_path`
- `compute_trajectory`

The active commissioning goal is backend interchangeability behind that API:

- `planner.backend:=timber`
  - current validated reference path
  - direct A-to-B / iLQR-style planning and proven execution bridge
- `planner.backend:=concrete`
  - CBS-oriented path
  - geometric planning against a centralized world-model scene
  - fully-actuated online trajectory stage with TOPP-RA path following

## Module layout

- `scripts/motion_planning_node.py`
  - thin executable entrypoint
- `scripts/cbmp/node.py`
  - node composition, ROS I/O, service registration, backend selection
- `scripts/cbmp/config.py`
  - typed parameter loading
- `scripts/cbmp/services.py`
  - shared service handlers and planning-context resolution
- `scripts/cbmp/runtime.py`
  - geometric planning runtime, scene construction, trajectory helpers
- `scripts/cbmp/backends/`
  - backend-specific adapters for `timber` and `concrete`
- `scripts/cbmp/state.py`
  - in-memory plans, trajectories, wall-plan progress, runtime caches
- `motion_planning/`
  - vendored geometry / optimization / acados tooling
- `motion_planning_tools/`
  - standalone tools and benchmarks

## Current architecture split

### Shared shell

The shared shell owns:

- ROS services
- wall-plan progression
- named configurations
- planned-path publication to RViz
- execution dispatch
- world-model context lookup

### Timber backend

The timber backend remains the current execution reference:

- proven for `Move empty`
- proven for `Single block plan`
- proven for `Single block execute`

Its strengths are turnaround and current robustness. Its main weakness is that obstacle knowledge historically lived outside the centralized world model.

### Concrete / CBS backend

The concrete backend is the CBS path under commissioning:

- `plan_geometric_path` builds an FCL scene from the centralized world-model planning scene
- `plan_and_compute_trajectory` uses that same planning context through the shared shell
- the online trajectory stage now solves IK along the geometric path and time-parameterizes the joint path with TOPP-RA

## Centralized planning scene

The current CBS direction is to consume one centralized scene from the world model:

- dynamic blocks from persistent world state
- static crane/environment obstacles
- RViz visualization from the same source

The scene is provided by:

- `/world_model_node/get_planning_scene`

This service is now the intended source of truth for CBS/FCL obstacle queries. `get_coarse_blocks` remains for compatibility and block-oriented consumers.

## Near-term commissioning target

The next clean milestone is:

- `Move empty` stays green on timber
- `Single block plan` works on both `planner.backend:=timber|concrete`
- CBS geometric planning consumes the centralized planning scene
- the concrete backend plans and executes online without acados in the commissioning loop

## Trajectory optimization direction

There are now two distinct tracks:

### Online commissioning-safe path

- fully actuated staged path-following
- geometric plan first
- IK along the geometric path second
- TOPP-RA time-parameterization third
- retain `FIXED_TIME_INTERPOLATION` only as a debug fallback

### R&D path

- standalone acados benchmark harness
- curated fixed benchmark cases
- future free-end-time full-dynamics OCP experiments

The repo now includes:

- `scripts/acados_benchmark.py`
- `motion_planning/data/acados_bench_cases.yaml`

These are intentionally standalone so solver tuning does not require RViz/BT/Gazebo.

Current rule:

- acados is offline-only for now
- `TOPPRA_PATH_FOLLOWING` is the concrete online default
