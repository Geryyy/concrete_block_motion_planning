# Standalone Planner Handoff

## Scope

This document captures the current CBS/concrete motion-planning state after moving planner development back into the standalone `motion_planning` module.

It is written so a fresh agent can continue planner work without replaying the full conversation history.

## Current Goal

Rebuild concrete motion planning from the bottom up with the simplest robust stack:

1. solver/model consistency
2. deterministic joint-space path generation
3. simple timing
4. only later compare against heavier methods like TOPP-RA variants and `acados`

The current development rule is:

- do planner invention in the standalone lab
- treat CBS/ROS/Gazebo as integration layers

## Current Architecture

### Standalone lab

Primary files:

- `motion_planning/standalone/types.py`
- `motion_planning/standalone/scenarios.py`
- `motion_planning/standalone/evaluate.py`
- `motion_planning/standalone/plotting.py`
- `motion_planning/standalone/compare_solvers.py`
- `motion_planning/standalone/time_parameterization.py`
- `motion_planning/standalone/stacks/joint_goal_interpolation.py`
- `motion_planning/standalone/stacks/cartesian_anchor_joint_spline.py`
- `motion_planning_tools/standalone/run_planner_experiment.py`

### Core solve layer

Important files:

- `motion_planning/pipeline/joint_goal_stage.py`
- `motion_planning/mechanics/analytic/steady_state.py`
- `motion_planning/pipeline/joint_space_stage.py`

### Scene library

Existing scene definitions used for block geometry:

- `motion_planning/scenarios.py`
- `motion_planning/data/generated_scenarios.yaml`

## What Was Changed

### 1. Standalone planner lab was added

The repo now has a lightweight runner that works without ROS or Gazebo.

Supported modes:

- planner runs
- solver comparison
- optional simple timing
- optional matplotlib plotting
- optional real block-scene overlay

### 2. Solver truth checks were tightened

`CraneSteadyState` and `JointGoalStage` now expose and propagate:

- FK position error
- FK yaw error
- FK pose of returned state
- IK backend/status

The solve layer now rejects meter-scale nonsense instead of silently returning `success=True`.

Current default truth tolerance is looser than before:

- FK position tolerance is currently `2e-2 m`
- yaw tolerance is still small

This is intentional for now so seeded reachable cases pass while large failures are still rejected.

### 3. Simplified planner path is in place

The current lightweight planner ladder includes:

- `joint_goal_interpolation`
  - solve or seed start/goal
  - interpolate actuated joints
  - complete passive joints via steady-state during FK/evaluation

- `cartesian_anchor_joint_spline`
  - use sparse Cartesian anchors
  - solve anchors
  - fit actuated-joint spline
  - fallback to direct start-goal path if all interior anchors fail

This fallback is intentional right now to keep the stack usable while solver coverage improves.

### 4. Real scene visualization is available

The plotter can now render:

- a real 3D block scene from the scenario library
- reference TCP path
- realized TCP path
- XY/XZ projections
- joint paths

### 5. A reachable scene-backed demo was added

New standalone scenario:

- `scene_demo_step_01_reachable`

This uses:

- block geometry from `step_01_first_on_ground`
- a translated scene so geometry lines up with the validated reachable path demo
- the validated short-range planner start/goal seeds

This is currently the best “real blocks + real path” demo in the repo.

## Current Verified Commands

### Scene-backed reachable demo with plot

```bash
python3 src/concrete_block_stack/concrete_block_motion_planning/motion_planning_tools/standalone/run_planner_experiment.py \
  --mode planner \
  --stack joint_goal_interpolation \
  --scenario scene_demo_step_01_reachable \
  --timing simple \
  --plot
```

### Short reachable anchor planner

```bash
python3 src/concrete_block_stack/concrete_block_motion_planning/motion_planning_tools/standalone/run_planner_experiment.py \
  --mode planner \
  --stack cartesian_anchor_joint_spline \
  --scenario short_reachable_move
```

### Solver comparison for the reachable short case

```bash
python3 src/concrete_block_stack/concrete_block_motion_planning/motion_planning_tools/standalone/run_planner_experiment.py \
  --mode solver_compare \
  --scenario short_reachable_move
```

### Focused verification

```bash
python3 -m pytest -q \
  src/concrete_block_stack/concrete_block_motion_planning/motion_planning/tests/test_standalone_lab.py \
  src/concrete_block_stack/concrete_block_motion_planning/motion_planning/tests/test_joint_goal_stage.py \
  src/concrete_block_stack/concrete_block_motion_planning/motion_planning/tests/test_crane_steady_state.py
```

Current verified state:

- `10 passed`

## Important Current Scenarios

### Good / validated enough for iteration

- `short_reachable_move`
- `yaw_change_probe`
- `scene_demo_step_01_reachable`

### Still failing / debugging targets

- `single_block_transfer`
- raw scene tasks such as:
  - `scene_step_01_first_on_ground`
  - other direct scene-library tasks without curated reachable seeds

These are valuable because they expose real solve limitations.

## Known Problems

### 1. Solver coverage is still incomplete

Even with the improved truth checks, raw scene tasks are not yet broadly solvable.

Typical failure pattern:

- start/goal world pose seems reasonable
- numeric IK returns something plausible
- final FK truth check still shows a significant miss

### 2. Anchor planner often drops interior anchors

The current `cartesian_anchor_joint_spline` is intentionally conservative.

When interior anchor solves fail, it currently falls back to a direct start-goal path instead of failing hard.

This is acceptable for v1 iteration, but not the final desired behavior.

### 3. CBS runtime is no longer the right debugging surface

The CBS runtime should eventually consume logic validated in the standalone lab.

Do not start the next debugging round inside:

- `scripts/cbmp/runtime.py`
- BT launch files
- Gazebo

unless the issue is clearly integration-specific.

## Next Recommended Steps

### Immediate next step

Write additional handoff or experiment notes only after using the standalone lab as the reference.

### Technical next step

Improve solver/model consistency on actual scene tasks:

1. investigate why direct scene-library tasks fail without curated seeds
2. compare concrete solve behavior against timber reference solve logic where useful
3. determine whether the problem is:
   - poor seed handling
   - insufficient IK budget
   - branch selection
   - frame/pose convention mismatch
   - passive completion drift

### After solver improvements

1. make more scene-backed demos reachable:
   - `step_02_second_in_front`
   - `step_03_third_on_top`
2. strengthen the anchor planner so it uses real interior anchors more often
3. compare simple timing against existing TOPP-RA backends
4. keep `acados` as a later comparison backend only

## Do Not Forget

- The worktree was clean and the focused planner tests were passing when this handoff was written.
- The currently best demonstration command is the `scene_demo_step_01_reachable` plot command above.
- The main unsolved problem is not plotting or ROS wiring anymore; it is solver coverage on real scene tasks.
