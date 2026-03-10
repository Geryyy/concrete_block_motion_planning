# Motion Planning

## Project Requirements and Implementation Roadmap (Pipeline Context)

Requirements baseline date: **2026-03-04**

This section captures the current end-to-end requirements for the concrete-block pipeline and a modular strategy to implement and verify each part before full integration.

### Mission Goal

Build a robust perception-planning-execution pipeline for concrete block assembly with uncertainty compensation for a flexible/inaccurate crane.

### Functional Requirements

1. Scene discovery and coverage
- Move crane through predefined/planned viewpoints to cover the full scene.
- Detect all visible blocks and build/update the world model.
- Avoid duplicate block spawning by associating new observations to existing world-model blocks.

2. Assembly plan generation
- Generate a basic wall construction plan:
  - bottom row: 3 blocks
  - top row: 2 blocks with lateral offset (interlocking pattern)
- Keep explicit ordering constraints (which block must be placed before the next one).

3. Grasping stage (per block)
- Move to a pre-grasp vicinity pose near target block.
- Trigger on-demand perception refinement for accurate target pose.
- Plan and execute collision-free TCP motion to block center for grasp.

4. Placement/assembly stage (per block)
- Move to a pre-place vicinity pose near target placement area.
- First placed block may use absolute placement.
- For all following blocks, place relative to previously placed block(s).
- Estimate:
  - pose of pre-placed reference block
  - pose of block currently in gripper
- Use relative measurement for path/planning correction to compensate manipulator uncertainty/flexibility.

5. World model as central truth
- Maintain block lifecycle/state (`free`, `move`, `placed`, `removed`) and pose quality (`coarse`, `precise`).
- On every perception update:
  - associate to known block when match exists
  - create new block entry when unmatched
- Provide consistent interfaces for BT and motion planning.

6. Testing environment split
- Perception testing: rosbags of manually grasped/moved blocks.
- Motion planning + BT testing: Gazebo simulation workspace.
- Implementation must be modular and fine-grained, with independent verification gates.

### Current Implementation Mapping (What Already Exists)

- Perception already exposes BT-friendly one-shot service calls via `world_model_node`:
  - `run_pose_estimation` (`SCENE_DISCOVERY`, `REFINE_BLOCK`, `REFINE_GRASPED`)
  - `get_coarse_blocks`
  - `set_mode`
- Motion planning already supports staged services:
  - `plan_geometric_path`
  - `compute_trajectory`
  - `execute_trajectory`
- Behavior tree plugins already exist for:
  - `RunPoseEstimation`
  - `PlanGeometricPath`
  - `ComputeTrajectory`
  - `ExecuteTrajectory`

Primary missing or incomplete items to close:
- explicit viewpoint coverage policy/implementation for full-scene discovery
- deterministic wall-plan generator with interlocking constraints
- robust relative placement update path (reference + grasped block fusion into placement command)
- production controller integration for trajectory execution (current execution path remains placeholder in motion planning node)
- stricter world-model association and state-transition assertions under uncertainty

### Modular Implementation Plan (Verify Before Plumbing)

Phase 0. Interface freeze and observability
- Finalize service/message contracts across perception, world model, motion planning, BT blackboard keys.
- Add structured run logs and IDs (scan ID, plan ID, trajectory ID, block ID).
- Exit criteria:
  - all modules can be started independently
  - service contracts documented and stable

Phase 1. World-model state engine hardening
- Implement/validate association policy (match vs spawn) with confidence/time gating.
- Enforce valid state transitions (`FREE -> MOVE -> PLACED`, etc.).
- Add assertion/reporting for stale or conflicting updates.
- Verification:
  - rosbag replay tests for track continuity and correct spawn/merge behavior
  - unit tests for transition and association logic

Phase 2. Scene coverage/scanning module
- Add viewpoint sequence planner/executor (initially deterministic sweep, later optimized).
- At each viewpoint: trigger `SCENE_DISCOVERY`, merge into world model.
- Stop when coverage/novelty criteria are satisfied.
- Verification:
  - rosbag-based replay with synthetic viewpoint order
  - metric: detected unique blocks vs expected count

Phase 3. Assembly plan module (wall 3+2 interlocking)
- Implement basic wall planner that outputs ordered placement tasks:
  - bottom: `B0, B1, B2`
  - top offset: `T0, T1`
- Each task includes target pose type:
  - absolute for first anchor block
  - relative for subsequent blocks
- Verification:
  - pure unit tests for generated poses/order/offset constraints
  - Gazebo dry-run check of target markers

Phase 4. Grasp pipeline module
- BT subtree: approach vicinity -> `REFINE_BLOCK` -> collision-free plan -> grasp execution.
- Fail-safe retries: re-refine and replan on low confidence/planning failure.
- Verification:
  - rosbag-in-the-loop perception checks
  - Gazebo grasp success rate and collision checks

Phase 5. Relative assembly refinement module
- Before each placement after first:
  - estimate reference block pose (`REFINE_BLOCK` on placed block)
  - estimate grasped block pose (`REFINE_GRASPED`)
- Compute relative correction transform and update place pose.
- Plan/execute corrected trajectory.
- Verification:
  - Gazebo perturbation tests (pose bias/noise injected)
  - metric: placement error reduction vs baseline absolute placement

Phase 6. End-to-end BT orchestration and recovery
- Integrate phases into full BT with explicit recovery branches:
  - re-scan
  - re-select target
  - replan trajectory
  - skip/abort policy
- Verification:
  - scenario tests in Gazebo: full 3+2 wall completion
  - robustness tests with dropped detections and trajectory failures

### Verification Matrix (Required)

1. Perception module (rosbags)
- block identity stability across motion
- spawn-vs-associate correctness
- coarse/precise update behavior

2. Motion planning module (simulation)
- geometric path success and collision checks
- trajectory feasibility and tracking quality
- deterministic response under repeated seeds

3. BT orchestration module (simulation)
- correct call order and blackboard updates
- recovery behavior correctness
- end-to-end wall completion rate

4. Relative correction benefit
- compare absolute-only placement vs relative-measurement-based placement
- report translational and angular final placement errors

### Recommended Immediate Next Steps

1. Implement Phase 1 test harness first (world-model association/state assertions from rosbags).
2. Implement Phase 3 wall-plan generator in isolation with unit tests.
3. Add Phase 5 relative-correction computation as a standalone module with Gazebo perturbation benchmarks.
4. After those pass independently, integrate into BT and run full-stack scenarios.

### Issue Checklist (Concrete Package/File Targets)

Use this checklist as implementation tickets. Mark done only when code + tests + launch-level validation pass.

`P0` Interface Freeze and Observability

- [ ] `P0-01` Freeze cross-package service and BT contracts
  Targets: `concrete_block_perception/srv/RunPoseEstimation.srv`, `concrete_block_perception/srv/GetCoarseBlocks.srv`, `concrete_block_perception/srv/SetPerceptionMode.srv`, `concrete_block_motion_planning/srv/PlanGeometricPath.srv`, `concrete_block_motion_planning/srv/ComputeTrajectory.srv`, `concrete_block_motion_planning/srv/ExecuteTrajectory.srv`, `concrete_block_behavior_tree/behavior_trees/concrete_block_assembly.xml`, `concrete_block_behavior_tree/config/default.yaml`
  Done criteria: request/response fields and BT blackboard keys documented and unchanged across one sprint.

- [ ] `P0-02` Add structured run identifiers and status reporting
  Targets: `concrete_block_perception/src/nodes/world_model_node.cpp`, `concrete_block_motion_planning/scripts/motion_planning_node.py`, `concrete_block_behavior_tree/src/plugins/action/*.cpp`
  Done criteria: logs include `scan_id`, `plan_id`, `trajectory_id`, `block_id`; failures are attributable per stage.

`P1` World Model Hardening

- [ ] `P1-01` Implement explicit association vs spawn policy with gating
  Targets: `concrete_block_perception/src/nodes/world_model_node.cpp`, `concrete_block_perception/src/utils/world_model_utils.cpp`, `concrete_block_perception/include/concrete_block_perception/utils/world_model_utils.hpp`, `concrete_block_perception/config/world_model.yaml`
  Done criteria: configurable distance/time/confidence thresholds drive deterministic decision logic.

- [ ] `P1-02` Enforce block lifecycle/state transition assertions
  Targets: `concrete_block_perception/msg/Block.msg`, `concrete_block_perception/src/nodes/world_model_node.cpp`
  Done criteria: invalid transitions rejected and logged with reason.

- [ ] `P1-03` Add rosbag regression harness for world-model behavior
  Targets: `concrete_block_perception/launch/rosbag_block_world_model.launch.py`, `concrete_block_perception/launch/rosbag_block_world_model_test_modes.launch.py`, `concrete_block_perception/test/`
  Done criteria: reproducible pass/fail checks for identity stability and spawn/merge behavior.

`P2` Scene Coverage

- [ ] `P2-01` Add viewpoint coverage sequencer (deterministic sweep first)
  Targets: `concrete_block_behavior_tree/behavior_trees/concrete_block_assembly.xml`, `concrete_block_behavior_tree/config/default.yaml`, `concrete_block_behavior_tree/src/plugins/action/run_pose_estimation.cpp`
  Done criteria: sequence requests `SCENE_DISCOVERY` from multiple viewpoints and aggregates updates.

- [ ] `P2-02` Add coverage completion metric and stop condition
  Targets: `concrete_block_perception/src/nodes/world_model_node.cpp`, `concrete_block_perception/config/world_model.yaml`
  Done criteria: stop scan when novelty gain or expected count threshold is met.

`P3` Wall Plan Generator (3+2 Interlocking)

- [ ] `P3-01` Create deterministic assembly plan generator
  Targets: `concrete_block_motion_planning/motion_planning/scenarios.py`, `concrete_block_motion_planning/motion_planning/pipeline/geometric_stage.py`, `concrete_block_motion_planning/motion_planning/data/generated_scenarios.yaml`
  Done criteria: emits ordered tasks for `B0,B1,B2,T0,T1` with interlocking top-row offset.

- [ ] `P3-02` Encode absolute-first then relative placements
  Targets: `concrete_block_motion_planning/motion_planning/core/world_model.py`, `concrete_block_motion_planning/motion_planning/core/types.py`
  Done criteria: first placement can be absolute; subsequent targets are defined relative to placed references.

- [ ] `P3-03` Add unit tests for wall-plan structure and constraints
  Targets: `concrete_block_motion_planning/motion_planning/tests/` (add `test_wall_plan_generator.py`)
  Done criteria: tests assert ordering, offsets, and valid target frames.

`P4` Grasp Pipeline

- [ ] `P4-01` Implement BT subtree for approach -> refine -> plan -> grasp
  Targets: `concrete_block_behavior_tree/behavior_trees/concrete_block_assembly.xml`, `concrete_block_behavior_tree/behavior_trees/minimal_smoke.xml`
  Done criteria: each block pickup includes mandatory `REFINE_BLOCK` before final grasp motion.

- [ ] `P4-02` Add retry policy for low-confidence or planning failure
  Targets: `concrete_block_behavior_tree/behavior_trees/concrete_block_assembly.xml`, `concrete_block_behavior_tree/src/plugins/action/plan_geometric_path.cpp`, `concrete_block_behavior_tree/src/plugins/action/compute_trajectory.cpp`
  Done criteria: bounded retries with explicit fallback method and clear terminal failure.

`P5` Relative Assembly Refinement

- [ ] `P5-01` Add reference + grasped block perception update before place
  Targets: `concrete_block_behavior_tree/behavior_trees/concrete_block_assembly.xml`, `concrete_block_perception/src/nodes/world_model_node.cpp`
  Done criteria: placement stage calls both `REFINE_BLOCK(reference)` and `REFINE_GRASPED(grasped)`.

- [ ] `P5-02` Compute and apply relative correction transform in placement planning
  Targets: `concrete_block_motion_planning/motion_planning/api.py`, `concrete_block_motion_planning/motion_planning/pipeline/geometric_stage.py`, `concrete_block_motion_planning/scripts/motion_planning_node.py`
  Done criteria: corrected place pose uses live relative measurement and is persisted in plan metadata.

- [ ] `P5-03` Benchmark correction benefit under perturbations
  Targets: `concrete_block_motion_planning/motion_planning/tests/` (add `test_relative_placement_correction.py`), Gazebo test launch in workspace
  Done criteria: report translational/angular error reduction vs absolute placement baseline.

`P6` End-to-End Integration and Recovery

- [ ] `P6-01` Integrate all modules in full-stack launch paths
  Targets: `concrete_block_behavior_tree/launch/full_stack.launch.py`, `concrete_block_behavior_tree/launch/offline_test.launch.py`, `concrete_block_perception/launch/perception.launch.py`, `concrete_block_motion_planning/launch/motion_planning.launch.py`
  Done criteria: full pipeline runs from scan to final wall placement in simulation.

- [ ] `P6-02` Add recovery branches for perception/planning/execution failures
  Targets: `concrete_block_behavior_tree/behavior_trees/concrete_block_assembly.xml`
  Done criteria: re-scan/re-plan/retry branches are exercised in fault-injection runs.

- [ ] `P6-03` Validate completion KPI in Gazebo
  Targets: Gazebo scenario launch and test report artifacts in workspace
  Done criteria: repeatable end-to-end success rate and cycle-time metrics for 3+2 wall assembly.

### Suggested Execution Order (1-Week Blocks)

1. Week 1: `P0-01`, `P0-02`, `P1-01`
2. Week 2: `P1-02`, `P1-03`, `P3-01`
3. Week 3: `P3-02`, `P3-03`, `P4-01`
4. Week 4: `P4-02`, `P5-01`, `P5-02`
5. Week 5: `P5-03`, `P6-01`, `P6-02`, `P6-03`

### Scope Split Agreed (Now vs Later)

Implemented now (engineering work we can do immediately):

1. Rudimentary viewpoint coverage scan:
- BT moves crane to a fixed list of configurations.
- At each configuration: trigger one scene detection/update call.

2. Rudimentary wall plan source:
- YAML-driven plan for 3-bottom + 2-top interlocking wall.
- First block pose absolute; all following blocks defined relative to previous/placed blocks.

3. Motion execution backend:
- Replace placeholder execution path with real controller integration.

4. World-model hardening:
- Stronger association/state-transition assertions + tests.

Deferred to later joint perception sessions (with rosbag validation):

1. Fine-grained on-demand refinement strategy from coarse pose/FK priors.
2. Targeted single-block detection in cluttered FoV (avoid wrong block updates).
3. Rosbag time synchronization strategy for action-coupled evaluation.
4. Partial occlusion handling caused by gripper.
5. End-to-end KPI formalization (success rate, placement error reduction, cycle time).

### Concrete Plan (What I Can Execute Now)

`Step A` Fixed-viewpoint scan via BT (rudimentary)

- Add a YAML config for scan viewpoints (named crane configurations).
- Extend BT flow to iterate viewpoints and call detection/update at each point.
- Reuse existing service interfaces; no deep perception algorithm changes.

Targets:
- `concrete_block_behavior_tree/behavior_trees/concrete_block_assembly.xml`
- `concrete_block_behavior_tree/config/default.yaml`
- `concrete_block_behavior_tree/launch/full_stack.launch.py`

Done criteria:
- BT can run scan sequence across configured viewpoints.
- World model receives one update per viewpoint.

`Step B` YAML wall-plan generator (rudimentary, deterministic)

- Add wall-plan YAML schema (block IDs, relative references, offsets, orientation).
- Implement loader + validation.
- Produce ordered placement tasks for BT/planning consumption.

Targets:
- `concrete_block_motion_planning/motion_planning/data/` (new wall plan YAML)
- `concrete_block_motion_planning/motion_planning/scenarios.py`
- `concrete_block_motion_planning/motion_planning/core/types.py`
- `concrete_block_motion_planning/motion_planning/tests/` (new unit test)

Done criteria:
- Plan loaded from YAML and resolved into absolute targets at runtime.
- Unit tests validate ordering and relative-pose resolution.

`Step C` Execution backend integration (replace placeholder)

- Integrate `execute_trajectory` with an actual ROS execution path:
  - preferred: `control_msgs/action/FollowJointTrajectory` client
  - keep `dry_run` and fallback behavior for simulation bring-up.
- Add execution status and timeout/error propagation.

Targets:
- `concrete_block_motion_planning/scripts/motion_planning_node.py`
- `concrete_block_motion_planning/config/motion_planning.yaml`
- `concrete_block_motion_planning/launch/motion_planning.launch.py`

Done criteria:
- Non-dry-run calls forward trajectories to controller/action server.
- Service response reflects real execution outcome.

`Step D` World-model assertions + tests (without advanced perception changes)

- Harden association/spawn gating and state-transition checks.
- Add regression tests/harness for deterministic behavior under replay.

Targets:
- `concrete_block_perception/src/nodes/world_model_node.cpp`
- `concrete_block_perception/src/utils/world_model_utils.cpp`
- `concrete_block_perception/config/world_model.yaml`
- `concrete_block_perception/test/` and rosbag test launch files

Done criteria:
- Invalid transitions are rejected and logged.
- Replay tests show stable IDs and expected spawn/association behavior.

### Planned Later (Perception Co-Development Sessions)

1. Pose refinement seeded from coarse estimate/FK for:
- pre-placed reference block
- grasped block

2. Single-target perception in cluttered FoV:
- gating by expected region/ID prior and confidence policy.

3. Rosbag time-action synchronization design:
- deterministic trigger schedule for:
  - detect all blocks in scene
  - detect block in gripper
  - detect pre-placed block

4. Occlusion-specific handling and robustness experiments for gripper-induced partial visibility.

## Dependency Management

Dependencies are declared in layers:

- ROS package deps: `package.xml`
  - e.g. `rclpy`, message packages (`geometry_msgs`, `nav_msgs`, `trajectory_msgs`, ...)
- Python runtime/tooling deps: `pyproject.toml` and `requirements-*.txt`
  - core deps are in `requirements-core.txt`
  - benchmark extras are in `requirements-benchmark.txt`
- acados Python package deps: `acados/interfaces/acados_template/setup.py`
  - installed locally from source (`pip install -e acados/interfaces/acados_template`)

### Install core Python deps

```bash
python3 -m pip install --user -r requirements-core.txt
```

### Install benchmark extras

```bash
python3 -m pip install --user -r requirements-benchmark.txt
```

### Optional/system dependencies

- `python-fcl`: required for geometric/world-model collision checks (`fcl` Python module).
- `pinocchio`: required by trajectory and mechanics modules (typically installed via system package manager/robotics stack).
- `ompl` Python bindings: optional, only needed for `OMPL-RRT` benchmark method.
- `vpsto`: optional, only needed for `VP-STO` benchmark method.

## Available Planners

- Benchmark/tooling module (`motion_planning_tools.benchmark`):
  - `Powell`, `Nelder-Mead`, `CEM`, `VP-STO`, `OMPL-RRT`
- Clean planner API (`motion_planning.planners.create_planner`):
  - `Powell`, `Nelder-Mead`, `CEM`

## Run Benchmark

Run Optuna benchmark search with persistent storage:

```bash
python3 -m motion_planning_tools.benchmark.cli \
  --trials 100 \
  --optuna-storage sqlite:///optuna_bench.db
```

Default behavior:
- Supported benchmark methods include: `Powell`, `Nelder-Mead`, `CEM`, `VP-STO`, `OMPL-RRT`.
- `--optuna-jobs` defaults to all CPU cores. You can also pass `--optuna-jobs 0` (or negative) to auto-use all cores.
- If `--optuna-storage` is SQLite and `--optuna-jobs > 1`, the benchmark automatically uses persistent Optuna Journal storage (`*.journal`) to avoid SQLite lock errors.
- If Journal storage is unavailable in your Optuna version, it falls back to single-worker SQLite safely.

For best persistent parallel throughput at larger scale, use PostgreSQL/MySQL as Optuna storage.

## Run Planner Query

Run a single scenario with the demo CLI:

```bash
python3 -m motion_planning.demo \
  --scenario step_02_second_in_front \
  --planner CEM
```

Runtime query API (ROS-friendly):

```python
from motion_planning import WorldModel, plan

wm = WorldModel()
wm.reset()
wm.add_block(size=(0.6, 0.9, 0.6), position=(1.0, 0.0, 0.3), object_id="base_0")
print(wm.query_block("base_0"))

res = plan(
    start=(0.0, 0.0, 0.3),
    end=(1.2, 0.0, 0.3),
    method="CEM",
    world_model=wm,
    moving_block_size=(0.6, 0.9, 0.6),
    optimized_params_file="motion_planning/data/optimized_params.yaml",
)
print(res.success, res.message)
```

## Run Tests

Run the package test suite:

```bash
conda run -n mp_env pytest -q motion_planning/tests
```

Verbose output:

```bash
conda run -n mp_env pytest motion_planning/tests -v
```

Note:
- Running plain `pytest` from repo root may also collect vendored `acados/` tests. Use `motion_planning/tests` to validate this package.

## URDF ↔ MJCF Conversion

Conversion/consistency tools are in:

- `motion_planning/conversion/README.md`

Run the conversion CLI:

```bash
python -m motion_planning.conversion --help
```

Example full pipeline:

```bash
python -m motion_planning.conversion pipeline \
  --urdf crane_urdf/crane.urdf \
  --out-dir /tmp/motion_planning_conversion \
  --samples 20 --seed 0
```

Check URDF/MJCF model-core sync (ignores scene/actuator/site extras):

```bash
python -m motion_planning.conversion check-sync \
  --urdf crane_urdf/crane.urdf \
  --mjcf crane_urdf/crane.xml
```

Run a planner query with benchmark-optimized parameters (Python API):

```python
from pathlib import Path
from motion_planning.scenarios import ScenarioLibrary
from motion_planning.pipeline import run_geometric_planning_from_benchmark_params

wm = ScenarioLibrary("motion_planning/data/generated_scenarios.yaml")
sc = wm.build_scenario("step_02_second_in_front")

res = run_geometric_planning_from_benchmark_params(
    world_scenario=sc,
    method="CEM",
    optimized_params_file=Path("motion_planning/data/optimized_params.yaml"),
)
print(res.success, res.message)
```

## Clean API (Geometric Stage)

Minimal package API is available in `motion_planning/` for clean integration between geometric planning and future trajectory optimization.

Use benchmark-optimized planner parameters directly:

```python
from pathlib import Path
from motion_planning.scenarios import ScenarioLibrary
from motion_planning.pipeline import run_geometric_planning_from_benchmark_params

wm = ScenarioLibrary("motion_planning/data/generated_scenarios.yaml")
sc = wm.build_scenario("step_02_second_in_front")

res = run_geometric_planning_from_benchmark_params(
    world_scenario=sc,
    method="CEM",
    optimized_params_file=Path("motion_planning/data/optimized_params.yaml"),
)

P = res.path.sample(101)
print(res.success, res.message, P.shape)
```

## acados Setup (Trajectory Optimization)

For trajectory optimization (`acados + CasADi + Pinocchio`), use the following package-local setup.

### 1) Initialize acados source (recommended as submodule)

If this repository tracks `acados` as a submodule:

```bash
git submodule update --init --recursive acados
```

If `acados` is not present yet, add it:

```bash
git submodule add https://github.com/acados/acados.git acados
git submodule update --init --recursive
```

### 2) Build and install acados

```bash
cd acados
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DACADOS_WITH_QPOASES=ON -DACADOS_WITH_OSQP=ON
make -j$(nproc)
make install
cd ../..
```

### 3) Install Python dependencies

```bash
python3 -m pip install --user casadi
python3 -m pip install --user -e acados/interfaces/acados_template
```

### 4) Build `t_renderer` (required by acados template codegen)

Install Rust toolchain (if needed), then build:

```bash
# one-time if cargo/rustc are missing
sudo apt-get update && sudo apt-get install -y cargo rustc

cd acados/interfaces/acados_template/tera_renderer
cargo build --release
mkdir -p ../../../bin
cp target/release/t_renderer ../../../bin/
cd ../../../..
```

### 5) Activate runtime environment

Use the package helper script:

```bash
source acados_interface_setup.sh
```

This sets:
- `ACADOS_SOURCE_DIR`
- `LD_LIBRARY_PATH`
- `PYTHONPATH`
- `PATH` (for `t_renderer`)

### 6) Verify installation

```bash
python3 -c "import acados_template, casadi, pinocchio; print('acados runtime OK')"
ls "$ACADOS_SOURCE_DIR/bin/t_renderer"
```

### Reference docs

Official docs (for advanced/custom builds):

- Installation: https://docs.acados.org/installation/
- Python interface: https://docs.acados.org/python_interface/index.html

Notes:

- `ACADOS_SOURCE_DIR` should point to a built acados tree containing `lib/link_libs.json`.
- If `t_renderer` is missing, build it from `acados/interfaces/acados_template/tera_renderer`.
- Run `source acados_interface_setup.sh` (not `bash acados_interface_setup.sh`) so exports persist in the current shell.
- `motion_planning/trajectory/crane_acados_ocp_setup.py` now prefers the repository-local `acados/` checkout when present, to avoid template/runtime mismatches.

## Run Trajectory Optimization + Visualization

Generate a **configuration-space path-following** trajectory (B-spline path with progress states `s, sdot` and progress acceleration input `v = sddot`) and save plot + trajectory artifact locally:

```bash
source acados_interface_setup.sh
conda run -n mp_env python motion_planning/trajectory/run_crane_acados_ocp_example.py \
  --traj-out ./crane_acados_ocp_trajectory.npz \
  --plot-out ./crane_acados_ocp_example.png
```

Replay the optimized trajectory in MuJoCo using computed-torque control (with telescope tie `q4=q5=0.5*telescope_ref` and rails `q9/q11` held at lower limits):

```bash
conda run -n mp_env python motion_planning/simulation/mujoco_pd_replay.py \
  --traj ./crane_acados_ocp_trajectory.npz \
  --report-out ./crane_pd_replay_report.npz \
  --kp 20 --kd 5 --tail-s 0.2 --no-view
```

Open MuJoCo viewer during replay:

```bash
conda run -n mp_env python motion_planning/simulation/mujoco_pd_replay.py \
  --traj ./crane_acados_ocp_trajectory.npz
```

Inspect the generated trajectory artifact:

```bash
python - <<'PY'
import numpy as np
d = np.load("crane_acados_ocp_trajectory.npz")
print(sorted(d.files))
print("q:", d["q_trajectory"].shape, "dq:", d["dq_trajectory"].shape)
print("s:", d["s_trajectory"].shape, "sdot:", d["sdot_trajectory"].shape)
PY
```

## Run Analytic IK (URDF-Style)

Simple IK example using the analytic mechanics module:

```bash
conda run -n mp_env python motion_planning/example_crane_analytic_ik.py \
  --target-mode fk-demo
```

Solve IK for an explicit TCP pose:

```bash
conda run -n mp_env python motion_planning/example_crane_analytic_ik.py \
  --target-mode pose \
  --target-pos "-8.60268,-3.194779,3.604759" \
  --target-quat-wxyz "-0.000922,0.97471,-0.223468,0.000906"
```

Notes:
- Telescope is modeled as a single DoF (`q4_big_telescope`), while `q5_small_telescope` is tied internally (`q5=q4`) for URDF compatibility.
- `--solve-passive` enables passive-joint equilibrium coupling (uses analytic dynamics).
- Numeric IK in `CraneKinematics` has been removed; use `motion_planning.mechanics.analytic.AnalyticInverseKinematics` for IK workflows.

## Inspect URDF Model Info (Frames/Joints)

Primary (analytic mechanics): print full model information (joints + frames):

```bash
conda run -n mp_env python -c "from motion_planning.mechanics.analytic import AnalyticModelConfig, ModelDescription; cfg=AnalyticModelConfig.default(); d=ModelDescription(cfg); d.print_info()"
```

Print only frame names (useful to choose `base_frame` / `end_frame` for kinematics):

```bash
conda run -n mp_env python -c "from motion_planning.mechanics.analytic import AnalyticModelConfig, ModelDescription; cfg=AnalyticModelConfig.default(); d=ModelDescription(cfg); print('\n'.join([f['name'] for f in d.frame_info()]))"
```

Fallback (Pinocchio wrapper) command:

```bash
conda run -n mp_env python -c "from motion_planning.kinematics import CraneKinematics; k=CraneKinematics('crane_urdf/crane.urdf'); k.print_model_info()"
```
