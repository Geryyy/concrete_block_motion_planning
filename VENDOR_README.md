# Motion Planning

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
