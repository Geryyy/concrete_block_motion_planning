# URDF ↔ MJCF Conversion

This module provides tools to:

- compile URDF to MJCF using MuJoCo's parser,
- synchronize MJCF body inertials from URDF inertials,
- compare URDF and MJCF for inertia, kinematics, and dynamics consistency.

## CLI

Run via:

```bash
python -m motion_planning.conversion <command> ...
```

Available commands:

- `compile`
- `sync-inertias`
- `compare-inertia`
- `compare-kinematics`
- `compare-dynamics`
- `pipeline`
- `check-sync`

## Examples

Compile URDF to MJCF:

```bash
python -m motion_planning.conversion compile \
  --urdf crane_urdf/crane.urdf \
  --out /tmp/crane_compiled.xml
```

Synchronize inertials from URDF into MJCF:

```bash
python -m motion_planning.conversion sync-inertias \
  --urdf crane_urdf/crane.urdf \
  --mjcf /tmp/crane_compiled.xml \
  --out /tmp/crane_synced.xml
```

Compare inertias:

```bash
python -m motion_planning.conversion compare-inertia \
  --urdf crane_urdf/crane.urdf \
  --mjcf /tmp/crane_synced.xml
```

Run complete pipeline:

```bash
python -m motion_planning.conversion pipeline \
  --urdf crane_urdf/crane.urdf \
  --out-dir /tmp/motion_planning_conversion \
  --samples 30 \
  --seed 0
```

JSON output:

```bash
python -m motion_planning.conversion pipeline \
  --urdf crane_urdf/crane.urdf \
  --out-dir /tmp/motion_planning_conversion \
  --json
```

Check that committed `crane.xml` stays synchronized with `crane.urdf`
for the model core (ignores scene, actuators, and marker sites):

```bash
python -m motion_planning.conversion check-sync \
  --urdf crane_urdf/crane.urdf \
  --mjcf crane_urdf/crane.xml
```

## Python API

See:

- `motion_planning.conversion.compile_urdf_to_mjcf`
- `motion_planning.conversion.synchronize_mjcf_inertials_from_urdf`
- `motion_planning.conversion.compare_urdf_inertials_to_mjcf`
- `motion_planning.conversion.compare_pin_models_kinematics`
- `motion_planning.conversion.compare_pin_models_dynamics`
