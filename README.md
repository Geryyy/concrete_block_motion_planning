# concrete_block_motion_planning

Wall plan server and grip trajectory planner for concrete block assembly. Provides the motion-planning side of the BT pick-and-place pipeline driven by [concrete_block_behavior_tree](../concrete_block_behavior_tree/).

## Contents

```text
config/   Wall plans and planner parameters
motion_planning/  Python backend (cbmp): trajectory generation, kinematics, optimization
scripts/  ROS 2 entrypoints
  wall_plan_server.py         Serves the next assembly task to the BT
  grip_traj_server_simple.py  Plans grip / approach trajectories
  grip_trajectory.py
srv/      GetNextAssemblyTask.srv
utils/
```

## ROS interface

| Service / Topic | Type | Direction | Purpose |
|---|---|---|---|
| `~/get_next_assembly_task` | `GetNextAssemblyTask` | server | BT pulls the next block to place |

Consumes block state from `concrete_block_world_model_interfaces` and emits trajectories via `timber_crane_planning_interfaces`.

## Build

```bash
colcon build --packages-select concrete_block_motion_planning --symlink-install
source install/setup.bash
```

The backend depends on CasADi, Pinocchio, and acados (installed in the devcontainer).
