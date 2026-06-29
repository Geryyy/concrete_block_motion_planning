# concrete_block_motion_planning

Grip-trajectory planner for the concrete-block pick-and-place pipeline. Owns the `grip_traj_server` node, which turns the gripper's descend / close / open / lift phases into joint trajectories, backed by the vendored `cbmp` kinematics/mechanics backend. Driven by the grip subtrees in [concrete_block_behavior_tree](../concrete_block_behavior_tree/).

> Wall-plan / task sequencing is **not** here — it lives in [concrete_block_assembly_planning](../concrete_block_assembly_planning/). The long-range point-to-point ("A2B") move service (`a2b_movement`) is served by the **timber_crane** stack, not this package.

## Responsibilities

- Plan the short, near-vertical gripper trajectories around a grasp: descend to CoG, close, open/release, lift away.
- Respect payload geometry/mass so the planner accounts for a carried block.
- Publish the resulting TCP path for visualization.

## Contents

```text
config/grip_traj_simple.yaml   Planner parameters (joints, dt, lift height, gripper open/close)
motion_planning/               Python backend (cbmp): mechanics, data, kinematics helpers
scripts/
  grip_traj_server_simple.py   ROS 2 node: serves grip_traj_movement
  grip_trajectory.py           Trajectory generation (compute_grip_trajectory)
utils/gripper_inertia.py
```

## ROS interface

| Name | Type | Direction | Purpose |
|---|---|---|---|
| `grip_traj_movement` | `timber_crane_planning_interfaces/CalcGripMovement` | **server** | BT requests a phase trajectory (descend / close / open / lift) |
| `/joint_states` | `sensor_msgs/JointState` | subscriber | current configuration seed |
| `tcp_path` | `nav_msgs/Path` | publisher | planned tool-center path (RViz) |

## Dependencies & interactions

- **Interface dep:** `timber_crane_planning_interfaces` (`CalcGripMovement` srv) — this package deliberately does **not** depend on the CBS world-model interfaces; it is a pure trajectory service.
- **Other deps:** `trajectory_msgs`, `nav_msgs`, `sensor_msgs`, `geometry_msgs`, `rclpy`.
- **Consumed by:** [concrete_block_behavior_tree](../concrete_block_behavior_tree/) — `SubTreeDescendTo`, `SubTreeGripper`, `SubTreeLift` all call `grip_traj_movement`; the resulting trajectory is handed to the controllers by `SubTreeExecuteTrajectory`.

## Build

```bash
colcon build --packages-select concrete_block_motion_planning --symlink-install
source install/setup.bash
```

The backend depends on CasADi, Pinocchio and acados (installed in the devcontainer).
