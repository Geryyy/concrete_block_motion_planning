# Timber Compatibility Contract

This note records the timber motion-planning structure that the CBS stack
should stay compatible with. It is intentionally CBS-local documentation:
the goal is to mirror the proven timber structure from the concrete/CBS side
without changing the timber workspace unless that becomes strictly necessary.

## Why This Exists

The timber and concrete backends should be interchangeable at the BT/service
boundary:

- the same BT nodes call into either backend
- the same execution service executes trajectories from either backend
- the same visualization and validation layers can inspect the resulting motion

To make that possible, the stack needs a clear answer to three separate
questions:

1. What is the backend's full-state trajectory contract?
2. What is the controller's commanded-joint subset?
3. Where is the mapping between those two contracts defined?

Timber already answers those questions consistently. Concrete/CBS should
replicate that structure, not invent a different one.

## Timber Reference Structure

### Service and BT surface

The timber BT requests an A-to-B movement, receives a `JointTrajectory`, and
then executes it through `/trajectory_controller_a2b/follow_joint_trajectory`.

Key files:

- `src/epsilon_crane_behavior_tree/behavior_trees/subtree_a2b_movement.xml`
- `src/epsilon_crane_behavior_tree/src/plugins/action/calc_a2b_movement.cpp`
- `src/timber_crane_motion_planning/src/a2b_server_base.cpp`
- `src/timber_crane_planning_interfaces/srv/CalcMovement.srv`

### Full-state trajectory contract

The timber planner returns an 8-slot full-state `JointTrajectory` in this order:

1. `theta1_slewing_joint`
2. `theta2_boom_joint`
3. `theta3_arm_joint`
4. `q4_big_telescope`
5. `theta6_tip_joint`
6. `theta7_tilt_joint`
7. `theta8_rotator_joint`
8. `theta10_outer_jaw_joint`

This order is serialized in:

- `src/matlab_codegen/mp_tools/src/mp_tools.cpp`

### Controller command contract

The timber ros2_control setup distinguishes between:

- `joints`: full 8-joint state/trajectory space
- `command_joints`: reduced commanded subset

For the timber hydraulic controller, the commanded subset is:

1. `theta1_slewing_joint`
2. `theta2_boom_joint`
3. `theta3_arm_joint`
4. `q4_big_telescope`
5. `theta8_rotator_joint`
6. `theta10_outer_jaw_joint`

This is defined in:

- `src/epsilon_crane_bringup_sim/config/ros2_control/crane_controller_hydraulic_common.ros2_control.yaml`

### Passive and derived joints

`theta6_tip_joint` and `theta7_tilt_joint` are part of the trajectory/state
contract, but they are not directly commanded through ros2_control.

The timber controller plugin reads the full 8-joint trajectory/state, then maps
the reduced command space into the commanded interfaces internally.

Key files:

- `src/timber_crane_cpp/control/timber_crane_mpc/src/jtc_mpc_plugin.cpp`
- `src/timber_crane_cpp/control/timber_crane_mpc/src/jtc_mpc_plugin_parameters.yaml`

The important structural pattern is:

- full-state trajectory contract is bigger than command space
- passive/derived joints remain visible for state, planning, and FK
- only the actuated subset is sent to controller interfaces
- the mapping is explicit and centralized

## Concrete PZS100 Compatibility Goal

Concrete should preserve the same structural pattern, even though the PZS100
tool has different gripper joint names.

For the current CBS setup:

- full-state trajectory contract:
  - `theta1_slewing_joint`
  - `theta2_boom_joint`
  - `theta3_arm_joint`
  - `q4_big_telescope`
  - `theta6_tip_joint`
  - `theta7_tilt_joint`
  - `theta8_rotator_joint`
  - `q9_left_rail_joint`

- commanded-joint subset:
  - `theta1_slewing_joint`
  - `theta2_boom_joint`
  - `theta3_arm_joint`
  - `q4_big_telescope`
  - `theta8_rotator_joint`
  - `q9_left_rail_joint`

- mimicked tool joint:
  - `q11_right_rail_joint` mimics `q9_left_rail_joint`

Implications:

- `theta6_tip_joint` and `theta7_tilt_joint` stay state-only
- the second rail stays derived/mimicked, not independently commanded
- the backend may plan in a richer state space, but the controller adapter must
  always know the commanded subset explicitly

## Compatibility Rule

Concrete/CBS should match timber structurally:

- one explicit full-state trajectory contract
- one explicit commanded-joint subset
- one explicit mapping layer between them

Concrete/CBS should *not* copy timber's exact raw joint names. Instead it
should mirror the timber structure using backend-specific profiles.

## Practical Refactor Constraint

The timber workspace under `src/...` should stay untouched unless that proves
strictly necessary for interchangeability. All compatibility work should start
inside:

- `src/concrete_block_stack/concrete_block_motion_planning/...`

Only if the timber side exposes a hard-coded assumption that cannot be matched
from CBS should we consider a timber-side change, and then only on `feat/cbs`.
