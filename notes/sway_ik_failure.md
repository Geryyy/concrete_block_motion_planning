# TODO: descend IK aborts when the load is still swaying

Observed 2026-08-05, Gazebo PZS100 operator-guided wall build
(`operator_guided_single_block`). The tree reached `place_from_hover` and died
immediately:

```
[grip_traj_server] Grip request | phase=1 current_K8=(6.06, 3.41, 1.65) target_K8=(6.06, 3.35, 0.46) ...
[grip_traj_server] ERROR IK failed: IK returned None
[grip_traj_server] ERROR Trajectory generation failed: IK failed for descend target
[bt_action_server] ERROR CalcGripMovementService: Service call failed, code 0
-> Plan descend FAILURE -> descend_to -> place_from_hover -> tree FAILURE
```

There was visible residual sway at the moment "Place block" was pressed.

## Why sway can make this IK infeasible

The passive joints are the two sway DOFs, `theta6_tip_joint` and
`theta7_tilt_joint` (`motion_planning/mechanics/model_description.py:42`),
i.e. the pendulum the gripper hangs from, with a 0.95 m rail below K8
(`tcp_z_offset`).

`_ik_solve` pins them at their **measured** values and solves with only the five
actuated joints (`scripts/grip_traj_server_simple.py:343-360`):

```python
measured_passive = {name: seed[name] for name in self._config.passive_joints if name in seed}
fixed = measured_passive
result = self._ik_solver.solve(..., act_names=IK_ACTUATED, fixed=fixed)
```

`IK_ACTUATED = [theta1_slewing, theta2_boom, theta3_arm, q4_big_telescope,
theta8_rotator]`. So the solver must hit the target pose exactly, with 5 DOF,
while the pendulum is frozen mid-swing. A few degrees of tip/tilt moves the tool
point several cm on a 0.95 m rail; a target that solves at equilibrium can be
infeasible while swinging.

## The real defect: the steady-state fallback short-circuits

`use_passive_steady_state: true` is set for descend in
`config/grip_traj_simple.yaml`, but the loop breaks before it can help
(`scripts/grip_traj_server_simple.py:352-364`):

```python
for _ in range(self._passive_steady_state_iterations):   # 3
    result = self._ik_solver.solve(...)      # iteration 1 uses MEASURED passive angles
    solved_fixed = fixed
    if result is None or not result.success or not use_passive_steady_state:
        break        # first failure exits -- passive_joint_equilibrium() never runs
```

`passive_joint_equilibrium()` — the function that would supply the settled
pendulum state — is only reached **after a successful solve**. So exactly in the
case it was written for (a bad measured passive state) it is skipped.

## Candidate fixes

- [ ] On first-solve failure with `use_passive_steady_state`, retry once from
      `passive_joint_equilibrium()` instead of breaking. Cheapest fix, keeps the
      measured-state fast path.
- [ ] Gate the descend on settling: `/grip_traj/passive_settled` already exists
      (`passive_settle_velocity_threshold_radps: 0.02`, `passive_settle_window_s:
      0.75`) but is **informational only** and blocks nothing. Either have the BT
      wait on it before `SubTreeDescendTo`, or have the service return a
      distinguishable "not settled" code so the tree can retry rather than abort.
- [ ] Distinguish "IK infeasible" from "not settled" in the service response —
      today both surface as `code 0` / `IK returned None`, which is why the tree
      aborts the whole goal instead of retrying.
- [ ] Consider raising `duration_descend` / `duration_lift` (currently 5.0 s);
      the config comment already flags these as the first knob when lift-off
      excites passive-joint sway.

## Not yet ruled out: plain reach limit

The failing target is meaningfully further out than the pickup that succeeded in
the same run (frame `K0_mounting_base`):

| | K8 target | radius |
|---|---|---|
| pickup (OK) | (6.06, 0.35, 0.31) | 6.07 m |
| place (failed) | (6.06, 3.35, 0.46) | **6.92 m** |

~0.85 m further out radially — `q4_big_telescope` may simply be at its limit.

**Test that separates the two:** wait for `/grip_traj/passive_settled == true`,
then press "5. Place block" again.
Succeeds -> sway. Fails identically -> reach limit, and the wall origin/slot
needs to move inward.

Do this before implementing any of the fixes above — it decides whether this is
an IK-robustness task or a workspace-layout task.
