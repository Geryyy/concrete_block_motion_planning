from __future__ import annotations

import math
import numpy as np

from motion_planning.pipeline import JointGoalStage

from .types import SolverComparisonResult, StandaloneScenario


def _wrap_to_pi(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def compare_concrete_solver(
    scenario: StandaloneScenario,
    *,
    point: str = "goal",
) -> SolverComparisonResult:
    stage = JointGoalStage()
    target_xyz = scenario.goal_world_xyz if point == "goal" else scenario.start_world_xyz
    target_yaw = scenario.goal_yaw_rad if point == "goal" else scenario.start_yaw_rad
    planner_seed_q = scenario.planner_goal_q if point == "goal" else scenario.planner_start_q
    q_seed = {}
    if planner_seed_q is not None:
        q_seed = {
            name: float(planner_seed_q[i])
            for i, name in enumerate(stage.config.actuated_joints)
        }
    solve = stage.solve_world_pose(
        goal_world=target_xyz,
        target_yaw_rad=target_yaw,
        q_seed=q_seed,
    )

    q_act = np.asarray([float(solve.q_actuated.get(name, 0.0)) for name in stage.config.actuated_joints], dtype=float)
    q_dyn = dict(solve.q_dynamic)
    for follower, leader in stage.config.tied_joints.items():
        if leader in q_dyn:
            q_dyn[follower] = float(q_dyn[leader])

    fk = stage._steady_state._ik._analytic._fk(  # intentional standalone debug probe
        q_dyn,
        base_frame=stage.config.base_frame,
        end_frame=stage.config.target_frame,
    )
    fk_xyz = np.asarray(fk[:3, 3], dtype=float)
    fk_yaw = float(math.atan2(fk[1, 0], fk[0, 0]))
    pos_err = float(np.linalg.norm(fk_xyz - np.asarray(target_xyz, dtype=float)))
    yaw_err_deg = float(np.degrees(abs(_wrap_to_pi(fk_yaw - float(target_yaw)))))

    metadata = {
        "target_xyz": tuple(float(v) for v in target_xyz),
        "target_yaw_rad": float(target_yaw),
        "fk_xyz_base": tuple(float(v) for v in solve.fk_xyz_base),
        "fk_yaw_rad": float(solve.fk_yaw_rad),
        "passive_residual": float(solve.passive_residual),
    }
    return SolverComparisonResult(
        name=f"concrete_{point}",
        success=bool(solve.success),
        message=solve.message,
        q_actuated=q_act,
        q_dynamic=q_dyn,
        fk_xyz=fk_xyz,
        fk_yaw_rad=fk_yaw,
        position_error_m=pos_err,
        yaw_error_deg=yaw_err_deg,
        ik_backend=str(solve.ik_backend),
        metadata=metadata,
    )


def compare_solver_suite(scenario: StandaloneScenario) -> list[SolverComparisonResult]:
    results = [
        compare_concrete_solver(scenario, point="start"),
        compare_concrete_solver(scenario, point="goal"),
    ]
    return results
