"""VP-STO path planning standalone stack.

Uses CMA-ES-optimized B-spline trajectory (ported from timber VP-STO)
for the 5 actuated crane joints.
"""

from __future__ import annotations

import numpy as np

from motion_planning.pipeline import JointGoalStage, JointSpaceCartesianPlanner
from motion_planning.planners.vpsto.crane_model import CraneVpstoModel
from motion_planning.planners.vpsto.solver import VpstoSolver

from ..evaluate import evaluate_plan
from ..types import StandalonePlanResult, StandaloneScenario


def plan_vpsto_path_planning(scenario: StandaloneScenario) -> StandalonePlanResult:
    stage = JointGoalStage()
    planner = JointSpaceCartesianPlanner(
        urdf_path=stage.config.urdf_path,
        target_frame=stage.config.target_frame,
        reduced_joint_names=stage.config.actuated_joints,
    )
    act_names = list(stage.config.actuated_joints)

    # ---- Resolve start joint config ----
    if scenario.planner_start_q is not None:
        q_start = np.asarray(scenario.planner_start_q, dtype=float)
    else:
        start_solve = stage.solve_world_pose(
            goal_world=scenario.start_world_xyz,
            target_yaw_rad=scenario.start_yaw_rad,
            q_seed={},
        )
        if not start_solve.success:
            return _fail("start solve failed: " + start_solve.message, act_names)
        q_start = np.asarray(
            [start_solve.q_actuated.get(name, 0.0) for name in act_names],
            dtype=float,
        )

    # ---- VP-STO target: [x, y, z, yaw] ----
    yd = np.array([
        scenario.goal_world_xyz[0],
        scenario.goal_world_xyz[1],
        scenario.goal_world_xyz[2],
        scenario.goal_yaw_rad,
    ], dtype=float)

    # ---- Run VP-STO ----
    model = CraneVpstoModel(n_eval=20)
    solver = VpstoSolver(model, n_via=5, n_eval=20, n_samples=32)

    vpsto_result = solver.solve(q_start, yd)
    if not vpsto_result.success:
        return _fail("VP-STO failed to find feasible path", act_names)

    q_waypoints = vpsto_result.q_traj

    # ---- FK for TCP path ----
    xyz_list = []
    yaw_list = []
    q_seed = {name: float(q_start[i]) for i, name in enumerate(act_names)}
    for q in q_waypoints:
        xyz_i, yaw_i, q_seed = planner.fk_world_pose(q, q_seed=q_seed)
        xyz_list.append(xyz_i)
        yaw_list.append(yaw_i)

    # Reference: straight line from actual FK start to intended goal
    fk_start = np.asarray(xyz_list[0], dtype=float)
    goal_xyz = np.asarray(scenario.goal_world_xyz, dtype=float)
    n = len(q_waypoints)
    reference_xyz = np.vstack([
        np.linspace(fk_start[i], goal_xyz[i], n, dtype=float)
        for i in range(3)
    ]).T
    reference_yaw = np.linspace(
        float(yaw_list[0]), scenario.goal_yaw_rad, n, dtype=float,
    )

    result = StandalonePlanResult(
        stack_name="vpsto_path_planning",
        success=True,
        message=f"VP-STO: T={vpsto_result.T:.2f}s, {vpsto_result.iterations} CMA-ES iters, cost={vpsto_result.cost:.3f}",
        q_waypoints=q_waypoints,
        tcp_xyz=np.asarray(xyz_list, dtype=float),
        tcp_yaw_rad=np.asarray(yaw_list, dtype=float),
        reference_xyz=reference_xyz,
        reference_yaw_rad=reference_yaw,
        diagnostics={
            "vpsto_T": vpsto_result.T,
            "vpsto_cost": vpsto_result.cost,
            "vpsto_iterations": float(vpsto_result.iterations),
            "waypoint_count": float(len(q_waypoints)),
        },
    )
    evaluate_plan(result)
    return result


def _fail(msg: str, act_names: list[str]) -> StandalonePlanResult:
    n = len(act_names)
    return StandalonePlanResult(
        stack_name="vpsto_path_planning",
        success=False,
        message=msg,
        q_waypoints=np.zeros((0, n), dtype=float),
        tcp_xyz=np.zeros((0, 3), dtype=float),
        tcp_yaw_rad=np.zeros(0, dtype=float),
        reference_xyz=np.zeros((0, 3), dtype=float),
        reference_yaw_rad=np.zeros(0, dtype=float),
        diagnostics={},
    )
