from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motion_planning.pipeline import JointGoalStage, JointSpaceCartesianPlanner

from .types import StandalonePlanResult, StandaloneScenario


@dataclass(frozen=True)
class StandalonePlanningContext:
    stage: JointGoalStage
    planner: JointSpaceCartesianPlanner
    actuated_joint_names: tuple[str, ...]
    q_start: np.ndarray
    q_goal: np.ndarray
    q_start_seed_map: dict[str, float]
    q_goal_seed_map: dict[str, float]


def fail_result(
    *,
    stack_name: str,
    actuated_joint_count: int,
    message: str,
    diagnostics: dict[str, float | str] | None = None,
) -> StandalonePlanResult:
    return StandalonePlanResult(
        stack_name=stack_name,
        success=False,
        message=message,
        q_waypoints=np.zeros((0, actuated_joint_count), dtype=float),
        tcp_xyz=np.zeros((0, 3), dtype=float),
        tcp_yaw_rad=np.zeros(0, dtype=float),
        reference_xyz=np.zeros((0, 3), dtype=float),
        reference_yaw_rad=np.zeros(0, dtype=float),
        diagnostics={} if diagnostics is None else dict(diagnostics),
    )


def build_planning_context(
    scenario: StandaloneScenario,
    *,
    stack_name: str,
) -> StandalonePlanningContext | StandalonePlanResult:
    stage = JointGoalStage()
    planner = JointSpaceCartesianPlanner(
        urdf_path=stage.config.urdf_path,
        target_frame=stage.config.target_frame,
        reduced_joint_names=stage.config.actuated_joints,
    )
    actuated_joint_names = tuple(str(name) for name in stage.config.actuated_joints)

    def _complete_seed(q_red: np.ndarray, q_seed: dict[str, float]) -> dict[str, float]:
        solved = stage._steady_state.complete_from_actuated(
            {name: float(q_red[i]) for i, name in enumerate(actuated_joint_names)},
            q_seed=q_seed,
        )
        return dict(solved.q_dynamic) if solved.success else dict(q_seed)

    q_start: np.ndarray | None = None
    start_seed: dict[str, float] = {}
    if scenario.planner_start_q is not None:
        q_start = np.asarray(scenario.planner_start_q, dtype=float)
        start_seed = dict(scenario.planner_start_q_seed_map or {
            name: float(q_start[i]) for i, name in enumerate(actuated_joint_names)
        })
        start_seed = _complete_seed(q_start, start_seed)
    else:
        start_solve = stage.solve_world_pose(
            goal_world=scenario.start_world_xyz,
            target_yaw_rad=scenario.start_yaw_rad,
            q_seed={},
        )
        if not start_solve.success:
            return fail_result(
                stack_name=stack_name,
                actuated_joint_count=len(actuated_joint_names),
                message=f"start solve failed: {start_solve.message}",
            )
        q_start = np.asarray(
            [start_solve.q_actuated.get(name, 0.0) for name in actuated_joint_names],
            dtype=float,
        )
        start_seed = dict(start_solve.q_dynamic)

    if scenario.planner_goal_q is not None:
        q_goal = np.asarray(scenario.planner_goal_q, dtype=float)
        goal_seed = dict(scenario.planner_goal_q_seed_map or start_seed)
        goal_seed = _complete_seed(q_goal, goal_seed)
    else:
        goal_solve = stage.solve_world_pose(
            goal_world=scenario.goal_world_xyz,
            target_yaw_rad=scenario.goal_yaw_rad,
            q_seed=start_seed,
        )
        if not goal_solve.success:
            return fail_result(
                stack_name=stack_name,
                actuated_joint_count=len(actuated_joint_names),
                message=f"goal solve failed: {goal_solve.message}",
            )
        q_goal = np.asarray(
            [goal_solve.q_actuated.get(name, 0.0) for name in actuated_joint_names],
            dtype=float,
        )
        goal_seed = dict(goal_solve.q_dynamic)

    if q_start is None:
        return fail_result(
            stack_name=stack_name,
            actuated_joint_count=len(actuated_joint_names),
            message="failed to initialize start joint state",
        )

    return StandalonePlanningContext(
        stage=stage,
        planner=planner,
        actuated_joint_names=actuated_joint_names,
        q_start=q_start,
        q_goal=q_goal,
        q_start_seed_map=start_seed,
        q_goal_seed_map=goal_seed,
    )


def evaluate_tcp_path(
    planner: JointSpaceCartesianPlanner,
    actuated_joint_names: tuple[str, ...],
    q_waypoints: np.ndarray,
    *,
    q_seed_map: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = []
    yaw = []
    seed_map = (
        dict(q_seed_map)
        if q_seed_map is not None
        else {name: float(q_waypoints[0, i]) for i, name in enumerate(actuated_joint_names)}
    )
    for q in np.asarray(q_waypoints, dtype=float):
        xyz_i, yaw_i, seed_map = planner.fk_world_pose(q, q_seed=seed_map)
        xyz.append(xyz_i)
        yaw.append(yaw_i)
    return np.asarray(xyz, dtype=float), np.asarray(yaw, dtype=float)
