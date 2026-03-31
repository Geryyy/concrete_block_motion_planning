from __future__ import annotations

from pathlib import Path

import numpy as np

from motion_planning.pipeline import JointSpaceGlobalPathRequest
from motion_planning.trajectory.planning_limits import load_planning_limits_yaml

from ..common import StandalonePlanningContext, build_planning_context, fail_result
from ..evaluate import evaluate_plan
from ..types import StandalonePlanResult, StandaloneScenario


def plan_joint_space_global_path(scenario: StandaloneScenario) -> StandalonePlanResult:
    planning = build_planning_context(
        scenario,
        stack_name="joint_space_global_path",
    )
    if isinstance(planning, StandalonePlanResult):
        return planning
    assert isinstance(planning, StandalonePlanningContext)

    repo_root = Path(__file__).resolve().parents[3]
    planning_limits = repo_root / "motion_planning" / "trajectory" / "planning_limits.yaml"
    joint_limits, _ = load_planning_limits_yaml(planning_limits)
    planning.planner._joint_position_limits = dict(joint_limits)

    result = planning.planner.plan_global_path(
        JointSpaceGlobalPathRequest(
            scene=_overlay_scene_for_scenario(scenario),
            moving_block_size=_moving_block_size_for_scenario(scenario),
            q_start=planning.q_start,
            q_goal=planning.q_goal,
            q_start_seed_map=planning.q_start_seed_map,
            q_goal_seed_map=planning.q_goal_seed_map,
            start_approach_direction_world=scenario.start_approach_direction_world,
            goal_approach_direction_world=scenario.goal_approach_direction_world,
        )
    )
    if not result.success and result.q_waypoints.size == 0:
        return fail_result(
            stack_name="joint_space_global_path",
            actuated_joint_count=len(planning.actuated_joint_names),
            message=result.message,
            diagnostics=result.diagnostics,
        )

    plan_result = StandalonePlanResult(
        stack_name="joint_space_global_path",
        success=result.success,
        message=result.message,
        q_waypoints=np.asarray(result.q_waypoints, dtype=float),
        tcp_xyz=np.asarray(result.tcp_xyz, dtype=float),
        tcp_yaw_rad=np.asarray(result.tcp_yaw_rad, dtype=float),
        reference_xyz=np.asarray(result.tcp_xyz, dtype=float),
        reference_yaw_rad=np.asarray(result.tcp_yaw_rad, dtype=float),
        diagnostics=dict(result.diagnostics),
    )
    if plan_result.q_waypoints.size > 0:
        evaluate_plan(plan_result)
    return plan_result


def _overlay_scene_for_scenario(scenario: StandaloneScenario):
    from motion_planning.scenarios import ScenarioLibrary

    if scenario.overlay_scene_name is None:
        return ScenarioLibrary().build_scenario("step_01_first_on_ground").scene
    return ScenarioLibrary().build_scenario(scenario.overlay_scene_name).scene


def _moving_block_size_for_scenario(scenario: StandaloneScenario) -> tuple[float, float, float]:
    from motion_planning.scenarios import ScenarioLibrary

    if scenario.overlay_scene_name is None:
        return ScenarioLibrary().build_scenario("step_01_first_on_ground").moving_block_size
    cfg = ScenarioLibrary().build_scenario(scenario.overlay_scene_name)
    return tuple(float(v) for v in cfg.moving_block_size)
