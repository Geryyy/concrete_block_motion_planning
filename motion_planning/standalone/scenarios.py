from __future__ import annotations

import math

import numpy as np

from motion_planning.scenarios import ScenarioLibrary

from .types import StandaloneScenario


def _world_pose_from_actuated(
    q_actuated: tuple[float, float, float, float, float],
) -> tuple[tuple[float, float, float], float]:
    from motion_planning.pipeline import JointGoalStage

    stage = JointGoalStage()
    act_names = list(stage.config.actuated_joints)
    q_map = {name: float(q_actuated[i]) for i, name in enumerate(act_names)}
    completed = stage._steady_state.complete_from_actuated(q_map, q_seed=q_map)
    if not completed.success:
        raise RuntimeError(f"Failed to complete passive equilibrium for scenario seed: {completed.message}")
    q_full = dict(completed.q_dynamic)
    for follower, leader in stage.config.tied_joints.items():
        if leader in q_full:
            q_full[follower] = float(q_full[leader])
    T = stage._steady_state._ik._analytic._fk(
        q_full,
        base_frame=stage.config.base_frame,
        end_frame=stage.config.target_frame,
    )
    xyz = tuple(float(v) for v in np.asarray(T[:3, 3], dtype=float).reshape(3))
    yaw_rad = float(math.atan2(float(T[1, 0]), float(T[0, 0])))
    return xyz, yaw_rad


def make_default_scenarios() -> dict[str, StandaloneScenario]:
    short_start_q = (0.05, -0.85, 0.55, 0.35, -0.05)
    short_goal_q = (0.18, -0.92, 0.62, 0.52, 0.08)
    short_start_xyz, short_start_yaw = _world_pose_from_actuated(short_start_q)
    short_goal_xyz, short_goal_yaw = _world_pose_from_actuated(short_goal_q)

    scenarios = [
        StandaloneScenario(
            name="single_block_transfer",
            description="Default no-Gazebo block-transfer pose pair used for planner debugging.",
            start_world_xyz=(-10.98, -3.71, 2.15),
            goal_world_xyz=(-11.025, 0.240, 3.000),
            start_yaw_rad=0.0,
            goal_yaw_rad=0.0,
            anchor_count=6,
        ),
        StandaloneScenario(
            name="short_reachable_move",
            description="Small Cartesian move to debug IK/FK consistency near a nominal region.",
            start_world_xyz=short_start_xyz,
            goal_world_xyz=short_goal_xyz,
            start_yaw_rad=short_start_yaw,
            goal_yaw_rad=short_goal_yaw,
            planner_start_q=short_start_q,
            planner_goal_q=short_goal_q,
            anchor_count=5,
            overlay_scene_name="step_01_first_on_ground",
        ),
        StandaloneScenario(
            name="yaw_change_probe",
            description="Short move with yaw change to compare IK/steady-state behavior.",
            start_world_xyz=short_start_xyz,
            goal_world_xyz=short_goal_xyz,
            start_yaw_rad=short_start_yaw,
            goal_yaw_rad=short_goal_yaw,
            planner_start_q=short_start_q,
            planner_goal_q=short_goal_q,
            anchor_count=5,
            overlay_scene_name="step_01_first_on_ground",
        ),
    ]
    scene_lib = ScenarioLibrary()
    for scene_name in scene_lib.list_scenarios():
        cfg = scene_lib.build_scenario(scene_name)
        scenarios.append(
            StandaloneScenario(
                name=f"scene_{scene_name}",
                description=f"Standalone task derived from existing scene '{scene_name}'.",
                start_world_xyz=tuple(float(v) for v in cfg.start),
                goal_world_xyz=tuple(float(v) for v in cfg.goal),
                start_yaw_rad=float(math.radians(cfg.start_yaw_deg)),
                goal_yaw_rad=float(math.radians(cfg.goal_yaw_deg)),
                anchor_count=6,
                overlay_scene_name=scene_name,
            )
        )

    # Real scene-backed reachable demo:
    # reuse the existing step_01 block scene, but translate the full scene so its
    # placement goal matches the validated short-range reachable goal pose.
    step01 = scene_lib.build_scenario("step_01_first_on_ground")
    translated_goal = np.asarray(short_goal_xyz, dtype=float)
    original_goal = np.asarray(step01.goal, dtype=float)
    scene_translation = tuple((translated_goal - original_goal).tolist())
    short_delta = np.asarray(short_start_xyz, dtype=float) - np.asarray(short_goal_xyz, dtype=float)
    demo_start = tuple((translated_goal + short_delta).tolist())
    demo_goal = tuple(translated_goal.tolist())
    scenarios.append(
        StandaloneScenario(
            name="scene_demo_step_01_reachable",
            description="Reachable standalone demo tied to the existing step_01 block scene.",
            start_world_xyz=demo_start,
            goal_world_xyz=demo_goal,
            start_yaw_rad=short_start_yaw,
            goal_yaw_rad=short_goal_yaw,
            planner_start_q=short_start_q,
            planner_goal_q=short_goal_q,
            anchor_count=5,
            overlay_scene_name="step_01_first_on_ground",
            overlay_scene_translation=scene_translation,
        )
    )
    return {scenario.name: scenario for scenario in scenarios}
