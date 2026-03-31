from __future__ import annotations

import math
import numpy as np

from motion_planning.scenarios import ScenarioLibrary

from .types import StandaloneScenario


# ---------------------------------------------------------------------------
# Validated joint configs for CBS block-stacking scenarios.
# Positions are in K0_mounting_base frame (computed offline via FK).
# Joint order: [theta1, theta2, theta3, q4, theta8]
# ---------------------------------------------------------------------------

# Hover positions (z ≈ -2.87 m — TCP high above table)
_Q_HOVER_CENTER = (0.05,  -0.80,  0.55, 0.40, -0.05)   # FK → (0.841, 0.046, -2.874)
_Q_HOVER_RIGHT  = (0.25,  -0.80,  0.55, 0.40, -0.25)   # FK → (0.814, 0.224, -2.874)
_Q_HOVER_LEFT   = (-0.15, -0.80,  0.55, 0.40,  0.15)   # FK → (0.832, -0.133, -2.874)

# Placement positions (z ≈ -3.65 m — TCP at table/block surface)
_Q_GOAL_CENTER = (0.05,  -0.92,  0.68, 0.65, -0.05)    # FK → (0.443,  0.026, -3.647)
_Q_GOAL_RIGHT  = (0.25,  -0.92,  0.68, 0.65, -0.25)    # FK → (0.428,  0.125, -3.647)
_Q_GOAL_LEFT   = (-0.15, -0.92,  0.68, 0.65,  0.15)    # FK → (0.438, -0.073, -3.647)
_Q_GOAL_ON_TOP = (0.05,  -0.86,  0.62, 0.65, -0.05)    # FK → (0.605,  0.034, -3.515)

# Hover start XYZ/yaw (K0 frame, from FK above)
_XYZ_HOVER_CENTER = (0.841,  0.046, -2.874)
_XYZ_HOVER_RIGHT  = (0.814,  0.224, -2.874)
_XYZ_HOVER_LEFT   = (0.832, -0.133, -2.874)
_YAW_90 = math.radians(90.0)

# Placement XYZ (K0 frame, from FK of goal configs above)
_XYZ_GOAL_CENTER = (0.443,  0.026, -3.647)
_XYZ_GOAL_RIGHT  = (0.428,  0.125, -3.647)
_XYZ_GOAL_LEFT   = (0.438, -0.073, -3.647)
_XYZ_GOAL_ON_TOP = (0.605,  0.034, -3.515)

# Map: ScenarioLibrary name → (start_q, start_xyz, start_yaw, goal_q, goal_xyz)
_CBS_SCENARIO_STARTS: dict[str, tuple] = {
    "step_01_first_on_ground":    (_Q_HOVER_CENTER, _XYZ_HOVER_CENTER, _YAW_90, _Q_GOAL_CENTER, _XYZ_GOAL_CENTER),
    "step_02_second_beside_first":(_Q_HOVER_RIGHT,  _XYZ_HOVER_RIGHT,  _YAW_90, _Q_GOAL_RIGHT,  _XYZ_GOAL_RIGHT),
    "step_03_third_on_top":       (_Q_HOVER_CENTER, _XYZ_HOVER_CENTER, _YAW_90, _Q_GOAL_ON_TOP, _XYZ_GOAL_ON_TOP),
    "step_04_between_two_blocks": (_Q_HOVER_CENTER, _XYZ_HOVER_CENTER, _YAW_90, _Q_GOAL_CENTER, _XYZ_GOAL_CENTER),
}


def _unit(v: tuple[float, float, float]) -> tuple[float, float, float]:
    arr = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(arr))
    if n <= 1e-12:
        return (0.0, 0.0, -1.0)
    arr = arr / n
    return (float(arr[0]), float(arr[1]), float(arr[2]))


_CURATED_SCENARIOS: tuple[StandaloneScenario, ...] = (
    StandaloneScenario(
        name="short_reachable_move",
        description="Short validated reachable motion in free space for standalone commissioning.",
        start_world_xyz=_XYZ_HOVER_CENTER,
        goal_world_xyz=(0.781, 0.146, -2.874),
        start_yaw_rad=_YAW_90,
        goal_yaw_rad=_YAW_90,
        start_approach_direction_world=_unit((0.781 - _XYZ_HOVER_CENTER[0], 0.146 - _XYZ_HOVER_CENTER[1], -2.874 - _XYZ_HOVER_CENTER[2])),
        goal_approach_direction_world=(0.0, 0.0, -1.0),
        planner_start_q=_Q_HOVER_CENTER,
        planner_goal_q=(0.17, -0.80, 0.55, 0.40, -0.17),
        anchor_count=6,
    ),
    StandaloneScenario(
        name="yaw_change_probe",
        description="Reachable probe that mainly exercises yaw change handling.",
        start_world_xyz=_XYZ_HOVER_CENTER,
        goal_world_xyz=_XYZ_HOVER_LEFT,
        start_yaw_rad=_YAW_90,
        goal_yaw_rad=math.radians(60.0),
        start_approach_direction_world=_unit((_XYZ_HOVER_LEFT[0] - _XYZ_HOVER_CENTER[0], _XYZ_HOVER_LEFT[1] - _XYZ_HOVER_CENTER[1], _XYZ_HOVER_LEFT[2] - _XYZ_HOVER_CENTER[2])),
        goal_approach_direction_world=(0.0, 0.0, -1.0),
        planner_start_q=_Q_HOVER_CENTER,
        planner_goal_q=_Q_HOVER_LEFT,
        anchor_count=6,
    ),
    StandaloneScenario(
        name="single_block_transfer",
        description="Scene-backed block transfer target used to expose current solve limitations.",
        start_world_xyz=_XYZ_HOVER_CENTER,
        goal_world_xyz=_XYZ_GOAL_CENTER,
        start_yaw_rad=_YAW_90,
        goal_yaw_rad=_YAW_90,
        start_approach_direction_world=_unit((_XYZ_GOAL_CENTER[0] - _XYZ_HOVER_CENTER[0], _XYZ_GOAL_CENTER[1] - _XYZ_HOVER_CENTER[1], _XYZ_GOAL_CENTER[2] - _XYZ_HOVER_CENTER[2])),
        goal_approach_direction_world=(0.0, 0.0, -1.0),
        planner_start_q=_Q_HOVER_CENTER,
        planner_goal_q=_Q_GOAL_CENTER,
        anchor_count=6,
        overlay_scene_name="step_01_first_on_ground",
    ),
    StandaloneScenario(
        name="scene_demo_step_01_reachable",
        description="Reachable demo with the step_01 scene overlay and validated seeds.",
        start_world_xyz=_XYZ_HOVER_CENTER,
        goal_world_xyz=(0.781, 0.146, -2.874),
        start_yaw_rad=_YAW_90,
        goal_yaw_rad=_YAW_90,
        start_approach_direction_world=_unit((0.781 - _XYZ_HOVER_CENTER[0], 0.146 - _XYZ_HOVER_CENTER[1], -2.874 - _XYZ_HOVER_CENTER[2])),
        goal_approach_direction_world=(0.0, 0.0, -1.0),
        planner_start_q=_Q_HOVER_CENTER,
        planner_goal_q=(0.17, -0.80, 0.55, 0.40, -0.17),
        anchor_count=6,
        overlay_scene_name="step_01_first_on_ground",
    ),
)


def make_default_scenarios() -> dict[str, StandaloneScenario]:
    """Return the canonical CBS standalone scenario set.

    All scenarios use validated joint configs (``planner_start_q``) so
    the VP-STO planner bypasses IK for the start configuration.
    Goal positions are loaded from ``generated_scenarios.yaml`` (K0 frame).
    """
    scene_lib = ScenarioLibrary()
    scenarios: list[StandaloneScenario] = []

    for scene_name in scene_lib.list_scenarios():
        cfg = scene_lib.build_scenario(scene_name)
        start_entry = _CBS_SCENARIO_STARTS.get(scene_name)

        if start_entry is not None:
            q_start, start_xyz, start_yaw, q_goal, goal_xyz = start_entry
            goal_yaw = math.radians(float(cfg.goal_yaw_deg))
        else:
            # Fallback: no validated configs — scenario will attempt IK
            q_start = None
            q_goal = None
            start_xyz = tuple(float(v) for v in cfg.start)
            goal_xyz = tuple(float(v) for v in cfg.goal)
            start_yaw = math.radians(float(cfg.start_yaw_deg))
            goal_yaw = math.radians(float(cfg.goal_yaw_deg))

        scenarios.append(
            StandaloneScenario(
                name=f"scene_{scene_name}",
                description=f"CBS crane scenario: {scene_name}",
                start_world_xyz=start_xyz,
                goal_world_xyz=goal_xyz,
                start_yaw_rad=start_yaw,
                goal_yaw_rad=goal_yaw,
                start_approach_direction_world=tuple(float(v) for v in cfg.start_approach_direction),
                goal_approach_direction_world=tuple(float(v) for v in cfg.goal_approach_direction),
                planner_start_q=q_start,
                planner_goal_q=q_goal,
                anchor_count=6,
                overlay_scene_name=scene_name,
            )
        )

    out = {sc.name: sc for sc in _CURATED_SCENARIOS}
    out.update({sc.name: sc for sc in scenarios})
    return out
