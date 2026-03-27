from __future__ import annotations

import math

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

# Hover start XYZ/yaw (K0 frame, from FK above)
_XYZ_HOVER_CENTER = (0.841,  0.046, -2.874)
_XYZ_HOVER_RIGHT  = (0.814,  0.224, -2.874)
_XYZ_HOVER_LEFT   = (0.832, -0.133, -2.874)
_YAW_90 = math.radians(90.0)

# Map: ScenarioLibrary name → (planner_start_q, start_xyz, start_yaw)
_CBS_SCENARIO_STARTS: dict[str, tuple] = {
    "step_01_first_on_ground":    (_Q_HOVER_CENTER, _XYZ_HOVER_CENTER, _YAW_90),
    "step_02_second_beside_first":(_Q_HOVER_RIGHT,  _XYZ_HOVER_RIGHT,  _YAW_90),
    "step_03_third_on_top":       (_Q_HOVER_CENTER, _XYZ_HOVER_CENTER, _YAW_90),
    "step_04_between_two_blocks": (_Q_HOVER_CENTER, _XYZ_HOVER_CENTER, _YAW_90),
}


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
            q_start, start_xyz, start_yaw = start_entry
            goal_yaw = math.radians(float(cfg.goal_yaw_deg))
        else:
            # Fallback: no validated start config — scenario will attempt IK
            q_start = None
            start_xyz = tuple(float(v) for v in cfg.start)
            start_yaw = math.radians(float(cfg.start_yaw_deg))
            goal_yaw = math.radians(float(cfg.goal_yaw_deg))

        scenarios.append(
            StandaloneScenario(
                name=f"scene_{scene_name}",
                description=f"CBS crane scenario: {scene_name}",
                start_world_xyz=start_xyz,
                goal_world_xyz=tuple(float(v) for v in cfg.goal),
                start_yaw_rad=start_yaw,
                goal_yaw_rad=goal_yaw,
                planner_start_q=q_start,
                anchor_count=6,
                overlay_scene_name=scene_name,
            )
        )

    return {sc.name: sc for sc in scenarios}
