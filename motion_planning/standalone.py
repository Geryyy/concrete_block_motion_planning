from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from motion_planning.joint_goal_stage import JointGoalStage
from motion_planning.joint_space_global_path import JointSpaceGlobalPathPlanner, JointSpaceGlobalPathRequest
from motion_planning.joint_space_stage import JointSpaceCartesianPlanner
from motion_planning.mechanics.reference_states import load_reference_states
from motion_planning.scenarios import ScenarioLibrary
from motion_planning.trajectory.planning_limits import load_planning_limits_yaml


@dataclass(frozen=True)
class StandaloneScenario:
    name: str
    description: str
    start_world_xyz: tuple[float, float, float]
    goal_world_xyz: tuple[float, float, float]
    start_yaw_rad: float = 0.0
    goal_yaw_rad: float = 0.0
    start_approach_direction_world: tuple[float, float, float] = (0.0, 0.0, -1.0)
    goal_approach_direction_world: tuple[float, float, float] = (0.0, 0.0, -1.0)
    planner_start_q: tuple[float, ...] | None = None
    planner_goal_q: tuple[float, ...] | None = None
    planner_start_q_seed_map: dict[str, float] | None = None
    planner_goal_q_seed_map: dict[str, float] | None = None
    anchor_count: int = 6
    overlay_scene_name: str | None = None
    overlay_scene_translation: tuple[float, float, float] | None = None


@dataclass
class PlanEvaluation:
    final_position_error_m: float
    final_yaw_error_deg: float
    max_position_error_m: float
    mean_position_error_m: float
    max_path_deviation_m: float
    path_length_m: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StandalonePlanResult:
    stack_name: str
    success: bool
    message: str
    q_waypoints: np.ndarray
    tcp_xyz: np.ndarray
    tcp_yaw_rad: np.ndarray
    reference_xyz: np.ndarray
    reference_yaw_rad: np.ndarray
    time_s: np.ndarray | None = None
    dq_waypoints: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    evaluation: PlanEvaluation | None = None


@dataclass(frozen=True)
class _PlanningContext:
    planner: JointSpaceCartesianPlanner
    actuated_joint_names: tuple[str, ...]
    q_start: np.ndarray
    q_goal: np.ndarray
    q_start_seed_map: dict[str, float]
    q_goal_seed_map: dict[str, float]


def _phi_tool_from_reduced_q(q: tuple[float, float, float, float, float]) -> float:
    theta1, _, _, _, theta8 = q
    return math.atan2(math.sin(theta1 - theta8), math.cos(theta1 - theta8))


def _reference_seed(name: str) -> dict[str, float]:
    for state in load_reference_states():
        if state.name == name:
            return dict(state.q_map)
    return {}


def _unit(v: tuple[float, float, float]) -> tuple[float, float, float]:
    arr = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(arr))
    return (0.0, 0.0, -1.0) if n <= 1e-12 else tuple(float(x) for x in (arr / n))


def _wrap_to_pi(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _point_to_polyline_distance(point: np.ndarray, polyline: np.ndarray) -> float:
    p = np.asarray(point, dtype=float).reshape(3)
    line = np.asarray(polyline, dtype=float).reshape(-1, 3)
    if line.shape[0] <= 1:
        return float(np.linalg.norm(p - line[0])) if line.shape[0] == 1 else 0.0
    best = float("inf")
    for a, b in zip(line[:-1], line[1:]):
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            cand = float(np.linalg.norm(p - a))
        else:
            t = max(0.0, min(1.0, float(np.dot(p - a, ab) / denom)))
            cand = float(np.linalg.norm(p - (a + t * ab)))
        best = min(best, cand)
    return best


def _actuated_vector(values: dict[str, float], names: tuple[str, ...]) -> np.ndarray:
    return np.asarray([values.get(name, 0.0) for name in names], dtype=float)


def _fail_result(*, stack_name: str, actuated_joint_count: int, message: str, diagnostics: dict[str, float | str] | None = None) -> StandalonePlanResult:
    return StandalonePlanResult(
        stack_name=stack_name,
        success=False,
        message=message,
        q_waypoints=np.zeros((0, actuated_joint_count), dtype=float),
        tcp_xyz=np.zeros((0, 3), dtype=float),
        tcp_yaw_rad=np.zeros(0, dtype=float),
        reference_xyz=np.zeros((0, 3), dtype=float),
        reference_yaw_rad=np.zeros(0, dtype=float),
        diagnostics=dict(diagnostics or {}),
    )


def evaluate_plan(result: StandalonePlanResult) -> PlanEvaluation:
    tcp_xyz = np.asarray(result.tcp_xyz, dtype=float).reshape(-1, 3)
    ref_xyz = np.asarray(result.reference_xyz, dtype=float).reshape(-1, 3)
    tcp_yaw = np.asarray(result.tcp_yaw_rad, dtype=float).reshape(-1)
    ref_yaw = np.asarray(result.reference_yaw_rad, dtype=float).reshape(-1)
    pos_err = np.linalg.norm(tcp_xyz - ref_xyz, axis=1)
    yaw_err_deg = np.degrees(np.abs([_wrap_to_pi(float(a - b)) for a, b in zip(tcp_yaw, ref_yaw)]))
    path_length_m = float(np.linalg.norm(np.diff(tcp_xyz, axis=0), axis=1).sum()) if tcp_xyz.shape[0] > 1 else 0.0
    max_path_deviation_m = max((_point_to_polyline_distance(point, ref_xyz) for point in tcp_xyz), default=0.0)
    result.evaluation = PlanEvaluation(
        final_position_error_m=float(pos_err[-1]),
        final_yaw_error_deg=float(yaw_err_deg[-1]),
        max_position_error_m=float(np.max(pos_err)),
        mean_position_error_m=float(np.mean(pos_err)),
        max_path_deviation_m=float(max_path_deviation_m),
        path_length_m=path_length_m,
    )
    return result.evaluation


def _build_planning_context(
    scenario: StandaloneScenario,
    *,
    stack_name: str,
) -> _PlanningContext | StandalonePlanResult:
    stage = JointGoalStage()
    names = tuple(map(str, stage.config.actuated_joints))
    planner = JointSpaceCartesianPlanner(
        urdf_path=stage.config.urdf_path,
        target_frame=stage.config.target_frame,
        reduced_joint_names=stage.config.actuated_joints,
    )

    def fail(message: str) -> StandalonePlanResult:
        return _fail_result(stack_name=stack_name, actuated_joint_count=len(names), message=message)

    def complete_seed(q_red: np.ndarray, seed: dict[str, float]) -> dict[str, float]:
        solved = stage._steady_state.complete_from_actuated(
            {name: float(q_red[i]) for i, name in enumerate(names)},
            q_seed=seed,
        )
        return dict(solved.q_dynamic) if solved.success else dict(seed)

    if scenario.planner_start_q is None:
        start = stage.solve_world_pose(goal_world=scenario.start_world_xyz, target_yaw_rad=scenario.start_yaw_rad, q_seed={})
        if not start.success:
            return fail(f"start solve failed: {start.message}")
        q_start = _actuated_vector(start.q_actuated, names)
        start_seed = dict(start.q_dynamic)
    else:
        q_start = np.asarray(scenario.planner_start_q, dtype=float)
        start_seed = dict(scenario.planner_start_q_seed_map or {name: float(q_start[i]) for i, name in enumerate(names)})

    if scenario.planner_goal_q is None:
        goal = stage.solve_world_pose(goal_world=scenario.goal_world_xyz, target_yaw_rad=scenario.goal_yaw_rad, q_seed=start_seed)
        if not goal.success:
            return fail(f"goal solve failed: {goal.message}")
        q_goal = _actuated_vector(goal.q_actuated, names)
        goal_seed = dict(goal.q_dynamic)
    else:
        q_goal = np.asarray(scenario.planner_goal_q, dtype=float)
        goal_seed = dict(scenario.planner_goal_q_seed_map or start_seed)

    return _PlanningContext(
        planner=planner,
        actuated_joint_names=names,
        q_start=q_start,
        q_goal=q_goal,
        q_start_seed_map=start_seed,
        q_goal_seed_map=goal_seed,
    )


def plan_joint_space_global_path(scenario: StandaloneScenario) -> StandalonePlanResult:
    planning = _build_planning_context(scenario, stack_name="joint_space_global_path")
    if isinstance(planning, StandalonePlanResult):
        return planning

    joint_limits, _ = load_planning_limits_yaml(Path(__file__).resolve().parent / "trajectory" / "planning_limits.yaml")
    planning.planner._joint_position_limits = dict(joint_limits)
    cfg = ScenarioLibrary().build_scenario(scenario.overlay_scene_name or "step_01_first_on_ground")
    result = planning.planner.plan_global_path(
        JointSpaceGlobalPathRequest(
            scene=cfg.scene,
            moving_block_size=tuple(float(v) for v in cfg.moving_block_size),
            q_start=planning.q_start,
            q_goal=planning.q_goal,
            q_start_seed_map=planning.q_start_seed_map,
            q_goal_seed_map=planning.q_goal_seed_map,
            start_approach_direction_world=scenario.start_approach_direction_world,
            goal_approach_direction_world=scenario.goal_approach_direction_world,
        )
    )
    if not result.success and result.q_waypoints.size == 0:
        return _fail_result(
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


_Q_HOVER_CENTER = (0.38024054715340927, 0.9565408595950765, -0.040579164841686555, 0.8338932389503255, 0.3806049296978129)
_Q_INIT_OUTSIDE = (0.785, 0.523599, 0.523602, 0.25, 0.0)
_SEED_INIT_OUTSIDE = {
    "theta1_slewing_joint": 0.785,
    "theta2_boom_joint": 0.523599,
    "theta3_arm_joint": 0.523602,
    "q4_big_telescope": 0.25,
    "q5_small_telescope": 0.25,
    "theta6_tip_joint": 0.5236,
    "theta7_tilt_joint": 1.5708,
    "theta8_rotator_joint": 0.0,
    "q9_left_rail_joint": 0.21,
    "q11_right_rail_joint": 0.21,
    "boom_cylinder_piston_in_barrel_linear_joint": 1.847141,
    "arm_cylinder_piston_in_barrel_linear_joint_left": 1.859136,
    "arm_cylinder_piston_in_barrel_linear_joint_right": 1.859136,
    "boom_cylinder_linkage_big_mounting_on_slewing_column": 0.253841,
    "boom_cylinder_linkage_small_mounting_on_boom": -0.355826,
    "boom_cylinder_mounting_on_slewing_column": 0.000685,
}
_XYZ_INIT_OUTSIDE = (4.130, 4.127, 1.010)
_Q_APPROACH_BLOCK_1 = (0.3805, 0.7000, 0.0442, 0.7704, 0.0)
_XYZ_APPROACH_BLOCK_1 = (5.0, 2.0, -0.04)
_Q_HOVER_RIGHT = (0.25, -0.80, 0.55, 0.40, -0.25)
_Q_HOVER_LEFT = (-0.15, -0.80, 0.55, 0.40, 0.15)
_SEED_INIT_HORIZONTAL = _reference_seed("initialization_horizontal")
_SEED_HOVER_CENTER = {
    **_SEED_INIT_HORIZONTAL,
    "theta1_slewing_joint": _Q_HOVER_CENTER[0],
    "theta2_boom_joint": _Q_HOVER_CENTER[1],
    "theta3_arm_joint": _Q_HOVER_CENTER[2],
    "q4_big_telescope": _Q_HOVER_CENTER[3],
    "q5_small_telescope": _Q_HOVER_CENTER[3],
    "theta8_rotator_joint": _Q_HOVER_CENTER[4],
}
_Q_GOAL_CENTER = (0.05, -0.92, 0.68, 0.65, -0.05)
_Q_GOAL_RIGHT = (0.25, -0.92, 0.68, 0.65, -0.25)
_Q_GOAL_LEFT = (-0.15, -0.92, 0.68, 0.65, 0.15)
_Q_GOAL_ON_TOP = (0.05, -0.86, 0.62, 0.65, -0.05)
_XYZ_HOVER_CENTER = (0.841, 0.046, -2.874)
_XYZ_HOVER_RIGHT = (0.814, 0.224, -2.874)
_XYZ_HOVER_LEFT = (0.832, -0.133, -2.874)
_XYZ_GOAL_CENTER = (0.443, 0.026, -3.647)
_XYZ_GOAL_RIGHT = (0.428, 0.125, -3.647)
_XYZ_GOAL_LEFT = (0.438, -0.073, -3.647)
_XYZ_GOAL_ON_TOP = (0.605, 0.034, -3.515)
_CBS_SCENARIO_STARTS = {
    "step_01_first_on_ground": (_Q_HOVER_CENTER, _XYZ_HOVER_CENTER, _phi_tool_from_reduced_q(_Q_HOVER_CENTER), _Q_GOAL_CENTER, _XYZ_GOAL_CENTER),
    "step_02_second_beside_first": (_Q_HOVER_RIGHT, _XYZ_HOVER_RIGHT, _phi_tool_from_reduced_q(_Q_HOVER_RIGHT), _Q_GOAL_RIGHT, _XYZ_GOAL_RIGHT),
    "step_03_third_on_top": (_Q_HOVER_CENTER, _XYZ_HOVER_CENTER, _phi_tool_from_reduced_q(_Q_HOVER_CENTER), _Q_GOAL_ON_TOP, _XYZ_GOAL_ON_TOP),
    "step_04_between_two_blocks": (_Q_HOVER_CENTER, _XYZ_HOVER_CENTER, _phi_tool_from_reduced_q(_Q_HOVER_CENTER), _Q_GOAL_CENTER, _XYZ_GOAL_CENTER),
}
_CURATED_SCENARIOS = (
    StandaloneScenario(
        name="short_reachable_move",
        description="Short validated reachable motion in free space for standalone commissioning.",
        start_world_xyz=_XYZ_HOVER_CENTER,
        goal_world_xyz=(0.781, 0.146, -2.874),
        start_yaw_rad=_phi_tool_from_reduced_q(_Q_HOVER_CENTER),
        goal_yaw_rad=_phi_tool_from_reduced_q((0.17, -0.80, 0.55, 0.40, -0.17)),
        start_approach_direction_world=_unit((0.781 - _XYZ_HOVER_CENTER[0], 0.146 - _XYZ_HOVER_CENTER[1], -2.874 - _XYZ_HOVER_CENTER[2])),
        planner_start_q=_Q_HOVER_CENTER,
        planner_start_q_seed_map=_SEED_HOVER_CENTER,
        planner_goal_q=(0.17, -0.80, 0.55, 0.40, -0.17),
    ),
    StandaloneScenario(
        name="yaw_change_probe",
        description="Reachable probe that mainly exercises yaw change handling.",
        start_world_xyz=_XYZ_HOVER_CENTER,
        goal_world_xyz=_XYZ_HOVER_LEFT,
        start_yaw_rad=_phi_tool_from_reduced_q(_Q_HOVER_CENTER),
        goal_yaw_rad=_phi_tool_from_reduced_q(_Q_HOVER_LEFT),
        start_approach_direction_world=_unit((_XYZ_HOVER_LEFT[0] - _XYZ_HOVER_CENTER[0], _XYZ_HOVER_LEFT[1] - _XYZ_HOVER_CENTER[1], _XYZ_HOVER_LEFT[2] - _XYZ_HOVER_CENTER[2])),
        planner_start_q=_Q_HOVER_CENTER,
        planner_start_q_seed_map=_SEED_HOVER_CENTER,
        planner_goal_q=_Q_HOVER_LEFT,
    ),
    StandaloneScenario(
        name="single_block_transfer",
        description="Scene-backed block transfer target used to expose current solve limitations.",
        start_world_xyz=_XYZ_HOVER_CENTER,
        goal_world_xyz=_XYZ_GOAL_CENTER,
        start_yaw_rad=_phi_tool_from_reduced_q(_Q_HOVER_CENTER),
        goal_yaw_rad=_phi_tool_from_reduced_q(_Q_GOAL_CENTER),
        start_approach_direction_world=_unit((_XYZ_GOAL_CENTER[0] - _XYZ_HOVER_CENTER[0], _XYZ_GOAL_CENTER[1] - _XYZ_HOVER_CENTER[1], _XYZ_GOAL_CENTER[2] - _XYZ_HOVER_CENTER[2])),
        planner_start_q=_Q_HOVER_CENTER,
        planner_start_q_seed_map=_SEED_HOVER_CENTER,
        planner_goal_q=_Q_GOAL_CENTER,
        overlay_scene_name="step_01_first_on_ground",
    ),
    StandaloneScenario(
        name="init_outside_to_block1_approach",
        description="PZS100 initialization_outside to pick approach above block_1 at [5,2,-0.84].",
        start_world_xyz=_XYZ_INIT_OUTSIDE,
        goal_world_xyz=_XYZ_APPROACH_BLOCK_1,
        start_yaw_rad=_phi_tool_from_reduced_q(_Q_INIT_OUTSIDE),
        goal_yaw_rad=float(math.atan2(2.0, 5.0)),
        start_approach_direction_world=_unit((_XYZ_APPROACH_BLOCK_1[0] - _XYZ_INIT_OUTSIDE[0], _XYZ_APPROACH_BLOCK_1[1] - _XYZ_INIT_OUTSIDE[1], _XYZ_APPROACH_BLOCK_1[2] - _XYZ_INIT_OUTSIDE[2])),
        planner_start_q=_Q_INIT_OUTSIDE,
        planner_start_q_seed_map=_SEED_INIT_OUTSIDE,
        planner_goal_q=_Q_APPROACH_BLOCK_1,
    ),
    StandaloneScenario(
        name="scene_demo_step_01_reachable",
        description="Reachable demo with the step_01 scene overlay and validated seeds.",
        start_world_xyz=_XYZ_HOVER_CENTER,
        goal_world_xyz=(0.781, 0.146, -2.874),
        start_yaw_rad=_phi_tool_from_reduced_q(_Q_HOVER_CENTER),
        goal_yaw_rad=_phi_tool_from_reduced_q((0.17, -0.80, 0.55, 0.40, -0.17)),
        start_approach_direction_world=_unit((0.781 - _XYZ_HOVER_CENTER[0], 0.146 - _XYZ_HOVER_CENTER[1], -2.874 - _XYZ_HOVER_CENTER[2])),
        planner_start_q=_Q_HOVER_CENTER,
        planner_start_q_seed_map=_SEED_HOVER_CENTER,
        planner_goal_q=(0.17, -0.80, 0.55, 0.40, -0.17),
        overlay_scene_name="step_01_first_on_ground",
    ),
)


def make_default_scenarios() -> dict[str, StandaloneScenario]:
    scenarios = {sc.name: sc for sc in _CURATED_SCENARIOS}
    scene_lib = ScenarioLibrary()
    for scene_name in scene_lib.list_scenarios():
        cfg = scene_lib.build_scenario(scene_name)
        if scene_name in _CBS_SCENARIO_STARTS:
            q_start, start_xyz, start_yaw, q_goal, goal_xyz = _CBS_SCENARIO_STARTS[scene_name]
            goal_yaw = math.radians(float(cfg.goal_yaw_deg))
        else:
            q_start = q_goal = None
            start_xyz = tuple(float(v) for v in cfg.start)
            goal_xyz = tuple(float(v) for v in cfg.goal)
            start_yaw = math.radians(float(cfg.start_yaw_deg))
            goal_yaw = math.radians(float(cfg.goal_yaw_deg))
        scenarios[f"scene_{scene_name}"] = StandaloneScenario(
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
            planner_start_q_seed_map=_SEED_HOVER_CENTER if q_start == _Q_HOVER_CENTER else None,
            overlay_scene_name=scene_name,
        )
    return scenarios


STACK_REGISTRY = {"joint_space_global_path": plan_joint_space_global_path}

DEFAULT_PLANNER_CFG = {
    "goal_approach_window_fraction": 0.1,
    "contact_window_fraction": 0.1,
}
FALLBACK_DIAGNOSTICS = {
    "reference_path_fallback_used": 1.0,
    "joint_anchor_fallback_used": 0.0,
}


def make_straight_curve_sampler(start_xyz: np.ndarray, goal_xyz: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    start = np.asarray(start_xyz, dtype=float).reshape(3)
    goal = np.asarray(goal_xyz, dtype=float).reshape(3)

    def sample(uq: np.ndarray) -> np.ndarray:
        u = np.asarray(uq, dtype=float).reshape(-1, 1)
        return (1.0 - u) * start.reshape(1, 3) + u * goal.reshape(1, 3)

    return sample


def make_linear_yaw_fn(start_yaw_deg: float, goal_yaw_deg: float) -> Callable[[np.ndarray], np.ndarray]:
    def yaw(uq: np.ndarray) -> np.ndarray:
        u = np.asarray(uq, dtype=float).reshape(-1)
        return np.asarray(start_yaw_deg + (goal_yaw_deg - start_yaw_deg) * u, dtype=float)

    return yaw


def is_cbs_stack(method: str) -> bool:
    return method.lower().replace("-", "_") == "joint_space_global_path"


def _fk_curve_from_joint_spline(
    planning: _PlanningContext,
    q_control_points: np.ndarray,
    u_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u_query, dtype=float).reshape(-1)
    q = planning.planner._clip_to_joint_limits(JointSpaceGlobalPathPlanner._sample_joint_bspline(q_control_points, u))
    q[0, :] = np.asarray(q_control_points[0], dtype=float)
    q[-1, :] = np.asarray(q_control_points[-1], dtype=float)
    seed = dict(planning.q_start_seed_map)
    xyz, yaw = [], []
    for qi in q:
        xi, yi, seed = planning.planner.fk_world_pose(qi, q_seed=seed)
        xyz.append(xi)
        yaw.append(yi)
    return np.asarray(xyz, dtype=float), np.asarray(yaw, dtype=float)


def plan_cbs_stack(method: str, demo_scenario_name: str) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, dict[str, Any]]:
    if not is_cbs_stack(method):
        raise ValueError("Only 'joint_space_global_path' is supported in the standalone demo")
    candidates = [sc for sc in make_default_scenarios().values() if sc.overlay_scene_name == demo_scenario_name]
    if not candidates:
        raise ValueError(f"No standalone scenario found for '{demo_scenario_name}'")
    candidates.sort(key=lambda sc: (not sc.name.startswith("scene_demo_"), sc.name.startswith("scene_"), sc.name))
    last_result = None
    for scenario in candidates:
        result = STACK_REGISTRY["joint_space_global_path"](scenario)
        if result.success:
            break
        last_result = result
    else:
        raise RuntimeError(last_result.message if last_result is not None else f"Planning failed for '{demo_scenario_name}'")

    planning = _build_planning_context(scenario, stack_name="joint_space_global_path")
    if isinstance(planning, StandalonePlanResult):
        raise RuntimeError(planning.message)
    q_control_points = np.asarray(result.diagnostics.get("q_control_points", np.empty((0, 0))), dtype=float)
    if q_control_points.shape != (4, len(planning.actuated_joint_names)):
        raise RuntimeError("joint-space control points missing from planner diagnostics")

    def curve_sampler(uq: np.ndarray) -> np.ndarray:
        xyz, _ = _fk_curve_from_joint_spline(planning, q_control_points, uq)
        return xyz.reshape(-1, 3)

    def yaw_fn(uq: np.ndarray) -> np.ndarray:
        _, yaw = _fk_curve_from_joint_spline(planning, q_control_points, uq)
        return np.degrees(yaw)

    return curve_sampler, np.asarray(result.diagnostics.get("via_tcp_xyz", np.empty((0, 3))), dtype=float), {
        "yaw_fn": yaw_fn,
        "success": bool(result.success),
        "message": str(result.message),
        "nit": int(result.diagnostics.get("optimizer_iterations", 0)),
        "preferred_clearance": 0.05,
        "diagnostics": {
            **dict(result.diagnostics),
            "tcp_xyz_path": np.asarray(result.tcp_xyz, dtype=float),
            "tcp_yaw_path_rad": np.asarray(result.tcp_yaw_rad, dtype=float),
            "q_maps_path": list(result.diagnostics.get("q_maps_path", [])),
        },
        "joint_anchor_fallback_used": float(result.diagnostics.get("joint_anchor_fallback_used", 0.0)),
        "reference_path_fallback_used": float(result.diagnostics.get("reference_path_fallback_used", 0.0)),
    }
