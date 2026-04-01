"""Verification tests for CBS motion planning with PZS100 initialization_outside config.

Tests FK, IK/steady-state, and joint-space path planning from the
``initialization_outside`` crane pose to an approach position above a concrete
block — matching the "Gazebo model with behavior tree (PZS100)" launch scenario.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

from motion_planning.geometry.scene import Scene
from motion_planning.kinematics.crane import CraneKinematics
from motion_planning.mechanics.analytic import (
    AnalyticModelConfig,
    CraneSteadyState,
    ModelDescription,
)
from motion_planning.mechanics.analytic.pinocchio_utils import q_map_to_pin_q
from motion_planning.pipeline import (
    JointSpaceCartesianPlanner,
    JointSpaceGlobalPathRequest,
)
from motion_planning.trajectory.planning_limits import load_planning_limits_yaml

# ---------------------------------------------------------------------------
# Constants from initialization_outside.yaml
# ---------------------------------------------------------------------------

Q_INIT_OUTSIDE_ACTUATED = {
    "theta1_slewing_joint": 0.785,
    "theta2_boom_joint": 0.523599,
    "theta3_arm_joint": 0.523602,
    "q4_big_telescope": 0.25,
    "theta8_rotator_joint": 0.0,
}

Q_INIT_OUTSIDE_SEED = {
    **Q_INIT_OUTSIDE_ACTUATED,
    "q5_small_telescope": 0.25,
    "theta6_tip_joint": 0.546470,
    "theta7_tilt_joint": 1.570521,
    "q9_left_rail_joint": 0.21,
    "q11_right_rail_joint": 0.21,
    "boom_cylinder_piston_in_barrel_linear_joint": 1.847141,
    "boom_cylinder_mounting_on_slewing_column": 0.000685,
    "boom_cylinder_linkage_big_mounting_on_slewing_column": 0.253841,
    "boom_cylinder_linkage_small_mounting_on_boom": -0.355826,
    "arm_cylinder_piston_in_barrel_linear_joint_right": 1.859136,
    "arm_cylinder_piston_in_barrel_linear_joint_left": 1.859136,
}

Q_INIT_OUTSIDE_REDUCED = (0.785, 0.523599, 0.523602, 0.25, 0.0)

# Block positions in K0_mounting_base frame (from world_model_seed_pick_place.yaml)
BLOCK_1_POS = np.array([5.0, 2.0, -0.84])
BLOCK_2_POS = np.array([5.0, -2.0, -0.84])
BLOCK_SIZE = (0.6, 0.9, 0.6)

# Approach: 0.5 m above block_1 top surface (-0.84 + 0.3 + 0.5 = -0.04)
APPROACH_BLOCK_1 = np.array([5.0, 2.0, -0.04])
APPROACH_YAW = float(np.arctan2(2.0, 5.0))  # ~0.38 rad


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg() -> AnalyticModelConfig:
    return AnalyticModelConfig.default()


@pytest.fixture(scope="module")
def desc(cfg: AnalyticModelConfig) -> ModelDescription:
    return ModelDescription(cfg)


@pytest.fixture(scope="module")
def ss(cfg: AnalyticModelConfig, desc: ModelDescription) -> CraneSteadyState:
    return CraneSteadyState(desc, cfg)


@pytest.fixture(scope="module")
def pin_kin(cfg: AnalyticModelConfig) -> CraneKinematics:
    return CraneKinematics(cfg.urdf_path)


# ---------------------------------------------------------------------------
# Test 1: FK from initialization_outside
# ---------------------------------------------------------------------------

def test_fk_initialization_outside(pin_kin: CraneKinematics, cfg: AnalyticModelConfig):
    """FK from initialization_outside joint values produces a physically reasonable TCP."""
    q_pin = q_map_to_pin_q(pin_kin.model, Q_INIT_OUTSIDE_SEED, pin)
    fk = pin_kin.forward_kinematics(
        q_pin, base_frame=cfg.base_frame, end_frame=cfg.target_frame
    )
    tcp = fk["base_to_end"]["translation"]
    radial = float(np.sqrt(tcp[0] ** 2 + tcp[1] ** 2))

    print(f"FK TCP position (K0): [{tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}]")
    print(f"Radial distance: {radial:.3f} m")

    # Crane is extended outward — radial distance should be 3–8 m
    assert 3.0 < radial < 8.0, f"Radial distance {radial:.2f} out of expected range"
    # Boom is elevated — z should be above mounting base (> -2 m in K0 frame)
    assert tcp[2] > -4.0, f"TCP z={tcp[2]:.2f} unexpectedly low"


# ---------------------------------------------------------------------------
# Test 2: Steady-state solve at approach position above block_1
# ---------------------------------------------------------------------------

def test_steady_state_approach_block1(ss: CraneSteadyState):
    """Steady-state IK + passive equilibrium at the approach pose above block_1."""
    result = ss.compute(
        target_pos=APPROACH_BLOCK_1,
        target_yaw=APPROACH_YAW,
        q_seed=Q_INIT_OUTSIDE_SEED,
    )

    print(f"Steady-state success: {result.success}")
    print(f"Message: {result.message}")
    print(f"FK position error: {result.fk_position_error_m:.6f} m")
    print(f"FK yaw error: {np.degrees(result.fk_yaw_error_rad):.4f} deg")
    print(f"Passive residual: {result.passive_residual:.2e}")
    print(f"Actuated: {result.q_actuated}")
    print(f"Passive: {result.q_passive}")

    assert result.success, result.message
    assert result.passive_residual < 1e-5
    assert result.fk_position_error_m < 0.02
    assert abs(result.fk_yaw_error_rad) < 0.1


# ---------------------------------------------------------------------------
# Test 3: Complete passive equilibrium from actuated-only input
# ---------------------------------------------------------------------------

def test_complete_from_actuated_init_outside(ss: CraneSteadyState):
    """Passive equilibrium from initialization_outside actuated joints."""
    result = ss.complete_from_actuated(
        Q_INIT_OUTSIDE_ACTUATED, q_seed=Q_INIT_OUTSIDE_SEED
    )

    print(f"Complete success: {result.success}")
    print(f"Passive residual: {result.passive_residual:.2e}")
    print(f"Passive: {result.q_passive}")
    print(f"FK TCP: {result.fk_xyz}")

    assert result.success, result.message
    assert result.passive_residual < 1e-5


# ---------------------------------------------------------------------------
# Test 4: Joint-space path planning init_outside → approach block_1
# ---------------------------------------------------------------------------

def test_joint_space_path_init_outside_to_approach_block1(
    ss: CraneSteadyState,
    cfg: AnalyticModelConfig,
):
    """Plan a joint-space path from initialization_outside to approach above block_1."""
    # Solve start steady state
    start_ss = ss.complete_from_actuated(
        Q_INIT_OUTSIDE_ACTUATED, q_seed=Q_INIT_OUTSIDE_SEED
    )
    assert start_ss.success, f"Start steady-state failed: {start_ss.message}"

    # Solve goal steady state
    goal_ss = ss.compute(
        target_pos=APPROACH_BLOCK_1,
        target_yaw=APPROACH_YAW,
        q_seed=Q_INIT_OUTSIDE_SEED,
    )
    assert goal_ss.success, f"Goal steady-state failed: {goal_ss.message}"

    # Extract reduced q (5-element) in actuated joint order
    act_names = cfg.actuated_joints
    q_start = np.array([start_ss.q_actuated[jn] for jn in act_names], dtype=float)
    q_goal = np.array([goal_ss.q_actuated[jn] for jn in act_names], dtype=float)

    print(f"q_start (actuated): {q_start}")
    print(f"q_goal  (actuated): {q_goal}")
    print(f"Start TCP: {start_ss.fk_xyz}")
    print(f"Goal  TCP: {goal_ss.fk_xyz}")

    # Build scene with two blocks
    scene = Scene()
    scene.add_block(size=BLOCK_SIZE, position=tuple(BLOCK_1_POS), object_id="block_1")
    scene.add_block(size=BLOCK_SIZE, position=tuple(BLOCK_2_POS), object_id="block_2")

    # Build planner
    repo_root = Path(__file__).resolve().parents[2]
    planning_limits_path = repo_root / "motion_planning" / "trajectory" / "planning_limits.yaml"
    joint_limits, _ = load_planning_limits_yaml(planning_limits_path)
    planner = JointSpaceCartesianPlanner(
        urdf_path=cfg.urdf_path,
        target_frame=cfg.target_frame,
        reduced_joint_names=cfg.actuated_joints,
        joint_position_limits=joint_limits,
    )

    # Compute approach direction (start → goal in world frame)
    start_xyz = np.asarray(start_ss.fk_xyz, dtype=float)
    goal_xyz = np.asarray(goal_ss.fk_xyz, dtype=float)
    direction = goal_xyz - start_xyz
    norm = float(np.linalg.norm(direction))
    start_approach = tuple(float(v) for v in (direction / max(norm, 1e-9)))

    result = planner.plan_global_path(
        JointSpaceGlobalPathRequest(
            scene=scene,
            moving_block_size=BLOCK_SIZE,
            q_start=q_start,
            q_goal=q_goal,
            start_approach_direction_world=start_approach,
            goal_approach_direction_world=(0.0, 0.0, -1.0),
        )
    )

    print(f"Planning success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Waypoints shape: {result.q_waypoints.shape}")
    print(f"Via points shape: {result.via_points.shape}")
    if "min_signed_distance_m" in result.diagnostics:
        print(f"Min clearance: {result.diagnostics['min_signed_distance_m']:.4f} m")

    assert result.success, result.message
    assert result.q_waypoints.shape[1] == 5
    assert result.via_points.shape[0] == 2
