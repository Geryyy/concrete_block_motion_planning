from __future__ import annotations

from motion_planning.adapters import (
    densify_trajectory_for_streaming,
    JointSampleMaps,
    expand_point_to_profile,
    infer_robot_profile,
    profile_command_joint_names,
    project_positions_to_command_joints,
)
from motion_planning.profiles import CONCRETE_PZS100_PROFILE, TIMBER_COMPAT_PROFILE
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def test_infer_robot_profile_from_full_state_joint_names() -> None:
    assert infer_robot_profile(TIMBER_COMPAT_PROFILE.full_state_joint_names) == TIMBER_COMPAT_PROFILE
    assert infer_robot_profile(CONCRETE_PZS100_PROFILE.full_state_joint_names) == CONCRETE_PZS100_PROFILE


def test_infer_robot_profile_from_command_joint_names() -> None:
    assert infer_robot_profile(TIMBER_COMPAT_PROFILE.command_joint_names) == TIMBER_COMPAT_PROFILE
    assert infer_robot_profile(CONCRETE_PZS100_PROFILE.command_joint_names) == CONCRETE_PZS100_PROFILE


def test_profile_command_joint_names_projects_to_actuated_subset() -> None:
    assert profile_command_joint_names(CONCRETE_PZS100_PROFILE.full_state_joint_names) == (
        "theta1_slewing_joint",
        "theta2_boom_joint",
        "theta3_arm_joint",
        "q4_big_telescope",
        "theta8_rotator_joint",
        "q9_left_rail_joint",
    )


def test_project_positions_to_command_joints_drops_passive_entries() -> None:
    positions = {
        "theta1_slewing_joint": 0.1,
        "theta2_boom_joint": -1.0,
        "theta3_arm_joint": -0.4,
        "q4_big_telescope": 2.2,
        "theta6_tip_joint": 0.3,
        "theta7_tilt_joint": 1.4,
        "theta8_rotator_joint": -0.2,
        "q9_left_rail_joint": 0.05,
    }
    projected = project_positions_to_command_joints(
        CONCRETE_PZS100_PROFILE.full_state_joint_names,
        positions,
    )
    assert set(projected.keys()) == set(CONCRETE_PZS100_PROFILE.command_joint_names)
    assert "theta6_tip_joint" not in projected
    assert "theta7_tilt_joint" not in projected


def test_expand_point_to_profile_maps_timber_gripper_to_pzs100() -> None:
    positions, velocities, accelerations = expand_point_to_profile(
        source_joint_names=TIMBER_COMPAT_PROFILE.full_state_joint_names,
        sample_maps=JointSampleMaps(
            positions={
                "theta1_slewing_joint": 0.1,
                "theta2_boom_joint": -1.2,
                "theta3_arm_joint": -0.5,
                "q4_big_telescope": 2.1,
                "theta6_tip_joint": 0.2,
                "theta7_tilt_joint": 1.5,
                "theta8_rotator_joint": -0.3,
                "theta10_outer_jaw_joint": 0.04,
            },
            velocities={"theta10_outer_jaw_joint": 0.01, "theta8_rotator_joint": 0.02},
            accelerations={"theta10_outer_jaw_joint": 0.03},
        ),
        latest_positions={"theta6_tip_joint": 0.2, "theta7_tilt_joint": 1.5},
        target_profile=CONCRETE_PZS100_PROFILE,
    )
    assert positions[-1] == 0.04
    assert velocities[-1] == 0.01
    assert accelerations[-1] == 0.03


def test_expand_point_to_profile_preserves_passive_state_defaults() -> None:
    positions, _, _ = expand_point_to_profile(
        source_joint_names=CONCRETE_PZS100_PROFILE.command_joint_names,
        sample_maps=JointSampleMaps(
            positions={"theta1_slewing_joint": 0.1, "q9_left_rail_joint": 0.03},
            velocities={},
            accelerations={},
        ),
        latest_positions={"theta6_tip_joint": 0.25, "theta7_tilt_joint": 1.45},
        target_profile=CONCRETE_PZS100_PROFILE,
    )
    assert positions[4] == 0.25
    assert positions[5] == 1.45


def test_densify_trajectory_for_streaming_adds_intermediate_samples() -> None:
    trajectory = JointTrajectory()
    trajectory.joint_names = list(CONCRETE_PZS100_PROFILE.full_state_joint_names)

    start = JointTrajectoryPoint()
    start.positions = [0.0] * len(trajectory.joint_names)
    start.velocities = [0.0] * len(trajectory.joint_names)
    start.time_from_start.sec = 0

    end = JointTrajectoryPoint()
    end.positions = [0.2] + [0.0] * (len(trajectory.joint_names) - 1)
    end.velocities = [0.0] * len(trajectory.joint_names)
    end.time_from_start.sec = 4

    trajectory.points = [start, end]
    dense = densify_trajectory_for_streaming(trajectory, sample_period_s=0.5)

    assert len(dense.points) > len(trajectory.points)
    assert dense.points[0].time_from_start.sec == 0
    assert dense.points[-1].time_from_start.sec == 4
    assert dense.points[1].velocities[0] == 0.05
    assert dense.points[-1].velocities[0] == 0.0
