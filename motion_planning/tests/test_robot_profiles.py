#!/usr/bin/env python3

from __future__ import annotations

from types import SimpleNamespace

import pytest

from motion_planning.robot_profile import JointMapping, RobotProfile, TrajectoryContract
from motion_planning.profiles import CONCRETE_PZS100_PROFILE, TIMBER_COMPAT_PROFILE


def test_timber_profile_matches_reference_structure() -> None:
    assert TIMBER_COMPAT_PROFILE.full_state_joint_names == (
        "theta1_slewing_joint",
        "theta2_boom_joint",
        "theta3_arm_joint",
        "q4_big_telescope",
        "theta6_tip_joint",
        "theta7_tilt_joint",
        "theta8_rotator_joint",
        "theta10_outer_jaw_joint",
    )
    assert TIMBER_COMPAT_PROFILE.command_joint_names == (
        "theta1_slewing_joint",
        "theta2_boom_joint",
        "theta3_arm_joint",
        "q4_big_telescope",
        "theta8_rotator_joint",
        "theta10_outer_jaw_joint",
    )
    assert TIMBER_COMPAT_PROFILE.passive_joint_names == (
        "theta6_tip_joint",
        "theta7_tilt_joint",
    )


def test_concrete_profile_encodes_pzs100_specifics() -> None:
    assert CONCRETE_PZS100_PROFILE.full_state_joint_names[-1] == "q9_left_rail_joint"
    assert CONCRETE_PZS100_PROFILE.command_joint_names[-1] == "q9_left_rail_joint"
    assert CONCRETE_PZS100_PROFILE.mimic_joint_map == {
        "q11_right_rail_joint": "q9_left_rail_joint"
    }
    assert CONCRETE_PZS100_PROFILE.compat_aliases["theta10_outer_jaw_joint"] == "q9_left_rail_joint"


def test_joint_mapping_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        JointMapping(
            full_state_joint_names=("a", "b"),
            command_joint_names=("a",),
            passive_joint_names=("a",),
        )


def test_trajectory_contract_validates_joint_order() -> None:
    trajectory = SimpleNamespace(joint_names=list(TIMBER_COMPAT_PROFILE.full_state_joint_names))
    contract = TrajectoryContract(profile=TIMBER_COMPAT_PROFILE, trajectory=trajectory)
    assert contract.command_joint_names == TIMBER_COMPAT_PROFILE.command_joint_names


def test_trajectory_contract_rejects_wrong_joint_order() -> None:
    trajectory = SimpleNamespace(joint_names=list(CONCRETE_PZS100_PROFILE.command_joint_names))
    with pytest.raises(ValueError, match="trajectory joint_names do not match"):
        TrajectoryContract(profile=CONCRETE_PZS100_PROFILE, trajectory=trajectory)


def test_robot_profile_exposes_mapping_indices() -> None:
    profile = RobotProfile(
        name="test",
        mapping=JointMapping(
            full_state_joint_names=("a", "b", "c", "d"),
            command_joint_names=("a", "d"),
            passive_joint_names=("b",),
        ),
    )
    assert profile.mapping.command_indices == (0, 3)
    assert profile.mapping.passive_indices == (1,)
