from __future__ import annotations

from motion_planning.contracts import JointMapping, RobotProfile


CONCRETE_PZS100_PROFILE = RobotProfile(
    name="concrete_pzs100",
    mapping=JointMapping(
        full_state_joint_names=(
            "theta1_slewing_joint",
            "theta2_boom_joint",
            "theta3_arm_joint",
            "q4_big_telescope",
            "theta6_tip_joint",
            "theta7_tilt_joint",
            "theta8_rotator_joint",
            "q9_left_rail_joint",
        ),
        command_joint_names=(
            "theta1_slewing_joint",
            "theta2_boom_joint",
            "theta3_arm_joint",
            "q4_big_telescope",
            "theta8_rotator_joint",
            "q9_left_rail_joint",
        ),
        passive_joint_names=("theta6_tip_joint", "theta7_tilt_joint"),
        mimic_joint_map={"q11_right_rail_joint": "q9_left_rail_joint"},
    ),
    tool_joint_names=("q9_left_rail_joint", "q11_right_rail_joint"),
    compat_aliases={"theta10_outer_jaw_joint": "q9_left_rail_joint"},
    notes=(
        "Concrete PZS100 mirrors the timber structural split: an 8-slot full-state "
        "trajectory contract with a 6-joint commanded subset. The right rail is "
        "mimicked from q9_left_rail_joint and is not independently commanded."
    ),
)
