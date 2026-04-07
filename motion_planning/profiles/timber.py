from __future__ import annotations

from motion_planning.robot_profile import JointMapping, RobotProfile


TIMBER_COMPAT_PROFILE = RobotProfile(
    name="timber_compat",
    mapping=JointMapping(
        full_state_joint_names=(
            "theta1_slewing_joint",
            "theta2_boom_joint",
            "theta3_arm_joint",
            "q4_big_telescope",
            "theta6_tip_joint",
            "theta7_tilt_joint",
            "theta8_rotator_joint",
            "theta10_outer_jaw_joint",
        ),
        command_joint_names=(
            "theta1_slewing_joint",
            "theta2_boom_joint",
            "theta3_arm_joint",
            "q4_big_telescope",
            "theta8_rotator_joint",
            "theta10_outer_jaw_joint",
        ),
        passive_joint_names=("theta6_tip_joint", "theta7_tilt_joint"),
    ),
    tool_joint_names=("theta10_outer_jaw_joint",),
    notes=(
        "Reference timber structure: 8-slot full-state trajectory contract with "
        "a 6-joint commanded subset. Tip/tilt remain state-only."
    ),
)
