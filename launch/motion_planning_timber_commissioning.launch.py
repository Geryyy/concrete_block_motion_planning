from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    commissioning_config = PathJoinSubstitution(
        [
            FindPackageShare("concrete_block_motion_planning"),
            "config",
            "motion_planning_timber_commissioning.yaml",
        ]
    )
    motion_planning_launch = PathJoinSubstitution(
        [
            FindPackageShare("concrete_block_motion_planning"),
            "launch",
            "motion_planning.launch.py",
        ]
    )
    named_cfg_file = PathJoinSubstitution(
        [
            FindPackageShare("concrete_block_motion_planning"),
            "config",
            "named_configurations.yaml",
        ]
    )
    wall_plan_file = PathJoinSubstitution(
        [
            FindPackageShare("concrete_block_motion_planning"),
            "motion_planning",
            "data",
            "wall_plans.yaml",
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument(
                "motion_planning_params_file", default_value=commissioning_config
            ),
            DeclareLaunchArgument(
                "named_configurations_file", default_value=named_cfg_file
            ),
            DeclareLaunchArgument("wall_plan_file", default_value=wall_plan_file),
            DeclareLaunchArgument(
                "planner_timber_a2b_service", default_value="a2b_movement"
            ),
            DeclareLaunchArgument(
                "planner_timber_goal_frame", default_value="K0_mounting_base"
            ),
            DeclareLaunchArgument(
                "planner_timber_move_empty_target_z", default_value="2.36"
            ),
            DeclareLaunchArgument(
                "execution_action_name",
                default_value="/trajectory_controller_a2b/follow_joint_trajectory",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(motion_planning_launch),
                launch_arguments={
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "motion_planning_params_file": LaunchConfiguration(
                        "motion_planning_params_file"
                    ),
                    "planner_backend": "timber",
                    "default_trajectory_method": "TIMBER_MOVE_EMPTY",
                    "named_configurations_file": LaunchConfiguration(
                        "named_configurations_file"
                    ),
                    "wall_plan_file": LaunchConfiguration("wall_plan_file"),
                    "execution_enabled": "true",
                    "execution_backend": "action",
                    "execution_action_name": LaunchConfiguration(
                        "execution_action_name"
                    ),
                    "execution_switch_controller": "false",
                    "execution_activate_controller": "trajectory_controllers",
                    "planner_timber_a2b_service": LaunchConfiguration(
                        "planner_timber_a2b_service"
                    ),
                    "planner_timber_goal_frame": LaunchConfiguration(
                        "planner_timber_goal_frame"
                    ),
                    "planner_timber_move_empty_target_z": LaunchConfiguration(
                        "planner_timber_move_empty_target_z"
                    ),
                }.items(),
            ),
        ]
    )
