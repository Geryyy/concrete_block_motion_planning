from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    compatibility_config = PathJoinSubstitution(
        [
            FindPackageShare("concrete_block_motion_planning"),
            "config",
            "motion_planning_concrete_compatibility.yaml",
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
                "motion_planning_params_file", default_value=compatibility_config
            ),
            DeclareLaunchArgument(
                "named_configurations_file", default_value=named_cfg_file
            ),
            DeclareLaunchArgument("wall_plan_file", default_value=wall_plan_file),
            DeclareLaunchArgument(
                "compatibility_a2b_service_name", default_value="a2b_movement"
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(motion_planning_launch),
                launch_arguments={
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "motion_planning_params_file": LaunchConfiguration(
                        "motion_planning_params_file"
                    ),
                    "planner_backend": "concrete",
                    "default_trajectory_method": "TOPPRA_PATH_FOLLOWING",
                    "named_configurations_file": LaunchConfiguration(
                        "named_configurations_file"
                    ),
                    "wall_plan_file": LaunchConfiguration("wall_plan_file"),
                    "execution_enabled": "false",
                    "compatibility_a2b_service_enabled": "true",
                    "compatibility_a2b_service_name": LaunchConfiguration(
                        "compatibility_a2b_service_name"
                    ),
                }.items(),
            ),
        ]
    )
