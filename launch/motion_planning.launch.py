from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = PathJoinSubstitution(
        [FindPackageShare("concrete_block_motion_planning"), "config", "motion_planning.yaml"]
    )
    named_cfg_file = PathJoinSubstitution(
        [FindPackageShare("concrete_block_motion_planning"), "config", "named_configurations.yaml"]
    )
    wall_plan_file = PathJoinSubstitution(
        [FindPackageShare("concrete_block_motion_planning"), "motion_planning", "data", "wall_plans.yaml"]
    )
    optimized_params_file = PathJoinSubstitution(
        [FindPackageShare("concrete_block_motion_planning"), "motion_planning", "data", "optimized_params.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("params_file", default_value=config_file),
            DeclareLaunchArgument("named_configurations_file", default_value=named_cfg_file),
            DeclareLaunchArgument("wall_plan_file", default_value=wall_plan_file),
            DeclareLaunchArgument(
                "geometric_optimized_params_file",
                default_value=optimized_params_file,
            ),
            Node(
                package="concrete_block_motion_planning",
                executable="motion_planning_node.py",
                name="concrete_block_motion_planning_node",
                output="screen",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {"named_configurations_file": LaunchConfiguration("named_configurations_file")},
                    {"wall_plan_file": LaunchConfiguration("wall_plan_file")},
                    {
                        "geometric_optimized_params_file": LaunchConfiguration(
                            "geometric_optimized_params_file"
                        )
                    },
                    {"use_sim_time": LaunchConfiguration("use_sim_time")},
                ],
            ),
        ]
    )
