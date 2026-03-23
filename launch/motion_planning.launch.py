from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = PathJoinSubstitution(
        [
            FindPackageShare("concrete_block_motion_planning"),
            "config",
            "motion_planning.yaml",
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
    optimized_params_file = PathJoinSubstitution(
        [
            FindPackageShare("concrete_block_motion_planning"),
            "motion_planning",
            "data",
            "optimized_params.yaml",
        ]
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable(
                name="LD_LIBRARY_PATH",
                value=[
                    "/opt/acados/build/acados:/opt/acados/lib:",
                    EnvironmentVariable("LD_LIBRARY_PATH", default_value=""),
                ],
            ),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument(
                "motion_planning_params_file", default_value=config_file
            ),
            DeclareLaunchArgument(
                "default_trajectory_method", default_value="ACADOS_PATH_FOLLOWING"
            ),
            DeclareLaunchArgument(
                "named_configurations_file", default_value=named_cfg_file
            ),
            DeclareLaunchArgument("wall_plan_file", default_value=wall_plan_file),
            DeclareLaunchArgument(
                "geometric_optimized_params_file",
                default_value=optimized_params_file,
            ),
            DeclareLaunchArgument("execution_enabled", default_value="false"),
            DeclareLaunchArgument("execution_backend", default_value="topic"),
            DeclareLaunchArgument(
                "execution_trajectory_topic",
                default_value="/trajectory_controllers/joint_trajectory",
            ),
            DeclareLaunchArgument(
                "execution_action_name",
                default_value="/trajectory_controller_a2b/follow_joint_trajectory",
            ),
            DeclareLaunchArgument("execution_result_timeout_s", default_value="120.0"),
            DeclareLaunchArgument("execution_switch_controller", default_value="false"),
            DeclareLaunchArgument(
                "execution_switch_service",
                default_value="/controller_manager/switch_controller",
            ),
            DeclareLaunchArgument(
                "execution_activate_controller",
                default_value="trajectory_controllers",
            ),
            DeclareLaunchArgument(
                "execution_deactivate_after_execution", default_value="true"
            ),
            DeclareLaunchArgument(
                "world_model_get_coarse_blocks_service",
                default_value="/world_model_node/get_coarse_blocks",
            ),
            Node(
                package="concrete_block_motion_planning",
                executable="motion_planning_node.py",
                name="concrete_block_motion_planning_node",
                output="screen",
                parameters=[
                    LaunchConfiguration("motion_planning_params_file"),
                    {
                        "default_trajectory_method": LaunchConfiguration(
                            "default_trajectory_method"
                        )
                    },
                    {
                        "named_configurations_file": LaunchConfiguration(
                            "named_configurations_file"
                        )
                    },
                    {"wall_plan_file": LaunchConfiguration("wall_plan_file")},
                    {
                        "geometric_optimized_params_file": LaunchConfiguration(
                            "geometric_optimized_params_file"
                        )
                    },
                    {"execution.enabled": LaunchConfiguration("execution_enabled")},
                    {"execution.backend": LaunchConfiguration("execution_backend")},
                    {
                        "execution.trajectory_topic": LaunchConfiguration(
                            "execution_trajectory_topic"
                        )
                    },
                    {
                        "execution.action_name": LaunchConfiguration(
                            "execution_action_name"
                        )
                    },
                    {
                        "execution.result_timeout_s": LaunchConfiguration(
                            "execution_result_timeout_s"
                        )
                    },
                    {
                        "execution.switch_controller": LaunchConfiguration(
                            "execution_switch_controller"
                        )
                    },
                    {
                        "execution.switch_service": LaunchConfiguration(
                            "execution_switch_service"
                        )
                    },
                    {
                        "execution.activate_controller": LaunchConfiguration(
                            "execution_activate_controller"
                        )
                    },
                    {
                        "execution.deactivate_after_execution": LaunchConfiguration(
                            "execution_deactivate_after_execution"
                        )
                    },
                    {
                        "world_model.get_coarse_blocks_service": LaunchConfiguration(
                            "world_model_get_coarse_blocks_service"
                        )
                    },
                    {"use_sim_time": LaunchConfiguration("use_sim_time")},
                ],
            ),
        ]
    )
