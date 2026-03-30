from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from rclpy.node import Node
from rclpy.parameter import Parameter


@dataclass(frozen=True)
class NodeConfig:
    planner_backend: str
    timber_a2b_service: str
    timber_grip_service: str
    compatibility_a2b_service_enabled: bool
    compatibility_a2b_service_name: str
    timber_goal_frame: str
    timber_move_empty_target_z: float
    timber_payload_density_kg_m3: float
    timber_payload_grippoint_xyz: Tuple[float, float, float]
    default_geometric_method: str
    default_trajectory_method: str
    path_interpolation_points: int
    moving_block_size: Tuple[float, float, float]
    optimized_params_file: str
    traj_default_horizon: int
    traj_fast_horizon: int
    traj_ctrl_pts_min: int
    traj_ctrl_pts_max: int
    traj_acados_verbose: bool
    traj_fixed_duration_s: float
    traj_fixed_num_points: int
    traj_toppra_gridpoints: int
    traj_joint_position_limits_file: str
    traj_joint_accel_limits_file: str
    execution_enabled: bool
    execution_backend: str
    execution_trajectory_topic: str
    execution_action_name: str
    execution_joint_states_topic: str
    execution_result_timeout_s: float
    execution_motion_check_timeout_s: float
    execution_motion_min_delta: float
    execution_switch_controller: bool
    execution_switch_service: str
    execution_activate_controller: str
    execution_deactivate_after_execution: bool
    robot_description_topic: str
    world_model_get_coarse_blocks_service: str
    world_model_get_planning_scene_service: str
    named_configurations_file: str
    default_named_joint_names: List[str]
    named_cfg_default_duration_s: float
    wall_plan_file: str
    default_wall_plan_name: str
    wall_plan_frame_id: str


def _vec3_or_default(
    values: object, default: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    if isinstance(values, (list, tuple)) and len(values) == 3:
        return (float(values[0]), float(values[1]), float(values[2]))
    return default


def declare_and_load_config(node: Node) -> NodeConfig:
    package_root = Path(__file__).resolve().parents[2]
    default_planning_limits = package_root / "motion_planning" / "trajectory" / "planning_limits.yaml"
    default_joint_accel_limits = package_root / "motion_planning" / "trajectory" / "joint_accel_limits.yaml"

    node.declare_parameter("planner.backend", "concrete")
    node.declare_parameter("planner.timber_a2b_service", "a2b_movement")
    node.declare_parameter("planner.timber_grip_service", "grip_traj_movement")
    node.declare_parameter("compatibility.a2b_service_enabled", False)
    node.declare_parameter("compatibility.a2b_service_name", "a2b_movement")
    node.declare_parameter("planner.timber_goal_frame", "K0_mounting_base")
    node.declare_parameter("planner.timber_move_empty_target_z", 2.36)
    node.declare_parameter("planner.timber_payload_density_kg_m3", 2400.0)
    node.declare_parameter("planner.timber_payload_grippoint_xyz", [0.0, 0.0, 0.0])
    node.declare_parameter("default_geometric_method", "POWELL")
    node.declare_parameter("default_trajectory_method", "TOPPRA_PATH_FOLLOWING")
    node.declare_parameter("path_interpolation_points", 81)
    node.declare_parameter("moving_block_size", [0.6, 0.9, 0.6])
    node.declare_parameter("geometric_optimized_params_file", "")
    node.declare_parameter("trajectory.default_horizon_steps", 120)
    node.declare_parameter("trajectory.fast_horizon_steps", 70)
    node.declare_parameter("trajectory.ctrl_points_min", 4)
    node.declare_parameter("trajectory.ctrl_points_max", 10)
    node.declare_parameter("trajectory.acados_verbose", False)
    node.declare_parameter("trajectory.fixed_duration_s", 10.0)
    node.declare_parameter("trajectory.fixed_num_points", 121)
    node.declare_parameter("trajectory.toppra_gridpoints", 121)
    node.declare_parameter(
        "trajectory.joint_position_limits_file",
        str(default_planning_limits),
    )
    node.declare_parameter(
        "trajectory.joint_accel_limits_file",
        str(default_joint_accel_limits),
    )
    node.declare_parameter("execution.enabled", False)
    node.declare_parameter("execution.backend", "topic")
    node.declare_parameter(
        "execution.trajectory_topic", "/trajectory_controllers/joint_trajectory"
    )
    node.declare_parameter(
        "execution.action_name",
        "/trajectory_controller_a2b/follow_joint_trajectory",
    )
    node.declare_parameter("execution.joint_states_topic", "/joint_states")
    node.declare_parameter("execution.result_timeout_s", 120.0)
    node.declare_parameter("execution.motion_check_timeout_s", 1.5)
    node.declare_parameter("execution.motion_min_delta", 0.02)
    node.declare_parameter("execution.switch_controller", False)
    node.declare_parameter(
        "execution.switch_service", "/controller_manager/switch_controller"
    )
    node.declare_parameter("execution.activate_controller", "trajectory_controllers")
    node.declare_parameter("execution.deactivate_after_execution", True)
    node.declare_parameter("robot_description_topic", "/robot_description_full")
    node.declare_parameter(
        "world_model.get_coarse_blocks_service",
        "/world_model_node/get_coarse_blocks",
    )
    node.declare_parameter(
        "world_model.get_planning_scene_service",
        "/world_model_node/get_planning_scene",
    )

    node.declare_parameter("named_configurations_file", "")
    node.declare_parameter(
        "default_named_configuration_joint_names",
        [
            "theta1_slewing_joint",
            "theta2_boom_joint",
            "theta3_arm_joint",
            "q4_big_telescope",
            "theta8_rotator_joint",
            "q9_left_rail_joint",
        ],
    )
    node.declare_parameter("named_configuration_default_duration_s", 4.0)

    node.declare_parameter("wall_plan_file", "")
    node.declare_parameter("default_wall_plan_name", "basic_interlocking_3_2")
    node.declare_parameter("wall_plan_frame_id", "world")

    def _param_str(name: str, fallback: Path | None = None) -> str:
        value = str(node.get_parameter(name).value)
        if value.strip():
            return value
        return str(fallback) if fallback is not None else value

    return NodeConfig(
        planner_backend=str(node.get_parameter("planner.backend").value),
        timber_a2b_service=str(node.get_parameter("planner.timber_a2b_service").value),
        timber_grip_service=str(node.get_parameter("planner.timber_grip_service").value),
        compatibility_a2b_service_enabled=bool(
            node.get_parameter("compatibility.a2b_service_enabled").value
        ),
        compatibility_a2b_service_name=str(
            node.get_parameter("compatibility.a2b_service_name").value
        ),
        timber_goal_frame=str(node.get_parameter("planner.timber_goal_frame").value),
        timber_move_empty_target_z=float(
            node.get_parameter("planner.timber_move_empty_target_z").value
        ),
        timber_payload_density_kg_m3=float(
            node.get_parameter("planner.timber_payload_density_kg_m3").value
        ),
        timber_payload_grippoint_xyz=_vec3_or_default(
            node.get_parameter("planner.timber_payload_grippoint_xyz").value,
            default=(0.0, 0.0, 0.0),
        ),
        default_geometric_method=str(
            node.get_parameter("default_geometric_method").value
        ),
        default_trajectory_method=str(
            node.get_parameter("default_trajectory_method").value
        ),
        path_interpolation_points=max(
            2, int(node.get_parameter("path_interpolation_points").value)
        ),
        moving_block_size=_vec3_or_default(
            node.get_parameter("moving_block_size").value,
            default=(0.6, 0.9, 0.6),
        ),
        optimized_params_file=str(
            node.get_parameter("geometric_optimized_params_file").value
        ),
        traj_default_horizon=int(
            node.get_parameter("trajectory.default_horizon_steps").value
        ),
        traj_fast_horizon=int(
            node.get_parameter("trajectory.fast_horizon_steps").value
        ),
        traj_ctrl_pts_min=int(node.get_parameter("trajectory.ctrl_points_min").value),
        traj_ctrl_pts_max=int(node.get_parameter("trajectory.ctrl_points_max").value),
        traj_acados_verbose=bool(node.get_parameter("trajectory.acados_verbose").value),
        traj_fixed_duration_s=float(
            node.get_parameter("trajectory.fixed_duration_s").value
        ),
        traj_fixed_num_points=max(
            2, int(node.get_parameter("trajectory.fixed_num_points").value)
        ),
        traj_toppra_gridpoints=max(
            10, int(node.get_parameter("trajectory.toppra_gridpoints").value)
        ),
        traj_joint_position_limits_file=_param_str(
            "trajectory.joint_position_limits_file", default_planning_limits
        ),
        traj_joint_accel_limits_file=_param_str(
            "trajectory.joint_accel_limits_file", default_joint_accel_limits
        ),
        execution_enabled=bool(node.get_parameter("execution.enabled").value),
        execution_backend=str(node.get_parameter("execution.backend").value),
        execution_trajectory_topic=str(
            node.get_parameter("execution.trajectory_topic").value
        ),
        execution_action_name=str(node.get_parameter("execution.action_name").value),
        execution_joint_states_topic=str(
            node.get_parameter("execution.joint_states_topic").value
        ),
        execution_result_timeout_s=float(
            node.get_parameter("execution.result_timeout_s").value
        ),
        execution_motion_check_timeout_s=float(
            node.get_parameter("execution.motion_check_timeout_s").value
        ),
        execution_motion_min_delta=float(
            node.get_parameter("execution.motion_min_delta").value
        ),
        execution_switch_controller=bool(
            node.get_parameter("execution.switch_controller").value
        ),
        execution_switch_service=str(
            node.get_parameter("execution.switch_service").value
        ),
        execution_activate_controller=str(
            node.get_parameter("execution.activate_controller").value
        ),
        execution_deactivate_after_execution=bool(
            node.get_parameter("execution.deactivate_after_execution").value
        ),
        robot_description_topic=str(
            node.get_parameter("robot_description_topic").value
        ),
        world_model_get_coarse_blocks_service=str(
            node.get_parameter("world_model.get_coarse_blocks_service").value
        ),
        world_model_get_planning_scene_service=str(
            node.get_parameter("world_model.get_planning_scene_service").value
        ),
        named_configurations_file=str(
            node.get_parameter("named_configurations_file").value
        ),
        default_named_joint_names=[
            str(v)
            for v in node.get_parameter("default_named_configuration_joint_names").value
        ],
        named_cfg_default_duration_s=float(
            node.get_parameter("named_configuration_default_duration_s").value
        ),
        wall_plan_file=str(node.get_parameter("wall_plan_file").value),
        default_wall_plan_name=str(node.get_parameter("default_wall_plan_name").value),
        wall_plan_frame_id=str(node.get_parameter("wall_plan_frame_id").value),
    )
