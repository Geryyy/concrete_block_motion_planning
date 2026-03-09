from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from rclpy.node import Node


@dataclass(frozen=True)
class NodeConfig:
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
    execution_enabled: bool
    named_configurations_file: str
    default_named_joint_names: List[str]
    named_cfg_default_duration_s: float
    wall_plan_file: str
    default_wall_plan_name: str
    wall_plan_frame_id: str


def _vec3_or_default(values: object, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if isinstance(values, (list, tuple)) and len(values) == 3:
        return (float(values[0]), float(values[1]), float(values[2]))
    return default


def declare_and_load_config(node: Node) -> NodeConfig:
    node.declare_parameter("default_geometric_method", "POWELL")
    node.declare_parameter("default_trajectory_method", "ACADOS_PATH_FOLLOWING")
    node.declare_parameter("path_interpolation_points", 81)
    node.declare_parameter("moving_block_size", [0.6, 0.9, 0.6])
    node.declare_parameter("geometric_optimized_params_file", "")
    node.declare_parameter("trajectory.default_horizon_steps", 120)
    node.declare_parameter("trajectory.fast_horizon_steps", 70)
    node.declare_parameter("trajectory.ctrl_points_min", 4)
    node.declare_parameter("trajectory.ctrl_points_max", 10)
    node.declare_parameter("trajectory.acados_verbose", False)
    node.declare_parameter("execution.enabled", False)

    node.declare_parameter("named_configurations_file", "")
    node.declare_parameter("default_named_configuration_joint_names", [])
    node.declare_parameter("named_configuration_default_duration_s", 4.0)

    node.declare_parameter("wall_plan_file", "")
    node.declare_parameter("default_wall_plan_name", "basic_interlocking_3_2")
    node.declare_parameter("wall_plan_frame_id", "world")

    return NodeConfig(
        default_geometric_method=str(node.get_parameter("default_geometric_method").value),
        default_trajectory_method=str(node.get_parameter("default_trajectory_method").value),
        path_interpolation_points=max(2, int(node.get_parameter("path_interpolation_points").value)),
        moving_block_size=_vec3_or_default(
            node.get_parameter("moving_block_size").value,
            default=(0.6, 0.9, 0.6),
        ),
        optimized_params_file=str(node.get_parameter("geometric_optimized_params_file").value),
        traj_default_horizon=int(node.get_parameter("trajectory.default_horizon_steps").value),
        traj_fast_horizon=int(node.get_parameter("trajectory.fast_horizon_steps").value),
        traj_ctrl_pts_min=int(node.get_parameter("trajectory.ctrl_points_min").value),
        traj_ctrl_pts_max=int(node.get_parameter("trajectory.ctrl_points_max").value),
        traj_acados_verbose=bool(node.get_parameter("trajectory.acados_verbose").value),
        execution_enabled=bool(node.get_parameter("execution.enabled").value),
        named_configurations_file=str(node.get_parameter("named_configurations_file").value),
        default_named_joint_names=[
            str(v) for v in node.get_parameter("default_named_configuration_joint_names").value
        ],
        named_cfg_default_duration_s=float(
            node.get_parameter("named_configuration_default_duration_s").value
        ),
        wall_plan_file=str(node.get_parameter("wall_plan_file").value),
        default_wall_plan_name=str(node.get_parameter("default_wall_plan_name").value),
        wall_plan_frame_id=str(node.get_parameter("wall_plan_frame_id").value),
    )
