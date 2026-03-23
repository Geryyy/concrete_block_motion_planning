from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CartesianPathFollowingConfig:
    """Configuration for the Cartesian task-space path-following OCP."""

    urdf_path: Path
    horizon_steps: int = 100
    actuated_joints: Sequence[str] = (
        "theta1_slewing_joint",
        "theta2_boom_joint",
        "theta3_arm_joint",
        "q4_big_telescope",
        "theta8_rotator_joint",
    )
    passive_joints: Sequence[str] = (
        "theta6_tip_joint",
        "theta7_tilt_joint",
    )
    lock_joint_names: Sequence[str] = (
        "truck_pitch",
        "truck_roll",
        "q9_left_rail_joint",
        "q11_right_rail_joint",
    )
    joint_position_overrides: Mapping[str, Tuple[Optional[float], Optional[float]]] = field(default_factory=dict)
    joint_position_limits_yaml: Path = Path(__file__).resolve().parents[1] / "planning_limits.yaml"
    dynamics_mode: str = "projected"
    passive_solve_damping: float = 1e-6
    spline_degree: int = 3
    spline_ctrl_points: int = 4
    sdot_ref: float = 0.1
    sdot_min: float = 0.0
    sdot_max: float = 1.0
    v_min: float = -0.6
    v_max: float = 0.6
    qdd_u_min: float = -1.0
    qdd_u_max: float = 1.0
    joint_accel_limits_yaml: Path = Path(__file__).resolve().parents[1] / "joint_accel_limits.yaml"
    validate_joint_limits_with_urdf: bool = True
    xyz_weight: float = 250.0
    terminal_xyz_weight: float = 250.0
    yaw_weight: float = 50.0
    terminal_yaw_weight: float = 100.0
    s_weight: float = 5.0
    sdot_weight: float = 3.0
    qdd_u_weight: float = 0.5
    v_weight: float = 1.5
    terminal_s_weight: float = 40.0
    terminal_sdot_weight: float = 6.0
    terminal_dq_weight: float = 5.0
    passive_q_sway_weight: float = 8.0
    passive_dq_sway_weight: float = 150.0
    terminal_passive_q_sway_weight: float = 80.0
    terminal_passive_dq_sway_weight: float = 260.0
    passive_dq_soft_max: float = 0.5
    passive_dq_use_slack: bool = True
    passive_dq_slack_weight: float = 300.0
    terminal_passive_dq_slack_weight: float = 600.0
    passive_dq_soft_abs_eps: float = 1e-6
    terminal_hard_zero_velocity: bool = True
    terminal_hard_end_progress: bool = True
    tool_frame_name: str = "K8_tool_center_point"
    payload_mass: float = 0.0
    payload_com_tcp: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    hessian_approx: str = "GAUSS_NEWTON"
    nlp_solver_type: str = "SQP"
    nlp_solver_max_iter: int = 1000
    qp_solver_iter_max: int = 100
    qp_tol: float = 1e-7
    nlp_tol: float = 1e-6
    terminal_hold_steps: int = 0
    optimize_time: bool = True
    fixed_time_duration_s: Optional[float] = None
    fixed_time_duration_candidates: Tuple[float, ...] = ()
    fixed_time_nominal_tcp_speed: float = 0.18
    fixed_time_min_duration_s: float = 6.0
    fixed_time_max_duration_s: float = 45.0
    fixed_time_sway_slack_s: float = 4.0
    warm_start_progress: bool = True
    code_export_dir: Path = Path("/tmp/crane_cartesian_path_following_codegen")
    solver_json_name: str = "crane_cartesian_path_following_ocp.json"
    precompile_on_init: bool = False
    print_model_prep: bool = False
