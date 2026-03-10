from __future__ import annotations

from typing import Any, Dict, List

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory

from concrete_block_motion_planning.srv import (
    ComputeTrajectory,
    ExecuteNamedConfiguration,
    ExecuteTrajectory,
    GetNextAssemblyTask,
    PlanGeometricPath,
)

from .config import NodeConfig, declare_and_load_config
from .runtime import RuntimeHelpersMixin
from .services import ServiceHandlersMixin
from .state import MotionPlanningState, RuntimeStatus
from .types import StoredGeometricPlan, StoredTrajectory, WallPlanTask


class ConcreteBlockMotionPlanningNode(ServiceHandlersMixin, RuntimeHelpersMixin, Node):
    def __init__(self) -> None:
        super().__init__("concrete_block_motion_planning_node")

        self._cfg: NodeConfig = declare_and_load_config(self)
        self._state = MotionPlanningState()

        # Keep legacy attribute names used by mixins while config/state are explicit.
        self._bind_state_aliases()
        self._bind_config_aliases()

        self._status_pub = self.create_publisher(String, "~/trajectory_backend_status", 10)
        self._trajectory_cmd_pub = None
        self._initialize_execution_io()

        self._initialize_runtime_and_data()
        self._register_services()
        self._log_startup()

    def _bind_state_aliases(self) -> None:
        self._geometric_plans: Dict[str, StoredGeometricPlan] = self._state.geometric_plans
        self._trajectories: Dict[str, StoredTrajectory] = self._state.trajectories
        self._named_configurations: Dict[str, JointTrajectory] = self._state.named_configurations
        self._wall_plans: Dict[str, List[WallPlanTask]] = self._state.wall_plans
        self._wall_plan_progress: Dict[str, int] = self._state.wall_plan_progress

        self._runtime_status: RuntimeStatus = self._state.runtime
        self._planner_scene = self._state.planner_scene
        self._optimized_planner_params = self._state.optimized_planner_params
        self._trajectory_optimizers = self._state.trajectory_optimizers

        self._analytic_cfg = self._state.analytic_cfg
        self._steady_state_solver = self._state.steady_state_solver
        self._reduced_joint_names = self._state.reduced_joint_names
        self._ik_seed_map = self._state.ik_seed_map
        self._T_world_base = self._state.t_world_base
        self._T_base_world = self._state.t_base_world

    @property
    def _planning_runtime_ready(self) -> bool:
        return self._runtime_status.planning_runtime_ready

    @_planning_runtime_ready.setter
    def _planning_runtime_ready(self, value: bool) -> None:
        self._runtime_status.planning_runtime_ready = bool(value)

    @property
    def _planning_runtime_reason(self) -> str:
        return self._runtime_status.planning_runtime_reason

    @_planning_runtime_reason.setter
    def _planning_runtime_reason(self, value: str) -> None:
        self._runtime_status.planning_runtime_reason = str(value)

    @property
    def _trajectory_runtime_available(self) -> bool:
        return self._runtime_status.trajectory_runtime_available

    @_trajectory_runtime_available.setter
    def _trajectory_runtime_available(self, value: bool) -> None:
        self._runtime_status.trajectory_runtime_available = bool(value)

    @property
    def _trajectory_runtime_reason(self) -> str:
        return self._runtime_status.trajectory_runtime_reason

    @_trajectory_runtime_reason.setter
    def _trajectory_runtime_reason(self, value: str) -> None:
        self._runtime_status.trajectory_runtime_reason = str(value)

    @property
    def _geometric_runtime_available(self) -> bool:
        return self._runtime_status.geometric_runtime_available

    @_geometric_runtime_available.setter
    def _geometric_runtime_available(self, value: bool) -> None:
        self._runtime_status.geometric_runtime_available = bool(value)

    @property
    def _geometric_runtime_reason(self) -> str:
        return self._runtime_status.geometric_runtime_reason

    @_geometric_runtime_reason.setter
    def _geometric_runtime_reason(self, value: str) -> None:
        self._runtime_status.geometric_runtime_reason = str(value)

    def _bind_config_aliases(self) -> None:
        self._default_geometric_method = self._cfg.default_geometric_method
        self._default_trajectory_method = self._cfg.default_trajectory_method
        self._n_points = self._cfg.path_interpolation_points
        self._moving_block_size = self._cfg.moving_block_size

        self._optimized_params_file = self._cfg.optimized_params_file
        self._traj_default_horizon = self._cfg.traj_default_horizon
        self._traj_fast_horizon = self._cfg.traj_fast_horizon
        self._traj_ctrl_pts_min = self._cfg.traj_ctrl_pts_min
        self._traj_ctrl_pts_max = self._cfg.traj_ctrl_pts_max
        self._traj_acados_verbose = self._cfg.traj_acados_verbose
        self._execution_enabled = self._cfg.execution_enabled
        self._execution_trajectory_topic = self._cfg.execution_trajectory_topic.strip()

        self._named_configurations_file = self._cfg.named_configurations_file
        self._default_named_joint_names = list(self._cfg.default_named_joint_names)
        self._named_cfg_default_duration_s = self._cfg.named_cfg_default_duration_s

        self._wall_plan_file = self._cfg.wall_plan_file
        self._default_wall_plan_name = self._cfg.default_wall_plan_name
        self._wall_plan_frame_id = self._cfg.wall_plan_frame_id

    def _initialize_runtime_and_data(self) -> None:
        self._ensure_motion_planning_module_path()
        self._geometric_runtime_available, self._geometric_runtime_reason = self._check_geometric_runtime()
        self._trajectory_runtime_available, self._trajectory_runtime_reason = self._check_trajectory_runtime()
        self._initialize_planning_runtime()

        self._load_named_configurations_from_file(self._named_configurations_file)
        self._load_wall_plans_from_file(self._wall_plan_file)

        self._publish_backend_status(
            "AVAILABLE" if self._planning_runtime_ready else "UNAVAILABLE",
            (
                f"planning_runtime={self._planning_runtime_ready} ({self._planning_runtime_reason}); "
                f"geometric_backend={self._geometric_runtime_available} ({self._geometric_runtime_reason}); "
                f"trajectory_backend={self._trajectory_runtime_available} ({self._trajectory_runtime_reason}); "
                f"named_configurations={len(self._named_configurations)}; "
                f"wall_plans={len(self._wall_plans)}; "
                f"execution_enabled={self._execution_enabled}"
            ),
        )

    def _initialize_execution_io(self) -> None:
        if not self._execution_enabled:
            return
        if not self._execution_trajectory_topic:
            self.get_logger().warn(
                "execution.enabled=true but execution.trajectory_topic is empty; execution disabled."
            )
            self._execution_enabled = False
            return
        self._trajectory_cmd_pub = self.create_publisher(
            JointTrajectory,
            self._execution_trajectory_topic,
            10,
        )

    def _register_services(self) -> None:
        # New stage-split services
        self._plan_geo_srv = self.create_service(
            PlanGeometricPath,
            "~/plan_geometric_path",
            self._handle_plan_geometric,
        )
        self._compute_traj_srv = self.create_service(
            ComputeTrajectory,
            "~/compute_trajectory",
            self._handle_compute_trajectory,
        )
        self._execute_traj_srv = self.create_service(
            ExecuteTrajectory,
            "~/execute_trajectory",
            self._handle_execute_trajectory,
        )
        self._execute_named_cfg_srv = self.create_service(
            ExecuteNamedConfiguration,
            "~/execute_named_configuration",
            self._handle_execute_named_configuration,
        )
        self._next_assembly_task_srv = self.create_service(
            GetNextAssemblyTask,
            "~/get_next_assembly_task",
            self._handle_get_next_assembly_task,
        )

    def _log_startup(self) -> None:
        self.get_logger().info(
            "ConcreteBlockMotionPlanningNode ready | "
            f"default_geometric_method={self._default_geometric_method} | "
            f"default_trajectory_method={self._default_trajectory_method} | "
            f"planning_runtime_ready={self._planning_runtime_ready} | "
            f"execution_enabled={self._execution_enabled} | "
            f"execution_topic={self._execution_trajectory_topic or '<none>'} | "
            f"named_configurations={len(self._named_configurations)} | "
            f"wall_plans={len(self._wall_plans)}"
        )
        if not self._planning_runtime_ready:
            self.get_logger().warn(f"Planning runtime disabled: {self._planning_runtime_reason}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ConcreteBlockMotionPlanningNode()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.remove_node(node)
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
