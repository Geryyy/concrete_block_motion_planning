from __future__ import annotations

from typing import Any, Dict, List

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformListener
from std_msgs.msg import String
from nav_msgs.msg import Path as NavPath
from trajectory_msgs.msg import JointTrajectory
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import SwitchController
from concrete_block_perception.srv import GetCoarseBlocks

from concrete_block_motion_planning.srv import (
    ComputeTrajectory,
    ExecuteNamedConfiguration,
    ExecuteTrajectory,
    GetNextAssemblyTask,
    PlanAndComputeTrajectory,
    PlanGeometricPath,
)

from .config import NodeConfig, declare_and_load_config
from .execution import ExecutionAdapter
from .named_configurations import NamedConfigurationResolver
from .backends import ConcretePlannerBackend, TimberPlannerBackend
from .runtime import RuntimeHelpersMixin
from .services import ServiceHandlersMixin
from .results import PlannerCapabilities
from .state import MotionPlanningState, RuntimeStatus
from .types import StoredGeometricPlan, StoredTrajectory, WallPlanTask


class ConcreteBlockMotionPlanningNode(ServiceHandlersMixin, RuntimeHelpersMixin, Node):
    def __init__(self) -> None:
        super().__init__("concrete_block_motion_planning_node")

        self._cfg: NodeConfig = declare_and_load_config(self)
        self._state = MotionPlanningState()

        # Bind mixin-facing attribute views while the node is split across helper classes.
        self._bind_state_aliases()
        self._bind_config_aliases()

        self._status_pub = self.create_publisher(String, "~/trajectory_backend_status", 10)
        self._planned_path_pub = self.create_publisher(NavPath, "~/planned_path", 10)
        self._trajectory_cmd_pub = None
        self._trajectory_action_client = None
        self._switch_controller_client = None
        self._get_coarse_blocks_client = None
        self._robot_description_sub = None
        self._tf_buffer = None
        self._tf_listener = None
        self._robot_description_xml = ""
        self._initialize_execution_io()
        self._initialize_world_model_io()
        self._initialize_robot_description_io()
        self._initialize_planner_backend_io()
        self._execution_adapter = ExecutionAdapter(self)
        self._named_configuration_resolver = NamedConfigurationResolver(
            named_configurations=self._state.named_configurations,
            trajectories=self._state.trajectories,
        )
        self._planner_backend = self._create_planner_backend()
        self._planner_capabilities: PlannerCapabilities = self._planner_backend.capabilities

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
        self._planner_backend_name = self._cfg.planner_backend.strip().lower()
        self._timber_a2b_service = self._cfg.timber_a2b_service.strip()
        self._timber_goal_frame = self._cfg.timber_goal_frame.strip()
        self._timber_move_empty_target_z = float(self._cfg.timber_move_empty_target_z)
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
        self._traj_fixed_duration_s = max(0.1, float(self._cfg.traj_fixed_duration_s))
        self._traj_fixed_num_points = max(2, int(self._cfg.traj_fixed_num_points))
        self._execution_enabled = self._cfg.execution_enabled
        self._execution_backend = self._cfg.execution_backend.strip().lower()
        self._execution_trajectory_topic = self._cfg.execution_trajectory_topic.strip()
        self._execution_action_name = self._cfg.execution_action_name.strip()
        self._execution_result_timeout_s = max(1.0, float(self._cfg.execution_result_timeout_s))
        self._execution_switch_controller = bool(self._cfg.execution_switch_controller)
        self._execution_switch_service = self._cfg.execution_switch_service.strip()
        self._execution_activate_controller = self._cfg.execution_activate_controller.strip()
        self._execution_deactivate_after_execution = bool(self._cfg.execution_deactivate_after_execution)
        self._robot_description_topic = self._cfg.robot_description_topic.strip()
        self._world_model_get_coarse_blocks_service = (
            self._cfg.world_model_get_coarse_blocks_service.strip()
        )

        self._named_configurations_file = self._cfg.named_configurations_file
        self._default_named_joint_names = list(self._cfg.default_named_joint_names)
        self._named_cfg_default_duration_s = self._cfg.named_cfg_default_duration_s

        self._wall_plan_file = self._cfg.wall_plan_file
        self._default_wall_plan_name = self._cfg.default_wall_plan_name
        self._wall_plan_frame_id = self._cfg.wall_plan_frame_id

    def _initialize_runtime_and_data(self) -> None:
        self._ensure_motion_planning_module_path()
        if self._planner_backend_name == "concrete":
            self._geometric_runtime_available, self._geometric_runtime_reason = self._check_geometric_runtime()
            self._trajectory_runtime_available, self._trajectory_runtime_reason = self._check_trajectory_runtime()
            if self._robot_description_topic and not self._robot_description_xml:
                self._planning_runtime_ready = False
                self._planning_runtime_reason = (
                    f"waiting for robot_description on '{self._robot_description_topic}'"
                )
            else:
                self._initialize_planning_runtime()
        else:
            self._geometric_runtime_available = False
            self._geometric_runtime_reason = (
                f"geometric stage owned by planner backend '{self._planner_backend_name}'"
            )
            self._trajectory_runtime_available = True
            self._trajectory_runtime_reason = (
                f"trajectory generation handled by planner backend '{self._planner_backend_name}'"
            )
            self._planning_runtime_ready = True
            self._planning_runtime_reason = (
                f"handled by planner backend '{self._planner_backend_name}'"
            )

        self._load_named_configurations_from_file(self._named_configurations_file)
        self._load_wall_plans_from_file(self._wall_plan_file)

        self._publish_backend_status(
            "AVAILABLE" if self._planning_runtime_ready else "UNAVAILABLE",
            (
                f"planning_runtime={self._planning_runtime_ready} ({self._planning_runtime_reason}); "
                f"planner_backend={self._planner_backend.backend_name}; "
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
        self._execution_cb_group = ReentrantCallbackGroup()
        if self._execution_backend == "action":
            if not self._execution_action_name:
                self.get_logger().warn(
                    "execution.enabled=true, backend=action but execution.action_name is empty; execution disabled."
                )
                self._execution_enabled = False
                return
            self._trajectory_action_client = ActionClient(
                self,
                FollowJointTrajectory,
                self._execution_action_name,
                callback_group=self._execution_cb_group,
            )
            if self._execution_switch_controller:
                if not self._execution_switch_service:
                    self.get_logger().warn(
                        "execution.switch_controller=true but execution.switch_service is empty; controller switching disabled."
                    )
                    self._execution_switch_controller = False
                else:
                    self._switch_controller_client = self.create_client(
                        SwitchController,
                        self._execution_switch_service,
                        callback_group=self._execution_cb_group,
                    )
            return

        if self._execution_backend != "topic":
            self.get_logger().warn(
                f"Unknown execution.backend '{self._execution_backend}'. Falling back to topic backend."
            )
            self._execution_backend = "topic"

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

    def _initialize_world_model_io(self) -> None:
        if not self._world_model_get_coarse_blocks_service:
            return
        self._get_coarse_blocks_client = self.create_client(
            GetCoarseBlocks,
            self._world_model_get_coarse_blocks_service,
        )

    def _initialize_robot_description_io(self) -> None:
        if not self._robot_description_topic:
            return
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._robot_description_sub = self.create_subscription(
            String,
            self._robot_description_topic,
            self._on_robot_description,
            qos,
        )

    def _initialize_planner_backend_io(self) -> None:
        if self._planner_backend_name != "timber":
            return
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

    def _create_planner_backend(self):
        if self._planner_backend_name == "timber":
            return TimberPlannerBackend(self)
        if self._planner_backend_name not in ("", "concrete"):
            self.get_logger().warn(
                f"Unknown planner.backend '{self._planner_backend_name}', falling back to concrete."
            )
            self._planner_backend_name = "concrete"
        return ConcretePlannerBackend(self)

    @staticmethod
    def _empty_trajectory() -> JointTrajectory:
        return JointTrajectory()

    def _on_robot_description(self, msg: String) -> None:
        if self._planner_backend_name != "concrete":
            return
        xml = str(msg.data)
        if not xml or xml == self._robot_description_xml:
            return
        self._robot_description_xml = xml
        self.get_logger().info(
            f"Received robot_description from '{self._robot_description_topic}'."
        )
        self._initialize_planning_runtime()
        if self._planning_runtime_ready:
            self.get_logger().info(
                "Planning runtime reinitialized successfully from robot_description."
            )
        else:
            self.get_logger().warn(
                "Planning runtime still unavailable after robot_description update: "
                f"{self._planning_runtime_reason}"
            )

    def _register_services(self) -> None:
        # New stage-split services
        self._plan_geo_srv = self.create_service(
            PlanGeometricPath,
            "~/plan_geometric_path",
            self._handle_plan_geometric,
        )
        self._plan_and_compute_srv = self.create_service(
            PlanAndComputeTrajectory,
            "~/plan_and_compute_trajectory",
            self._handle_plan_and_compute_trajectory,
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
            f"execution_backend={self._execution_backend} | "
            f"execution_action={self._execution_action_name or '<none>'} | "
            f"execution_topic={self._execution_trajectory_topic or '<none>'} | "
            f"named_configurations={len(self._named_configurations)} | "
            f"wall_plans={len(self._wall_plans)}"
        )
        if not self._planning_runtime_ready and not (
            self._robot_description_topic and not self._robot_description_xml
        ):
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
