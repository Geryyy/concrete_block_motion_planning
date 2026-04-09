#!/usr/bin/env python3
"""Simple grip trajectory server.

Drop-in replacement for timber_crane_motion_planning/grip_traj_server.
Uses cosine-interpolated joint-space trajectories with analytic IK.
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path as NavPath
from sensor_msgs.msg import JointState
from tf2_ros import StaticTransformBroadcaster
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from timber_crane_planning_interfaces.srv import CalcGripMovement

# Set up import paths for motion_planning library
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
# Add scripts/ dir so "import grip_trajectory" works
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
# Add package root so "from motion_planning.mechanics import ..." works
for _candidate in [_here, _here.parent]:
    if (_candidate / "motion_planning").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
else:
    try:
        from ament_index_python.packages import get_package_share_directory
        _share = Path(get_package_share_directory("concrete_block_motion_planning"))
        if str(_share) not in sys.path:
            sys.path.insert(0, str(_share))
    except Exception:
        pass

from grip_trajectory import GripTrajectoryConfig, compute_grip_trajectory  # noqa: E402
from motion_planning.mechanics import (  # noqa: E402
    AnalyticModelConfig,
    ModelDescription,
    create_crane_config,
)
from motion_planning.mechanics.crane_geometry import DEFAULT_CRANE_GEOMETRY  # noqa: E402
from motion_planning.mechanics.inverse_kinematics import AnalyticIKSolver  # noqa: E402
from motion_planning.mechanics.pinocchio_utils import fk_homogeneous, frame_id  # noqa: E402
from motion_planning.mechanics.pose_conventions import pose_from_pos_yaw  # noqa: E402


CONTROLLED_JOINT_NAMES = [
    "theta1_slewing_joint",
    "theta2_boom_joint",
    "theta3_arm_joint",
    "q4_big_telescope",
    "theta6_tip_joint",
    "theta7_tilt_joint",
    "theta8_rotator_joint",
    "q9_left_rail_joint",
]

# IK uses 7 dynamic joints (no gripper); these are the actuated subset for IK
IK_ACTUATED = [
    "theta1_slewing_joint",
    "theta2_boom_joint",
    "theta3_arm_joint",
    "q4_big_telescope",
    "theta8_rotator_joint",
]


class GripTrajServerSimple(Node):
    def __init__(self):
        super().__init__("grip_traj_server")

        # Parameters
        self.declare_parameter("service_name", "grip_traj_movement")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter(
            "controlled_joint_names", CONTROLLED_JOINT_NAMES
        )
        self.declare_parameter("dt_target", 0.01)
        self.declare_parameter("lift_height", 0.5)
        self.declare_parameter("gripper_open_position", 0.15)
        self.declare_parameter("gripper_close_position", 0.0)
        # Scale factor for gripper command: 0.5 in sim (mimic doubles it), 1.0 on real HW
        self.declare_parameter("gripper_command_scale", 0.5)
        # Offset from K8_tool_center_point to actual grip point (PZS100 rail length).
        # IK targets K8, so we shift the target up by this amount so the grip
        # point (virtual TCP) reaches the desired position.
        self.declare_parameter("tcp_z_offset", 0.7)
        # Offset from block CoG to the grip point on top of the block.
        # For a ~0.6m tall block this is ~0.3m (half the block height).
        # Combined with tcp_z_offset this gives the full CoG→K8 offset.
        self.declare_parameter("block_grip_z_offset", -0.3)
        # Per-segment durations (seconds) — tune these for commissioning
        self.declare_parameter("duration_gripper", 2.0)
        self.declare_parameter("duration_descend", 5.0)
        self.declare_parameter("duration_lift", 5.0)

        self._joint_names = (
            self.get_parameter("controlled_joint_names").value
        )
        grip_scale = self.get_parameter("gripper_command_scale").value
        self._cfg = GripTrajectoryConfig(
            dt=self.get_parameter("dt_target").value,
            lift_height=self.get_parameter("lift_height").value,
            gripper_open_position=self.get_parameter("gripper_open_position").value * grip_scale,
            gripper_close_position=self.get_parameter("gripper_close_position").value * grip_scale,
            duration_gripper=self.get_parameter("duration_gripper").value,
            duration_descend=self.get_parameter("duration_descend").value,
            duration_lift=self.get_parameter("duration_lift").value,
        )
        self._gripper_index = len(self._joint_names) - 1  # last joint is gripper
        self._tcp_z_offset = self.get_parameter("tcp_z_offset").value
        self._block_grip_z_offset = self.get_parameter("block_grip_z_offset").value

        # Joint state subscription
        self._latest_positions: dict[str, float] = {}
        self._js_stamp = None
        js_topic = self.get_parameter("joint_states_topic").value
        self.create_subscription(JointState, js_topic, self._on_joint_state, 10)

        # Initialize IK solver
        config = create_crane_config()
        self._desc = ModelDescription(config)
        self._ik_solver = AnalyticIKSolver(self._desc, config, DEFAULT_CRANE_GEOMETRY)
        self._config = config
        self._frame_cache: dict[str, int] = {}

        # Path visualization publisher (same topics as timber grip_traj_server)
        self._path_pub = self.create_publisher(NavPath, "tcp_path", 10)

        # Publish virtual_tcp TF derived from tcp_z_offset (single source of truth)
        self._tf_broadcaster = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "K8_tool_center_point"
        t.child_frame_id = "virtual_tcp"
        t.transform.translation.z = self._tcp_z_offset
        t.transform.rotation.w = 1.0
        self._tf_broadcaster.sendTransform(t)
        self.get_logger().info(
            f"Published virtual_tcp TF: z={self._tcp_z_offset:.3f}m from K8"
        )

        # Service
        svc_name = self.get_parameter("service_name").value
        self.create_service(CalcGripMovement, svc_name, self._handle_request)
        self.get_logger().info(
            f"Simple grip trajectory server ready on '{svc_name}'"
        )

    def _on_joint_state(self, msg: JointState):
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self._latest_positions[name] = msg.position[i]
        self._js_stamp = msg.header.stamp

    def _get_q0(self) -> np.ndarray | None:
        """Extract current joint positions ordered by controlled_joint_names."""
        if not self._latest_positions:
            return None
        q0 = np.zeros(len(self._joint_names))
        for i, name in enumerate(self._joint_names):
            if name not in self._latest_positions:
                self.get_logger().error(f"Joint '{name}' not in joint_states")
                return None
            q0[i] = self._latest_positions[name]
        # Normalize rotator angle to [-pi, pi]
        q0[6] = math.atan2(math.sin(q0[6]), math.cos(q0[6]))
        return q0

    def _ik_solve(
        self, target_xyz: np.ndarray, phi_tool_n: float, seed_q: np.ndarray
    ) -> np.ndarray | None:
        """Solve IK and return full joint vector (including gripper from seed)."""
        T = pose_from_pos_yaw(target_xyz, phi_tool_n)

        # Build seed dict from joint names (excluding gripper)
        seed = {}
        for i, name in enumerate(self._joint_names):
            if i != self._gripper_index:
                seed[name] = float(seed_q[i])

        # Fixed joints (passive)
        fixed = {
            "theta6_tip_joint": float(seed_q[4]),
            "theta7_tilt_joint": float(seed_q[5]),
        }

        result = self._ik_solver.solve(
            target_T_base_to_end=T,
            base_frame="K0_mounting_base",
            end_frame="K8_tool_center_point",
            seed=seed,
            act_names=IK_ACTUATED,
            fixed=fixed,
        )

        if result is None or not result.success:
            msg = "IK returned None" if result is None else result.message
            self.get_logger().error(f"IK failed: {msg}")
            return None

        # Build output joint vector
        q_out = seed_q.copy()
        for name, val in result.q_dynamic.items():
            if name in self._joint_names:
                idx = self._joint_names.index(name)
                q_out[idx] = val
        return q_out

    def _fk_transform(self, q: np.ndarray) -> np.ndarray:
        """Forward kinematics: joint vector → K8 4x4 homogeneous transform in K0."""
        import pinocchio as pin

        q_map = {}
        for i, name in enumerate(self._joint_names):
            if i != self._gripper_index:
                q_map[name] = float(q[i])
        # Add tied joints
        if "q5_small_telescope" not in q_map and "q4_big_telescope" in q_map:
            q_map["q5_small_telescope"] = q_map["q4_big_telescope"]
        # Add locked joints at zero
        for jn in self._config.locked_joints:
            if jn not in q_map:
                q_map[jn] = 0.0

        return fk_homogeneous(
            pin_model=self._desc.model,
            pin_data=self._desc.data,
            pin_module=pin,
            q_values=q_map,
            base_frame="K0_mounting_base",
            end_frame="K8_tool_center_point",
            frame_cache=self._frame_cache,
        )

    def _fk(self, q: np.ndarray) -> np.ndarray:
        """Forward kinematics: joint vector → K8 xyz in K0."""
        return self._fk_transform(q)[:3, 3].copy()

    def _fk_virtual_tcp(self, q: np.ndarray) -> np.ndarray:
        """Forward kinematics: joint vector → virtual TCP xyz in K0."""
        T = self._fk_transform(q)
        # Offset along K8's local z-axis
        return (T[:3, 3] + self._tcp_z_offset * T[:3, 2]).copy()

    def _handle_request(self, request, response):
        q0 = self._get_q0()
        if q0 is None:
            self.get_logger().error("No joint states available")
            response.success = 0
            return response

        # Input is block CoG. Shift up by:
        #   block_grip_z_offset  (CoG → grip point on block top)
        #   tcp_z_offset         (grip point → K8_tool_center_point)
        total_z_offset = self._block_grip_z_offset + self._tcp_z_offset
        target_xyz = np.array(
            [request.y_n.x, request.y_n.y, request.y_n.z + total_z_offset]
        )
        phi_tool_n = request.phi_tool_n
        phase = request.select_phases
        slow_down = max(request.slow_down, 0.1)

        current_xyz = self._fk(q0)
        self.get_logger().info(
            f"Grip request | phase={phase} "
            f"current_K8=({current_xyz[0]:.2f}, {current_xyz[1]:.2f}, {current_xyz[2]:.2f}) "
            f"target_K8=({target_xyz[0]:.2f}, {target_xyz[1]:.2f}, {target_xyz[2]:.2f}) "
            f"z_offsets=block_grip:{self._block_grip_z_offset:.2f}+tcp:{self._tcp_z_offset:.2f}={total_z_offset:.2f}m "
            f"phi={math.degrees(phi_tool_n):.1f}deg slow_down={slow_down:.1f}"
        )

        result = compute_grip_trajectory(
            q0=q0,
            target_xyz=target_xyz,
            phi_tool_n=phi_tool_n,
            phase=phase,
            slow_down=slow_down,
            ik_solve_fn=self._ik_solve,
            fk_fn=self._fk,
            cfg=self._cfg,
            gripper_index=self._gripper_index,
        )

        if not result.success:
            self.get_logger().error(f"Trajectory generation failed: {result.message}")
            response.success = 0
            return response

        # Convert to JointTrajectory msg
        traj = JointTrajectory()
        traj.joint_names = list(self._joint_names)

        for i in range(len(result.times)):
            pt = JointTrajectoryPoint()
            pt.positions = result.q_traj[i].tolist()
            pt.velocities = result.qd_traj[i].tolist()
            pt.accelerations = result.qdd_traj[i].tolist()
            t = result.times[i]
            pt.time_from_start = Duration(
                sec=int(t), nanosec=int((t - int(t)) * 1e9)
            )
            traj.points.append(pt)

        # Ensure final point has zero velocity
        if traj.points:
            traj.points[-1].velocities = [0.0] * len(self._joint_names)
            traj.points[-1].accelerations = [0.0] * len(self._joint_names)

        response.success = 1
        response.trajectory = traj
        response.tcp_path = []

        # Publish TCP path for RViz visualization
        self._publish_tcp_path(result.q_traj)

        self.get_logger().info(
            f"Trajectory computed | {len(traj.points)} points, "
            f"duration={result.times[-1]:.2f}s"
        )
        return response

    def _publish_tcp_path(self, q_traj: np.ndarray):
        """Compute FK for each trajectory point and publish as nav_msgs/Path."""
        path_msg = NavPath()
        path_msg.header.frame_id = "K0_mounting_base"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        # Sample every 10th point to keep it lightweight
        step = max(1, len(q_traj) // 50)
        for i in range(0, len(q_traj), step):
            xyz = self._fk_virtual_tcp(q_traj[i])
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(xyz[0])
            pose.pose.position.y = float(xyz[1])
            pose.pose.position.z = float(xyz[2])
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        # Always include last point
        if len(q_traj) > 0:
            xyz = self._fk(q_traj[-1])
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(xyz[0])
            pose.pose.position.y = float(xyz[1])
            pose.pose.position.z = float(xyz[2])
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self._path_pub.publish(path_msg)


def main():
    rclpy.init()
    node = GripTrajServerSimple()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
