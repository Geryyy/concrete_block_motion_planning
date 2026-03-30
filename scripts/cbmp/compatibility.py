from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from geometry_msgs.msg import Point, PoseStamped
from trajectory_msgs.msg import JointTrajectory

from timber_crane_planning_interfaces.srv import CalcMovement


PZS100_A2B_JOINT_ORDER: tuple[str, ...] = (
    "theta1_slewing_joint",
    "theta2_boom_joint",
    "theta3_arm_joint",
    "q4_big_telescope",
    "theta6_tip_joint",
    "theta7_tilt_joint",
    "theta8_rotator_joint",
    "q9_left_rail_joint",
)


@dataclass(frozen=True)
class A2BCompatibilityRequest:
    raw_request: CalcMovement.Request
    goal_pose: PoseStamped
    use_current_state: bool
    start_joint_positions: tuple[float, ...] | None
    slow_down: float
    t_end: float


def uses_current_joint_state(q0: Sequence[float]) -> bool:
    return all(abs(float(value)) < 1e-9 for value in q0)


def build_goal_pose_from_request(
    request: CalcMovement.Request,
    *,
    frame_id: str,
    yaw_to_quaternion,
) -> PoseStamped:
    goal_pose = PoseStamped()
    goal_pose.header.frame_id = str(frame_id)
    goal_pose.pose.position = Point(
        x=float(request.y_n.x),
        y=float(request.y_n.y),
        z=float(request.y_n.z),
    )
    goal_pose.pose.orientation = yaw_to_quaternion(float(request.phi_tool_n))
    return goal_pose


def translate_calc_movement_request(
    request: CalcMovement.Request,
    *,
    frame_id: str,
    yaw_to_quaternion,
) -> A2BCompatibilityRequest:
    use_current = uses_current_joint_state(request.q0)
    return A2BCompatibilityRequest(
        raw_request=request,
        goal_pose=build_goal_pose_from_request(
            request,
            frame_id=frame_id,
            yaw_to_quaternion=yaw_to_quaternion,
        ),
        use_current_state=use_current,
        start_joint_positions=None if use_current else tuple(float(v) for v in request.q0),
        slow_down=float(request.slow_down),
        t_end=float(request.t_end),
    )


def make_empty_compat_trajectory() -> JointTrajectory:
    trajectory = JointTrajectory()
    trajectory.joint_names = list(PZS100_A2B_JOINT_ORDER)
    return trajectory
