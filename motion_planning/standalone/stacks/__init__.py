from .cartesian_anchor_joint_spline import plan_cartesian_anchor_joint_spline
from .joint_goal_interpolation import plan_joint_goal_interpolation
from ..time_parameterization import apply_simple_time_scaling

STACK_REGISTRY = {
    "joint_goal_interpolation": plan_joint_goal_interpolation,
    "cartesian_anchor_joint_spline": plan_cartesian_anchor_joint_spline,
}

__all__ = [
    "STACK_REGISTRY",
    "apply_simple_time_scaling",
    "plan_cartesian_anchor_joint_spline",
    "plan_joint_goal_interpolation",
]
