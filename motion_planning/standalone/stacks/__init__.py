from .joint_space_global_path import plan_joint_space_global_path


STACK_REGISTRY = {
    "joint_space_global_path": plan_joint_space_global_path,
}

__all__ = ["STACK_REGISTRY", "plan_joint_space_global_path"]
