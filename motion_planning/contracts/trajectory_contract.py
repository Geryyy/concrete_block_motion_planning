from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .robot_profile import JointNames, RobotProfile


@dataclass(frozen=True)
class TrajectoryContract:
    """Shared motion contract shape exposed by backend adapters."""

    profile: RobotProfile
    trajectory: Any
    tcp_path: Any = None
    planner_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        joint_names = getattr(self.trajectory, "joint_names", None)
        if joint_names is None:
            return
        if tuple(joint_names) != self.profile.full_state_joint_names:
            raise ValueError(
                "trajectory joint_names do not match profile full_state_joint_names: "
                f"{tuple(joint_names)} != {self.profile.full_state_joint_names}"
            )

    @property
    def full_state_joint_names(self) -> JointNames:
        return self.profile.full_state_joint_names

    @property
    def command_joint_names(self) -> JointNames:
        return self.profile.command_joint_names

    @property
    def passive_joint_names(self) -> JointNames:
        return self.profile.passive_joint_names
