from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from trajectory_msgs.msg import JointTrajectory

from .ids import make_named_trajectory_id
from .types import StoredTrajectory


@dataclass(frozen=True)
class NamedConfigurationResolution:
    success: bool
    message: str
    trajectory_id: str
    trajectory: JointTrajectory | None


class NamedConfigurationResolver:
    def __init__(
        self,
        *,
        named_configurations: Dict[str, JointTrajectory],
        trajectories: Dict[str, StoredTrajectory],
    ) -> None:
        self._named_configurations = named_configurations
        self._trajectories = trajectories

    def resolve(self, configuration_name: str) -> NamedConfigurationResolution:
        cfg_name = configuration_name.strip()
        if not cfg_name:
            return NamedConfigurationResolution(
                success=False,
                message="configuration_name must not be empty.",
                trajectory_id="",
                trajectory=None,
            )

        trajectory = self._named_configurations.get(cfg_name)
        if trajectory is None:
            available = ", ".join(sorted(self._named_configurations.keys()))
            return NamedConfigurationResolution(
                success=False,
                message=(
                    f"Unknown named configuration '{cfg_name}'. "
                    f"Available: [{available}]"
                ),
                trajectory_id="",
                trajectory=None,
            )

        trajectory_id = make_named_trajectory_id(cfg_name)
        self._trajectories[trajectory_id] = StoredTrajectory(
            trajectory_id=trajectory_id,
            trajectory=trajectory,
            success=True,
            message=f"Named configuration trajectory '{cfg_name}'.",
            method="NAMED_CONFIGURATION",
            geometric_plan_id="",
        )
        return NamedConfigurationResolution(
            success=True,
            message=(
                f"Named configuration '{cfg_name}' converted to "
                f"trajectory_id={trajectory_id}."
            ),
            trajectory_id=trajectory_id,
            trajectory=trajectory,
        )

