from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Tuple


JointNames = Tuple[str, ...]


@dataclass(frozen=True)
class JointMapping:
    """Relationship between a backend full-state contract and command subset."""

    full_state_joint_names: JointNames
    command_joint_names: JointNames
    passive_joint_names: JointNames = ()
    mimic_joint_map: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        full_set = set(self.full_state_joint_names)
        command_set = set(self.command_joint_names)
        passive_set = set(self.passive_joint_names)
        mimic_values = set(self.mimic_joint_map.values())

        if len(full_set) != len(self.full_state_joint_names):
            raise ValueError("full_state_joint_names must be unique")
        if len(command_set) != len(self.command_joint_names):
            raise ValueError("command_joint_names must be unique")
        if len(passive_set) != len(self.passive_joint_names):
            raise ValueError("passive_joint_names must be unique")
        if not command_set.issubset(full_set):
            unknown = sorted(command_set - full_set)
            raise ValueError(f"command_joint_names must be subset of full_state_joint_names: {unknown}")
        if not passive_set.issubset(full_set):
            unknown = sorted(passive_set - full_set)
            raise ValueError(f"passive_joint_names must be subset of full_state_joint_names: {unknown}")
        if command_set & passive_set:
            overlap = sorted(command_set & passive_set)
            raise ValueError(f"command_joint_names and passive_joint_names must be disjoint: {overlap}")
        if not mimic_values.issubset(full_set):
            unknown = sorted(mimic_values - full_set)
            raise ValueError(
                "mimic joint values must be in full_state_joint_names: "
                f"{unknown}"
            )

    def indices_for(self, joint_names: JointNames) -> Tuple[int, ...]:
        by_name = {name: idx for idx, name in enumerate(self.full_state_joint_names)}
        return tuple(by_name[name] for name in joint_names)

    @property
    def command_indices(self) -> Tuple[int, ...]:
        return self.indices_for(self.command_joint_names)

    @property
    def passive_indices(self) -> Tuple[int, ...]:
        return self.indices_for(self.passive_joint_names)


@dataclass(frozen=True)
class RobotProfile:
    """Backend-specific joint naming/profile information for the shared contract."""

    name: str
    mapping: JointMapping
    tool_joint_names: JointNames = ()
    compat_aliases: Mapping[str, str] = field(default_factory=dict)
    notes: str = ""

    @property
    def full_state_joint_names(self) -> JointNames:
        return self.mapping.full_state_joint_names

    @property
    def command_joint_names(self) -> JointNames:
        return self.mapping.command_joint_names

    @property
    def passive_joint_names(self) -> JointNames:
        return self.mapping.passive_joint_names

    @property
    def mimic_joint_map(self) -> Mapping[str, str]:
        return self.mapping.mimic_joint_map
