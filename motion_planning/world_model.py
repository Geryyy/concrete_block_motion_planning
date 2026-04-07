from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

from motion_planning.geometry.scene import Scene

IdLike = Union[int, str]


@dataclass(frozen=True)
class BlockState:
    object_id: str | None
    size: Tuple[float, float, float]
    position: Tuple[float, float, float]
    quat: Tuple[float, float, float, float]


class WorldModel:
    """Mutable world model for planner queries (ROS-friendly surface)."""

    def __init__(self, scene: Scene | None = None) -> None:
        self._scene = scene if scene is not None else Scene()

    @classmethod
    def from_scene(cls, scene: Scene) -> "WorldModel":
        """Wrap an existing geometry scene in a WorldModel facade."""
        return cls(scene=scene)

    @property
    def scene(self) -> Scene:
        return self._scene

    def reset(self) -> None:
        self._scene = Scene()

    def add_block(
        self,
        *,
        size: Tuple[float, float, float],
        position: Tuple[float, float, float],
        quat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        object_id: str | None = None,
    ) -> str:
        return self._scene.add_block(size=size, position=position, quat=quat, object_id=object_id)

    def query_block(self, id_or_index: IdLike) -> BlockState:
        block = self._scene.get_block(id_or_index)
        return BlockState(
            object_id=block.object_id,
            size=tuple(float(v) for v in block.size),
            position=tuple(float(v) for v in block.position),
            quat=tuple(float(v) for v in block.quat),
        )

    def update_block(
        self,
        id_or_index: IdLike,
        *,
        position: Tuple[float, float, float] | None = None,
        quat: Tuple[float, float, float, float] | None = None,
        size: Tuple[float, float, float] | None = None,
    ) -> BlockState:
        """Update position, orientation, or size of an existing obstacle."""
        block = self._scene.update_block(id_or_index, position=position, quat=quat, size=size)
        return BlockState(
            object_id=block.object_id,
            size=tuple(float(v) for v in block.size),
            position=tuple(float(v) for v in block.position),
            quat=tuple(float(v) for v in block.quat),
        )

    def remove_block(self, id_or_index: IdLike) -> None:
        """Remove an obstacle from the scene."""
        self._scene.remove_block(id_or_index)
