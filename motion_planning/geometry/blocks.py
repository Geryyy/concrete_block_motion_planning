from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import fcl

from .utils import quat_to_rot

@dataclass
class Block:
    size: Tuple[float, float, float]          # (sx, sy, sz)
    position: Tuple[float, float, float]      # (x, y, z)
    quat: Tuple[float, float, float, float]   # (x, y, z, w)
    object_id: Optional[str] = None           # user or auto-assigned ID

    def fcl_object(self) -> fcl.CollisionObject:
        sx, sy, sz = self.size
        geom = fcl.Box(sx, sy, sz)
        R = quat_to_rot(self.quat)
        T = np.array(self.position, dtype=float)
        tf = fcl.Transform(R, T)
        return fcl.CollisionObject(geom, tf)

    def contains(self, p: np.ndarray) -> bool:
        """Check if point p (world) is inside the block."""
        R = quat_to_rot(self.quat)
        T = np.array(self.position, dtype=float)
        p_local = R.T @ (np.array(p, dtype=float) - T)
        hx, hy, hz = 0.5 * np.array(self.size, dtype=float)
        return (abs(p_local[0]) <= hx) and (abs(p_local[1]) <= hy) and (abs(p_local[2]) <= hz)

    def vertices_world(self):
        """8 vertices of the oriented box in world frame."""
        sx, sy, sz = self.size
        hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
        corners = np.array([
            [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
            [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz]
        ])
        R = quat_to_rot(self.quat)
        T = np.array(self.position, dtype=float)
        return (R @ corners.T).T + T