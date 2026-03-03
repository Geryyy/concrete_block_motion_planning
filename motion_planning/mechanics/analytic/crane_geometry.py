from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CraneGeometryConstants:
    # Geometry constants used by the analytic IK.
    a1: float = 0.18
    d1: float = 2.425
    a2: float = 3.49288333
    a3: float = 0.3925
    d4: float = 3.157001602823
    # Operational upper bound on the arm joint (tighter than URDF limit).
    # Prevents overextension / self-collision in the workspace.
    theta3_max: float = 1.4

    @property
    def p2(self) -> np.ndarray:
        # 2D point in boom plane used in dependent-joint closed form.
        return np.array([-self.a1, self.d1], dtype=float)


DEFAULT_CRANE_GEOMETRY = CraneGeometryConstants()
