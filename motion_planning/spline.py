from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class BSplinePath:
    """Minimal path container shared across planning stages."""

    xyz_fn: Callable[[np.ndarray], np.ndarray]
    yaw_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def sample(self, n: int = 101) -> np.ndarray:
        u = np.linspace(0.0, 1.0, int(n))
        return np.asarray(self.xyz_fn(u), dtype=float)

    def sample_yaw(self, n: int = 101) -> np.ndarray:
        if self.yaw_fn is None:
            return np.zeros(int(n), dtype=float)
        u = np.linspace(0.0, 1.0, int(n))
        return np.asarray(self.yaw_fn(u), dtype=float).reshape(-1)
