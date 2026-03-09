from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Scenario:
    """Canonical geometric planning input independent of concrete loaders."""

    scene: Any
    start: Tuple[float, float, float]
    goal: Tuple[float, float, float]
    moving_block_size: Tuple[float, float, float]
    start_yaw_deg: float = 0.0
    goal_yaw_deg: float = 0.0
    goal_normals: Tuple[Tuple[float, float, float], ...] = ()


@dataclass(frozen=True)
class PlannerRequest:
    scenario: Scenario
    config: Mapping[str, Any]
    options: Mapping[str, Any]


@dataclass
class PlannerResult:
    success: bool
    message: str
    path: "BSplinePath"
    metrics: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryRequest:
    """Input for a single trajectory-optimization solve.

    ``scenario`` and ``path`` may be ``None`` when the optimizer is called
    standalone (without a geometric planning stage).  Optimizers that need
    Cartesian control points check ``req.path is not None`` explicitly.

    Standard ``config`` keys accepted by all trajectory optimizers:
        ``q0``       – start joint config (full-model or reduced, nv,)
        ``q_goal``   – goal  joint config (full-model or reduced, nv,)
        ``dq0``      – initial joint velocities (default: zeros)

    Additional key for ``CartesianPathFollowingOptimizer``:
        ``ctrl_pts_xyz`` – explicit Cartesian control points (n_ctrl, 3);
                           overrides ``path.sample()`` when provided.
    """

    scenario: Optional[Scenario]
    path: Optional["BSplinePath"]
    config: Mapping[str, Any]


@dataclass
class TrajectoryResult:
    success: bool
    message: str
    time_s: np.ndarray
    state: np.ndarray
    control: np.ndarray
    cost: Optional[float] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WallPlacement:
    """Single resolved wall placement target."""

    block_id: str
    reference_block_id: Optional[str]
    absolute_position: Tuple[float, float, float]
    relative_offset: Tuple[float, float, float]
    yaw_deg: float
    size: Tuple[float, float, float]


@dataclass(frozen=True)
class WallPlan:
    """Ordered wall assembly plan."""

    name: str
    placements: Tuple[WallPlacement, ...]


# Late import type hints only.
from .spline import BSplinePath  # noqa: E402
