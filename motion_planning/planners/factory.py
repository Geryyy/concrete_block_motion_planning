from __future__ import annotations

from motion_planning.planners.base import Planner
from motion_planning.planners.spline import SplineOptimizerPlanner

_METHODS = {
    "POWELL": "Powell",
    "NELDER-MEAD": "Nelder-Mead",
    "NELDER_MEAD": "Nelder-Mead",
    "NELDERMEAD": "Nelder-Mead",
    "NELDER": "Nelder-Mead",
    "NM": "Nelder-Mead",
    "CEM": "CEM",
}


def create_planner(method: str) -> Planner:
    key = str(method).strip().upper()
    if key not in _METHODS:
        raise ValueError(
            f"Unsupported planner '{method}'. Supported: {', '.join(sorted(set(_METHODS.values())))}"
        )
    return SplineOptimizerPlanner(method=_METHODS[key])
