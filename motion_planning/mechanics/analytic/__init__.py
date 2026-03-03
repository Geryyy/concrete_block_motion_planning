"""Analytic mechanics module (kinematics + dynamics).

Quick-start
-----------
>>> from motion_planning.mechanics.analytic import (
...     AnalyticModelConfig,
...     ModelDescription,
...     AnalyticInverseKinematics,
...     CraneSteadyState,
... )
"""

from .config import AnalyticModelConfig
from .inverse_kinematics import AnalyticIKSolver, AnalyticInverseKinematics, IkSolveResult, NumericIKSolver
from .model_description import ModelDescription, create_crane_config
from .projected_dynamics import PassiveAccelResult, ProjectedUnderactuatedDynamics
from .split_dynamics import SplitPassiveAccelResult, SplitUnderactuatedDynamics
from .steady_state import CraneSteadyState, SteadyStateResult

__all__ = [
    "AnalyticModelConfig",
    "AnalyticIKSolver",
    "AnalyticInverseKinematics",
    "NumericIKSolver",
    "ModelDescription",
    "create_crane_config",
    "IkSolveResult",
    "ProjectedUnderactuatedDynamics",
    "PassiveAccelResult",
    "SplitUnderactuatedDynamics",
    "SplitPassiveAccelResult",
    "CraneSteadyState",
    "SteadyStateResult",
]
