"""iLQR trajectory optimizer for CBS crane.

Time-indexed LQR with double-integrator actuated joint dynamics.
Passive joint sway is computed as a post-processing forward pass.
"""

from .cost import TrackingCost
from .dynamics import DoubleIntegratorDynamics
from .reference import JointSplineReference
from .solver import ILQRResult, ILQRSolver

__all__ = [
    "DoubleIntegratorDynamics",
    "ILQRResult",
    "ILQRSolver",
    "JointSplineReference",
    "TrackingCost",
]
