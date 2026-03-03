"""motion_planning — robotic crane motion planning library.

Three main entry points, from highest to lowest level:

**1. Full two-stage planning** (geometric path + trajectory optimisation)::

    from motion_planning import MotionPlanner, Scene

    scene = Scene()
    scene.add_block(size=(2, 2, 2), position=(5, 0, 1))

    planner = MotionPlanner(scene=scene)
    result  = planner.plan(q_start, q_goal)   # → MotionPlanResult

    result.plot()

**2. Geometric path planning only** (Cartesian, collision-free)::

    from motion_planning import plan_path, Scene

    scene = Scene()
    result = plan_path(
        start_xyz, goal_xyz,
        world_model=scene,
        moving_block_size=(1, 1, 1),
    )                                          # → PlannerResult
    path = result.path                         # BSplinePath

**3. Trajectory optimisation only** (no geometric stage)::

    from motion_planning import plan_trajectory, CartesianPathFollowingConfig

    cfg = CartesianPathFollowingConfig(urdf_path=..., horizon_steps=80)
    result = plan_trajectory(q_start, q_goal, traj_config=cfg)  # → TrajectoryResult

    # Joint-space alternative:
    from motion_planning import CranePathFollowingAcadosConfig
    cfg = CranePathFollowingAcadosConfig(urdf_path=...)
    result = plan_trajectory(q_start, q_goal, traj_config=cfg)
"""

# ── Two-stage planner (primary API) ──────────────────────────────────────────
from .planner import MotionPlanner, MotionPlanResult, Trajectory, plan_trajectory

# ── Standalone geometric path planning ───────────────────────────────────────
from .api import plan as plan_path

# Backward-compatible alias.
plan = plan_path

# ── Scene management ──────────────────────────────────────────────────────────
from .core.world_model import BlockState, WorldModel
from .geometry.scene import Scene as GeometryScene

# Alias: Scene refers to the geometry scene object used by the geometric planner.
Scene = GeometryScene

# ── Kinematics ────────────────────────────────────────────────────────────────
from .kinematics import CraneKinematics

# ── Trajectory optimizer configs and classes (for plan_trajectory / advanced use) ──
from .trajectory import (
    TrajectoryOptimizer,
    CranePathFollowingAcadosConfig,
    CranePathFollowingAcadosOptimizer,
    CartesianPathFollowingConfig,
    CartesianPathFollowingOptimizer,
)

# ── Control ───────────────────────────────────────────────────────────────────
from .control import ComputedTorqueController, PDController

# ── Core types ────────────────────────────────────────────────────────────────
from .core.spline import BSplinePath
from .core.types import PlannerRequest, PlannerResult, Scenario, TrajectoryRequest, TrajectoryResult

__all__ = [
    # ── Full two-stage planning ───────────────────────────────────────────────
    "MotionPlanner",
    "MotionPlanResult",
    "Trajectory",
    # ── Geometric path planning only ─────────────────────────────────────────
    "plan_path",
    "plan",   # backward-compatible alias for plan_path
    # ── Trajectory optimisation only ─────────────────────────────────────────
    "plan_trajectory",
    # ── Scene ─────────────────────────────────────────────────────────────────
    "Scene",
    "WorldModel",
    "BlockState",
    # ── Kinematics ────────────────────────────────────────────────────────────
    "CraneKinematics",
    # ── Trajectory optimizer configs (for plan_trajectory) ───────────────────
    "CartesianPathFollowingConfig",      # task-space FK cost — default
    "CranePathFollowingAcadosConfig",    # joint-space path-following
    # ── Trajectory optimizer classes (for direct use) ─────────────────────────
    "TrajectoryOptimizer",
    "CartesianPathFollowingOptimizer",
    "CranePathFollowingAcadosOptimizer",
    # ── Control ───────────────────────────────────────────────────────────────
    "PDController",
    "ComputedTorqueController",
    # ── Core types ────────────────────────────────────────────────────────────
    "BSplinePath",
    "Scenario",
    "PlannerRequest",
    "PlannerResult",
    "TrajectoryRequest",
    "TrajectoryResult",
]
