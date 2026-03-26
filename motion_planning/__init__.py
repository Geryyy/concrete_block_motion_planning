"""motion_planning — robotic crane motion planning library.

This package supports partial runtime availability:
- Geometry/world-model features require `fcl`.
- Trajectory optimization requires `acados_template`, `casadi`, `pinocchio`.

Imports below are guarded so lightweight or trajectory-only use remains possible
when geometric dependencies are not installed.
"""

__all__ = []

# ── Two-stage planner (primary API) ──────────────────────────────────────────
try:
    from .planner import MotionPlanner, MotionPlanResult, Trajectory, plan_trajectory

    __all__ += ["MotionPlanner", "MotionPlanResult", "Trajectory", "plan_trajectory"]
except Exception:  # pragma: no cover - optional dependency surface
    MotionPlanner = None
    MotionPlanResult = None
    Trajectory = None
    plan_trajectory = None

# ── Standalone geometric path planning ───────────────────────────────────────
try:
    from .api import plan as plan_path

    plan = plan_path
    __all__ += ["plan_path", "plan"]
except Exception:  # pragma: no cover - optional dependency surface
    plan_path = None
    plan = None

# ── Scene management ──────────────────────────────────────────────────────────
try:
    from .core.world_model import BlockState, WorldModel
    from .geometry.scene import Scene as GeometryScene

    Scene = GeometryScene
    __all__ += ["Scene", "WorldModel", "BlockState"]
except Exception:  # pragma: no cover - optional dependency surface
    BlockState = None
    WorldModel = None
    GeometryScene = None
    Scene = None

# ── Kinematics ────────────────────────────────────────────────────────────────
try:
    from .kinematics import CraneKinematics

    __all__ += ["CraneKinematics"]
except Exception:  # pragma: no cover - optional dependency surface
    CraneKinematics = None

# ── Trajectory optimizer configs/classes ─────────────────────────────────────
try:
    from .trajectory import (
        TrajectoryOptimizer,
        CranePathFollowingAcadosConfig,
        CranePathFollowingAcadosOptimizer,
        CartesianPathFollowingConfig,
        CartesianPathFollowingOptimizer,
    )

    __all__ += [
        "CartesianPathFollowingConfig",
        "CranePathFollowingAcadosConfig",
        "TrajectoryOptimizer",
        "CartesianPathFollowingOptimizer",
        "CranePathFollowingAcadosOptimizer",
    ]
except Exception:  # pragma: no cover - optional dependency surface
    TrajectoryOptimizer = None
    CranePathFollowingAcadosConfig = None
    CranePathFollowingAcadosOptimizer = None
    CartesianPathFollowingConfig = None
    CartesianPathFollowingOptimizer = None

# ── Control ───────────────────────────────────────────────────────────────────
try:
    from .control import ComputedTorqueController, PDController

    __all__ += ["PDController", "ComputedTorqueController"]
except Exception:  # pragma: no cover - optional dependency surface
    ComputedTorqueController = None
    PDController = None

# ── Core types ────────────────────────────────────────────────────────────────
try:
    from .core.spline import BSplinePath
    from .contracts import JointMapping, RobotProfile, TrajectoryContract
    from .core.types import (
        PlannerRequest,
        PlannerResult,
        Scenario,
        TrajectoryRequest,
        TrajectoryResult,
        WallPlacement,
        WallPlan,
    )

    __all__ += [
        "BSplinePath",
        "Scenario",
        "PlannerRequest",
        "PlannerResult",
        "TrajectoryRequest",
        "TrajectoryResult",
        "WallPlacement",
        "WallPlan",
        "JointMapping",
        "RobotProfile",
        "TrajectoryContract",
    ]
except Exception:  # pragma: no cover - optional dependency surface
    BSplinePath = None
    JointMapping = None
    PlannerRequest = None
    PlannerResult = None
    RobotProfile = None
    Scenario = None
    TrajectoryContract = None
    TrajectoryRequest = None
    TrajectoryResult = None
    WallPlacement = None
    WallPlan = None

# ── Wall planning data loader ────────────────────────────────────────────────
try:
    from .scenarios import WallPlanLibrary, build_wall_plan, list_wall_plans

    __all__ += ["WallPlanLibrary", "build_wall_plan", "list_wall_plans"]
except Exception:  # pragma: no cover - optional dependency surface
    WallPlanLibrary = None
    build_wall_plan = None
    list_wall_plans = None
