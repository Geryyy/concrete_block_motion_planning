"""motion_planning public API.

Stable entrypoints are intentionally small:
- `api.plan(...)` for geometric planning
- `planner.MotionPlanner` / `plan_trajectory(...)` for two-stage compatibility use
- core scenario/result types and scene/world-model helpers

Subsystems such as standalone experiments, solver internals, and trajectory
optimizers should be imported from their own modules when needed.
"""

__all__ = []

# ── Stable planning entrypoints ───────────────────────────────────────────────
try:
    from .planner import MotionPlanner, MotionPlanResult, Trajectory, plan_trajectory

    __all__ += ["MotionPlanner", "MotionPlanResult", "Trajectory", "plan_trajectory"]
except Exception:  # pragma: no cover - optional dependency surface
    MotionPlanner = None
    MotionPlanResult = None
    Trajectory = None
    plan_trajectory = None

try:
    from .api import plan as plan_path

    plan = plan_path
    __all__ += ["plan_path", "plan"]
except Exception:  # pragma: no cover - optional dependency surface
    plan_path = None
    plan = None

# ── Core data surfaces ────────────────────────────────────────────────────────
try:
    from .core.world_model import BlockState, WorldModel
    from .geometry.scene import Scene as GeometryScene
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

    Scene = GeometryScene
    __all__ += [
        "Scene",
        "WorldModel",
        "BlockState",
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
    BlockState = None
    WorldModel = None
    GeometryScene = None
    Scene = None
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

# ── Convenience loaders and widely used helpers ──────────────────────────────
try:
    from .kinematics import CraneKinematics
    from .control import ComputedTorqueController, PDController
    from .scenarios import WallPlanLibrary, build_wall_plan, list_wall_plans

    __all__ += [
        "CraneKinematics",
        "PDController",
        "ComputedTorqueController",
        "WallPlanLibrary",
        "build_wall_plan",
        "list_wall_plans",
    ]
except Exception:  # pragma: no cover - optional dependency surface
    CraneKinematics = None
    ComputedTorqueController = None
    PDController = None
    WallPlanLibrary = None
    build_wall_plan = None
    list_wall_plans = None
