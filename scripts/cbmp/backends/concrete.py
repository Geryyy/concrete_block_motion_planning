from __future__ import annotations

from geometry_msgs.msg import PoseStamped

from concrete_block_motion_planning.srv import ComputeTrajectory

from ..compatibility import A2BCompatibilityRequest, make_empty_compat_trajectory
from ..ids import make_geometric_plan_id
from ..results import A2BCompatibilityResult, BackendPlanResult, PlannerCapabilities
from .base import PlannerBackend


class ConcretePlannerBackend(PlannerBackend):
    def __init__(self, node) -> None:
        self._node = node

    @property
    def backend_name(self) -> str:
        return "concrete"

    @property
    def capabilities(self) -> PlannerCapabilities:
        return PlannerCapabilities(
            supports_move_empty=True,
            supports_named_configurations=True,
            supports_world_model_obstacles=True,
            supports_pick_place=True,
            supports_geometric_stage=True,
        )

    def plan_move_empty(
        self,
        *,
        start_pose: PoseStamped,
        goal_pose: PoseStamped,
        geometric_method: str,
        geometric_timeout_s: float,
        trajectory_method: str,
        trajectory_timeout_s: float,
        validate_dynamics: bool,
        planning_context: dict[str, object],
    ) -> BackendPlanResult:
        del geometric_timeout_s  # current concrete implementation does not use this timeout directly
        geometric_plan_id = make_geometric_plan_id()
        plan = self._node._build_geometric_plan(
            start_pose,
            goal_pose,
            geometric_method,
            planning_context=planning_context,
        )
        plan.geometric_plan_id = geometric_plan_id
        self._node._geometric_plans[geometric_plan_id] = plan
        if plan.success and plan.path is not None:
            self._node._planned_path_pub.publish(plan.path)
        if not plan.success:
            return BackendPlanResult(
                success=False,
                message=f"Geometric planning failed: {plan.message}",
                trajectory=self._node._empty_trajectory(),
                cartesian_path=plan.path,
                geometric_plan_id=geometric_plan_id,
            )

        traj_req = ComputeTrajectory.Request()
        traj_req.geometric_plan_id = geometric_plan_id
        traj_req.method = trajectory_method
        traj_req.timeout_s = trajectory_timeout_s
        traj_req.validate_dynamics = validate_dynamics
        traj_res = self._node._handle_compute_trajectory(
            traj_req,
            ComputeTrajectory.Response(),
        )
        actual_path = plan.path
        if traj_res.success and traj_res.trajectory_id:
            stored = self._node._trajectories.get(traj_res.trajectory_id)
            if stored is not None and stored.cartesian_path.poses:
                actual_path = stored.cartesian_path
        return BackendPlanResult(
            success=bool(traj_res.success),
            message=(
                "Combined plan+compute success. "
                f"{plan.message} | {traj_res.message}"
                if traj_res.success
                else "Trajectory stage failed after geometric success. "
                f"{plan.message} | {traj_res.message}"
            ),
            trajectory=traj_res.trajectory,
            cartesian_path=actual_path,
            geometric_plan_id=geometric_plan_id,
        )

    def plan_a2b_compat(
        self,
        *,
        request: A2BCompatibilityRequest,
    ) -> A2BCompatibilityResult:
        del request
        return A2BCompatibilityResult(
            success=False,
            message=(
                "CBS a2b compatibility path is wired, but the concrete planner "
                "implementation is not available yet."
            ),
            trajectory=make_empty_compat_trajectory(),
            tcp_path=[],
        )
