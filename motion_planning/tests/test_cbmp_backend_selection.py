from __future__ import annotations

import sys
from pathlib import Path

import pytest
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cbmp.backends.base import PlannerBackend
from cbmp.compatibility import A2BCompatibilityRequest
from cbmp.named_configurations import NamedConfigurationResolver
from cbmp.node import ConcreteBlockMotionPlanningNode
from cbmp.results import A2BCompatibilityResult
from cbmp.types import StoredTrajectory
from concrete_block_motion_planning.srv import PlanAndComputeTrajectory


class _FakeBackend(PlannerBackend):
    @property
    def backend_name(self) -> str:
        return "fake"

    @property
    def capabilities(self):
        from cbmp.results import PlannerCapabilities

        return PlannerCapabilities(True, True, False, False, False)

    def plan_move_empty(self, **kwargs):
        from cbmp.results import BackendPlanResult

        traj = JointTrajectory()
        traj.joint_names = ["theta1_slewing_joint"]
        point = JointTrajectoryPoint()
        point.positions = [0.0]
        point.velocities = [0.0]
        traj.points = [point]
        return BackendPlanResult(
            success=True,
            message="fake backend success",
            trajectory=traj,
            cartesian_path=kwargs["planning_context"].get("cartesian_path")
            if "cartesian_path" in kwargs["planning_context"]
            else type("DummyPath", (), {"poses": []})(),
            geometric_plan_id="fake_geo",
        )

    def plan_a2b_compat(
        self,
        *,
        request: A2BCompatibilityRequest,
    ) -> A2BCompatibilityResult:
        del request
        return A2BCompatibilityResult(
            success=False,
            message="fake compatibility path",
            trajectory=JointTrajectory(),
            tcp_path=[],
        )


def test_config_supports_planner_backend_parameters() -> None:
    from cbmp.config import declare_and_load_config

    class _Param:
        def __init__(self, value):
            self.value = value

    class _Node:
        def __init__(self):
            self._params = {
                "planner.backend": "timber",
                "planner.timber_a2b_service": "/a2b_movement",
                "planner.timber_grip_service": "/grip_traj_movement",
                "planner.timber_goal_frame": "K0_mounting_base",
                "planner.timber_move_empty_target_z": 2.5,
                "planner.timber_payload_density_kg_m3": 2500.0,
                "planner.timber_payload_grippoint_xyz": [0.1, 0.2, 0.3],
            }

        def declare_parameter(self, name, default):
            if name not in self._params:
                self._params[name] = default
            return _Param(self._params[name])

        def get_parameter(self, name):
            return _Param(self._params[name])

    cfg = declare_and_load_config(_Node())
    assert cfg.planner_backend == "timber"
    assert cfg.timber_a2b_service == "/a2b_movement"
    assert cfg.timber_grip_service == "/grip_traj_movement"
    assert cfg.timber_goal_frame == "K0_mounting_base"
    assert cfg.timber_move_empty_target_z == pytest.approx(2.5)
    assert cfg.timber_payload_density_kg_m3 == pytest.approx(2500.0)
    assert cfg.timber_payload_grippoint_xyz == pytest.approx((0.1, 0.2, 0.3))


def test_named_configuration_resolver_registers_stored_trajectory() -> None:
    named = {}
    stored: dict[str, StoredTrajectory] = {}
    traj = JointTrajectory()
    traj.joint_names = ["joint_1"]
    point = JointTrajectoryPoint()
    point.positions = [0.0]
    point.velocities = [0.0]
    traj.points = [point]
    named["home"] = traj

    resolver = NamedConfigurationResolver(
        named_configurations=named,
        trajectories=stored,
    )
    result = resolver.resolve("home")
    assert result.success is True
    assert result.trajectory is traj
    assert result.trajectory_id in stored
