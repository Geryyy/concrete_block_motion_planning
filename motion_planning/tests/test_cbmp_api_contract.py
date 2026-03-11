from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cbmp.ids import make_geometric_plan_id, make_named_trajectory_id, make_trajectory_id
from cbmp.node import ConcreteBlockMotionPlanningNode


def test_id_generation_prefixes_and_shape() -> None:
    geo = make_geometric_plan_id()
    traj = make_trajectory_id()
    named = make_named_trajectory_id("Home Pose")

    assert geo.startswith("geo_")
    assert traj.startswith("traj_")
    assert named.startswith("named_Home_Pose_")

    assert len(geo.split("_", maxsplit=1)[1]) == 8
    assert len(traj.split("_", maxsplit=1)[1]) == 8
    assert len(named.rsplit("_", maxsplit=1)[1]) == 8


def test_register_services_exposes_only_clean_api_endpoints() -> None:
    class _Registrar:
        def __init__(self) -> None:
            self.paths: list[str] = []

        def create_service(self, _srv_type, path: str, _handler):
            self.paths.append(path)
            return path

        def _handle_plan_geometric(self):
            pass

        def _handle_compute_trajectory(self):
            pass

        def _handle_plan_and_compute_trajectory(self):
            pass

        def _handle_execute_trajectory(self):
            pass

        def _handle_execute_named_configuration(self):
            pass

        def _handle_get_next_assembly_task(self):
            pass

    registrar = _Registrar()
    ConcreteBlockMotionPlanningNode._register_services(registrar)

    assert registrar.paths == [
        "~/plan_geometric_path",
        "~/plan_and_compute_trajectory",
        "~/compute_trajectory",
        "~/execute_trajectory",
        "~/execute_named_configuration",
        "~/get_next_assembly_task",
    ]
