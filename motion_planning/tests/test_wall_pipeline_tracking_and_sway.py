from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from motion_planning import Scene
from motion_planning.types import Scenario, TrajectoryRequest
from motion_planning.mechanics import create_crane_config
from motion_planning import run_geometric_planning
from motion_planning.scenarios import ScenarioLibrary
from motion_planning_tools.simulation.mujoco_pd_replay import replay_trajectory_with_pd
from motion_planning.trajectory.cartesian_path_following import (
    CartesianPathFollowingConfig,
    CartesianPathFollowingOptimizer,
)
from motion_planning.trajectory.planning_limits import load_planning_limits_yaml


class _ReplayTrajectory:
    def __init__(
        self,
        time_s: np.ndarray,
        q: np.ndarray,
        dq: np.ndarray,
        joint_names: list[str],
        actuated_joint_names: list[str],
    ) -> None:
        self.time_s = time_s
        self.q = q
        self.dq = dq
        self.joint_names = joint_names
        self.actuated_joint_names = actuated_joint_names


def _runtime_ready() -> bool:
    try:
        import acados_template  # noqa: F401
        import casadi  # noqa: F401
        import mujoco  # noqa: F401
        import pinocchio  # noqa: F401
    except Exception:
        return False
    src = os.environ.get("ACADOS_SOURCE_DIR", "")
    if not src:
        return False
    src_path = Path(src).expanduser().resolve()
    return (src_path / "lib" / "link_libs.json").exists() and (src_path / "bin" / "t_renderer").exists()


def _ensure_acados_runtime_env() -> None:
    src = Path(os.environ["ACADOS_SOURCE_DIR"]).expanduser().resolve()
    lib_dir = str(src / "lib")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if not ld:
        os.environ["LD_LIBRARY_PATH"] = lib_dir
    elif lib_dir not in ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld}"


def _build_wall_scene_from_generated(
    *,
    wall_steps: int,
    wall_offset_xyz: np.ndarray,
) -> tuple[Scene, list[dict[str, tuple[float, float, float]]]]:
    lib = ScenarioLibrary()
    names = sorted([n for n in lib.list_scenarios() if n.startswith("step_")])[: max(1, int(wall_steps))]

    scene = Scene()
    overlay_blocks: list[dict[str, tuple[float, float, float]]] = []
    seen: set[tuple[float, ...]] = set()
    for name in names:
        sc = lib.build_scenario(name)
        for b in sc.scene.blocks:
            if str(b.object_id).lower() == "table":
                continue
            pos = np.asarray(b.position, dtype=float) + wall_offset_xyz
            size = tuple(float(v) for v in b.size)
            key = tuple(np.round(np.concatenate((pos, np.asarray(size, dtype=float))), 6).tolist())
            if key in seen:
                continue
            seen.add(key)
            scene.add_block(size=size, position=tuple(float(v) for v in pos), object_id=f"{name}_{b.object_id}")
            overlay_blocks.append({"position": tuple(float(v) for v in pos), "size": size})
    return scene, overlay_blocks


def _point_to_polyline_distance(p: np.ndarray, poly: np.ndarray) -> float:
    best = float("inf")
    for i in range(poly.shape[0] - 1):
        a = poly[i]
        b = poly[i + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-12:
            d = float(np.linalg.norm(p - a))
        else:
            t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
            proj = a + t * ab
            d = float(np.linalg.norm(p - proj))
        if d < best:
            best = d
    return best


@pytest.mark.skipif(not _runtime_ready(), reason="acados/mujoco/pinocchio runtime not available")
def test_wall_pipeline_path_tracking_and_gripper_sway() -> None:
    _ensure_acados_runtime_env()

    repo_root = Path(__file__).resolve().parents[2]
    acfg = create_crane_config()
    limits_yaml = repo_root / "motion_planning" / "trajectory" / "planning_limits.yaml"
    _, taskspace_limits = load_planning_limits_yaml(limits_yaml)

    q_start = np.array([1.265, 0.291, 1.069, 0.165, 0.165, 0.211, 1.571, 1.36], dtype=float)
    q_goal = np.array([-0.372, 0.47, 0.99, 0.253, 0.253, 0.11, 1.571, 1.461], dtype=float)
    p_start = np.array([-8.891, -5.842, 2.460], dtype=float)
    p_goal = np.array([-13.023, 2.725, 3.362], dtype=float)

    scene, overlay_blocks = _build_wall_scene_from_generated(
        wall_steps=4,
        wall_offset_xyz=np.asarray([-11.2, -1.2, 2.2], dtype=float),
    )
    scenario = Scenario(
        scene=scene.scene,
        start=tuple(p_start.tolist()),
        goal=tuple(p_goal.tolist()),
        moving_block_size=(1.0, 1.0, 1.0),
    )

    # Keep this test reasonably fast while preserving wall-scene behavior.
    geo_result = run_geometric_planning(
        scenario=scenario,
        method="POWELL",
        config={"n_vias": 2, "w_curv": 1.5, "w_len": 1.0},
        options={"maxiter": 120},
    )
    assert geo_result.success, geo_result.message

    path_world = np.asarray(geo_result.path.sample(220), dtype=float)
    p_min = np.asarray(taskspace_limits[0], dtype=float).reshape(3)
    p_max = np.asarray(taskspace_limits[1], dtype=float).reshape(3)
    assert np.all(path_world >= p_min - 1e-2)
    assert np.all(path_world <= p_max + 1e-2)

    cfg = CartesianPathFollowingConfig(
        urdf_path=Path(acfg.urdf_path),
        horizon_steps=160,
        sdot_ref=0.05,
        sdot_max=0.35,
        xyz_weight=280.0,
        terminal_xyz_weight=900.0,
        s_weight=10.0,
        terminal_s_weight=120.0,
        passive_q_sway_weight=20.0,
        passive_dq_sway_weight=180.0,
        terminal_passive_q_sway_weight=180.0,
        terminal_passive_dq_sway_weight=700.0,
        passive_dq_soft_max=0.08,
        passive_dq_slack_weight=3000.0,
        terminal_passive_dq_slack_weight=8000.0,
        terminal_hold_steps=20,
        code_export_dir=Path("/tmp/test_wall_pipeline_codegen"),
        solver_json_name="test_wall_pipeline_ocp.json",
    )
    optimizer = CartesianPathFollowingOptimizer(cfg)

    req_cfg = {
        "q0": q_start,
        "q_goal": q_goal,
        "dq0": np.zeros_like(q_start),
        "joint_position_limits_yaml": str(limits_yaml),
        "T_max": 60.0,
        "time_weight": 5e-5,
        "qp_solver_iter_max": 220,
        "nlp_solver_max_iter": 2500,
        "qp_tol": 5e-7,
        "nlp_tol": 5e-6,
    }
    planner_ctrl = geo_result.diagnostics.get("ctrl_pts_xyz", None)
    if planner_ctrl is not None:
        planner_ctrl = np.asarray(planner_ctrl, dtype=float).reshape(-1, 3)
        req_cfg["ctrl_pts_xyz"] = planner_ctrl
        req_cfg["spline_ctrl_points"] = int(planner_ctrl.shape[0])
        if "spline_degree" in geo_result.diagnostics:
            req_cfg["spline_degree"] = int(geo_result.diagnostics["spline_degree"])
    else:
        req_cfg["path_ref_points"] = 10

    traj_result = optimizer.optimize(TrajectoryRequest(scenario=scenario, path=geo_result.path, config=req_cfg))
    assert traj_result.success, traj_result.message

    xyz_traj = np.asarray(traj_result.diagnostics["xyz_trajectory"], dtype=float)
    d_path = np.asarray([_point_to_polyline_distance(p, path_world) for p in xyz_traj], dtype=float)
    # Explicit path-tracking checks.
    assert float(np.median(d_path)) < 0.45, f"path median error too high: {np.median(d_path):.4f} m"
    assert float(np.quantile(d_path, 0.95)) < 0.90, f"path p95 error too high: {np.quantile(d_path, 0.95):.4f} m"

    sim = replay_trajectory_with_pd(
        _ReplayTrajectory(
            time_s=np.asarray(traj_result.time_s, dtype=float),
            q=np.asarray(traj_result.diagnostics["q_trajectory"], dtype=float),
            dq=np.asarray(traj_result.diagnostics["dq_trajectory"], dtype=float),
            joint_names=list(traj_result.diagnostics["reduced_joint_names"]),
            actuated_joint_names=list(traj_result.diagnostics["actuated_joint_names"]),
        ),
        view=False,
        mujoco_model=repo_root / "crane_urdf" / "crane.xml",
        kp=55.0,
        kd=25.0,
        speed=1.0,
        tail_s=6.0,
        overlay_blocks=overlay_blocks,
        monitor_tcp_site="K8_tool_center_point_tcp",
    )
    assert sim is not None
    assert sim.tcp_world is not None and sim.tcp_world.shape[0] == sim.time_s.shape[0]

    horizon = float(traj_result.time_s[-1])
    tail_mask = sim.time_s >= (horizon + 0.5)
    assert bool(np.any(tail_mask)), "tail window missing for sway check"
    tail_tcp = np.asarray(sim.tcp_world[tail_mask], dtype=float)
    tail_ref = tail_tcp[-1]
    tail_dev = np.linalg.norm(tail_tcp - tail_ref, axis=1)
    # Explicit residual sway check during post-horizon hold.
    assert float(np.quantile(tail_dev, 0.95)) < 0.35, (
        f"residual sway too high (tail p95): {np.quantile(tail_dev, 0.95):.4f} m"
    )
