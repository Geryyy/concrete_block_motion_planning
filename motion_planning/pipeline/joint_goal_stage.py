from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable, Mapping, Sequence

import numpy as np

from motion_planning.kinematics.crane import CraneKinematics
from motion_planning.mechanics.analytic import (
    CraneSteadyState,
    ModelDescription,
    create_crane_config,
)


@dataclass
class JointGoalSolveResult:
    success: bool
    message: str
    offset_m: float
    base_frame: str
    target_frame: str
    goal_world: np.ndarray
    goal_base: np.ndarray
    target_yaw_rad: float
    q_actuated: dict[str, float]
    q_passive: dict[str, float]
    q_dynamic: dict[str, float]
    passive_residual: float
    fk_position_error_m: float
    fk_yaw_error_rad: float
    fk_xyz_base: np.ndarray
    fk_xyz_world: np.ndarray
    fk_yaw_rad: float
    ik_backend: str


class JointGoalStage:
    """Standalone joint-goal stage for concrete commissioning.

    This layer intentionally solves only static, fully actuated-style goal states:
    current world/task pose -> equilibrium crane configuration.
    """

    def __init__(self) -> None:
        self._cfg = create_crane_config()
        self._cfg.urdf_path = str(self._resolve_existing_urdf_path(self._cfg.urdf_path))
        self._desc = ModelDescription(self._cfg)
        self._steady_state = CraneSteadyState(self._desc, self._cfg)
        self._kin = CraneKinematics(self._cfg.urdf_path)

        q_neutral = np.zeros(self._kin.model.nq, dtype=float)
        fk_base = self._kin.forward_kinematics(
            q_neutral,
            base_frame="world",
            end_frame=self._cfg.base_frame,
        )
        self._t_world_base = np.asarray(
            fk_base["base_to_end"]["homogeneous"], dtype=float
        )
        self._t_base_world = np.linalg.inv(self._t_world_base)

    @staticmethod
    def _resolve_existing_urdf_path(initial_path: str) -> Path:
        p = Path(initial_path).expanduser()
        if p.exists():
            return p.resolve()

        repo_root = Path(__file__).resolve()
        for parent in [Path.cwd(), repo_root.parent, *repo_root.parents]:
            cand = parent / "crane_urdf" / "crane.urdf"
            if cand.exists():
                return cand.resolve()
            xacro_cand = (
                parent / "src" / "epsilon_crane_description" / "urdf" / "crane.urdf.xacro"
            )
            if xacro_cand.exists():
                urdf_path = Path(tempfile.gettempdir()) / "joint_goal_stage_crane.urdf"
                proc = subprocess.run(
                    ["xacro", str(xacro_cand)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                urdf_path.write_text(proc.stdout, encoding="utf-8")
                return urdf_path.resolve()

        raise FileNotFoundError(
            f"Cannot locate crane URDF. Tried configured path '{initial_path}' "
            "and local crane_urdf/crane.urdf candidates."
        )

    @property
    def config(self):
        return self._cfg

    def world_to_base(self, p_world: Sequence[float]) -> np.ndarray:
        p = np.asarray(p_world, dtype=float).reshape(3)
        ph = np.concatenate([p, [1.0]], dtype=float)
        return (self._t_base_world @ ph)[:3]

    @staticmethod
    def generate_linear_preapproach_targets(
        start_world: Sequence[float],
        target_world: Sequence[float],
        offsets_m: Iterable[float],
    ) -> list[tuple[float, np.ndarray]]:
        start = np.asarray(start_world, dtype=float).reshape(3)
        target = np.asarray(target_world, dtype=float).reshape(3)
        direction = target - start
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            return [(0.0, target.copy())]
        direction /= norm
        out: list[tuple[float, np.ndarray]] = []
        for offset in offsets_m:
            off = max(0.0, float(offset))
            out.append((off, target - off * direction))
        return out

    def solve_world_pose(
        self,
        *,
        goal_world: Sequence[float],
        target_yaw_rad: float,
        offset_m: float = 0.0,
        q_seed: Mapping[str, float] | None = None,
    ) -> JointGoalSolveResult:
        goal_world_arr = np.asarray(goal_world, dtype=float).reshape(3)
        goal_base = self.world_to_base(goal_world_arr)
        ss = self._steady_state.compute(
            target_pos=goal_base,
            target_yaw=float(target_yaw_rad),
            q_seed=q_seed,
        )
        fk_xyz_base = np.asarray(ss.fk_xyz, dtype=float)
        fk_xyz_world = (self._t_world_base @ np.concatenate([fk_xyz_base, [1.0]], dtype=float))[:3]
        message = str(ss.message)
        if not ss.success:
            message = (
                f"{message} "
                f"[frames: world->{self._cfg.base_frame}->{self._cfg.target_frame}; "
                f"requested_world={np.array2string(goal_world_arr, precision=4, suppress_small=False)}, "
                f"requested_base={np.array2string(goal_base, precision=4, suppress_small=False)}, "
                f"requested_yaw={float(target_yaw_rad):.4f}rad; "
                f"fk_world={np.array2string(fk_xyz_world, precision=4, suppress_small=False)}, "
                f"fk_base={np.array2string(fk_xyz_base, precision=4, suppress_small=False)}, "
                f"fk_yaw={float(ss.fk_yaw_rad):.4f}rad; "
                f"ik_status={ss.ik_result.status}]"
            )
        return JointGoalSolveResult(
            success=bool(ss.success),
            message=message,
            offset_m=float(offset_m),
            base_frame=str(self._cfg.base_frame),
            target_frame=str(self._cfg.target_frame),
            goal_world=goal_world_arr,
            goal_base=np.asarray(goal_base, dtype=float),
            target_yaw_rad=float(target_yaw_rad),
            q_actuated=dict(ss.q_actuated),
            q_passive=dict(ss.q_passive),
            q_dynamic=dict(ss.q_dynamic),
            passive_residual=float(ss.passive_residual),
            fk_position_error_m=float(ss.fk_position_error_m),
            fk_yaw_error_rad=float(ss.fk_yaw_error_rad),
            fk_xyz_base=fk_xyz_base,
            fk_xyz_world=fk_xyz_world,
            fk_yaw_rad=float(ss.fk_yaw_rad),
            ik_backend=str(ss.ik_result.status),
        )

    def solve_preapproach_family(
        self,
        *,
        start_world: Sequence[float],
        target_world: Sequence[float],
        target_yaw_rad: float,
        offsets_m: Iterable[float],
        q_seed: Mapping[str, float] | None = None,
    ) -> list[JointGoalSolveResult]:
        out: list[JointGoalSolveResult] = []
        for offset_m, goal_world in self.generate_linear_preapproach_targets(
            start_world, target_world, offsets_m
        ):
            out.append(
                self.solve_world_pose(
                    goal_world=goal_world,
                    target_yaw_rad=target_yaw_rad,
                    offset_m=offset_m,
                    q_seed=q_seed,
                )
            )
        return out
