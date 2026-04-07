from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from motion_planning.mechanics import CraneKinematics
from motion_planning.mechanics import (
    CraneSteadyState,
    ModelDescription,
    create_crane_config,
    resolve_existing_urdf_path,
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


def _as_xyz(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(3)


class JointGoalStage:
    def __init__(self) -> None:
        self._cfg = create_crane_config()
        self._cfg.urdf_path = resolve_existing_urdf_path(self._cfg.urdf_path)
        self._desc = ModelDescription(self._cfg)
        self._steady_state = CraneSteadyState(self._desc, self._cfg)
        self._kin = CraneKinematics(self._cfg.urdf_path)
        self._t_world_base = np.asarray(
            self._kin.forward_kinematics(
                self._kin.neutral(),
                base_frame="world",
                end_frame=self._cfg.base_frame,
            )["base_to_end"]["homogeneous"],
            dtype=float,
        )
        self._t_base_world = np.linalg.inv(self._t_world_base)

    @property
    def config(self):
        return self._cfg

    def world_to_base(self, p_world: Sequence[float]) -> np.ndarray:
        return (self._t_base_world @ np.r_[_as_xyz(p_world), 1.0])[:3]

    @staticmethod
    def generate_linear_preapproach_targets(
        start_world: Sequence[float],
        target_world: Sequence[float],
        offsets_m: Iterable[float],
    ) -> list[tuple[float, np.ndarray]]:
        start = _as_xyz(start_world)
        target = _as_xyz(target_world)
        direction = target - start
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            return [(0.0, target.copy())]
        direction /= norm
        return [(max(0.0, float(offset)), target - max(0.0, float(offset)) * direction) for offset in offsets_m]

    def _result_message(
        self,
        *,
        success: bool,
        message: str,
        goal_world: np.ndarray,
        goal_base: np.ndarray,
        target_yaw_rad: float,
        fk_xyz_world: np.ndarray,
        fk_xyz_base: np.ndarray,
        fk_yaw_rad: float,
        ik_status: str,
    ) -> str:
        if success:
            return message
        return (
            f"{message} [frames: world->{self._cfg.base_frame}->{self._cfg.target_frame}; "
            f"requested_world={np.array2string(goal_world, precision=4, suppress_small=False)}, "
            f"requested_base={np.array2string(goal_base, precision=4, suppress_small=False)}, "
            f"requested_yaw={float(target_yaw_rad):.4f}rad; "
            f"fk_world={np.array2string(fk_xyz_world, precision=4, suppress_small=False)}, "
            f"fk_base={np.array2string(fk_xyz_base, precision=4, suppress_small=False)}, "
            f"fk_yaw={float(fk_yaw_rad):.4f}rad; ik_status={ik_status}]"
        )

    def solve_world_pose(
        self,
        *,
        goal_world: Sequence[float],
        target_yaw_rad: float,
        offset_m: float = 0.0,
        q_seed: Mapping[str, float] | None = None,
    ) -> JointGoalSolveResult:
        goal_world = _as_xyz(goal_world)
        goal_base = self.world_to_base(goal_world)
        ss = self._steady_state.compute(target_pos=goal_base, target_yaw=float(target_yaw_rad), q_seed=q_seed)
        fk_xyz_base = np.asarray(ss.fk_xyz, dtype=float)
        fk_xyz_world = (self._t_world_base @ np.r_[fk_xyz_base, 1.0])[:3]
        return JointGoalSolveResult(
            success=bool(ss.success),
            message=self._result_message(
                success=bool(ss.success),
                message=str(ss.message),
                goal_world=goal_world,
                goal_base=goal_base,
                target_yaw_rad=float(target_yaw_rad),
                fk_xyz_world=fk_xyz_world,
                fk_xyz_base=fk_xyz_base,
                fk_yaw_rad=float(ss.fk_yaw_rad),
                ik_status=str(ss.ik_result.status),
            ),
            offset_m=float(offset_m),
            base_frame=str(self._cfg.base_frame),
            target_frame=str(self._cfg.target_frame),
            goal_world=goal_world,
            goal_base=goal_base,
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
        return [
            self.solve_world_pose(
                goal_world=goal_world,
                target_yaw_rad=target_yaw_rad,
                offset_m=offset_m,
                q_seed=q_seed,
            )
            for offset_m, goal_world in self.generate_linear_preapproach_targets(
                start_world,
                target_world,
                offsets_m,
            )
        ]
