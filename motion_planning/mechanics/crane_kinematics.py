from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from .pinocchio_utils import q_map_to_pin_q
from .pose_conventions import phi_tool_from_transform


def _import_pinocchio():
    import pinocchio as pin

    return pin


def _se3_to_dict(se3) -> Dict[str, Any]:
    mat = se3.homogeneous
    return {
        "translation": np.asarray(se3.translation, dtype=float).copy(),
        "rotation": np.asarray(se3.rotation, dtype=float).copy(),
        "homogeneous": np.asarray(mat, dtype=float).copy(),
    }


class CraneKinematics:
    def __init__(self, urdf_path: str | Path):
        pin = _import_pinocchio()
        self.pin = pin
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()

    @property
    def nq(self) -> int:
        return int(self.model.nq)

    @property
    def nv(self) -> int:
        return int(self.model.nv)

    def neutral(self) -> np.ndarray:
        return np.asarray(self.pin.neutral(self.model), dtype=float).copy()

    def q_from_map(self, q_values: Mapping[str, float]) -> np.ndarray:
        return q_map_to_pin_q(self.model, q_values, self.pin)

    def _frame_id(self, name: str) -> int:
        if name == "world":
            return -1
        if not self.model.existFrame(name):
            raise KeyError(f"Frame '{name}' not found.")
        return int(self.model.getFrameId(name))

    def _update_kinematics(self, q: np.ndarray) -> None:
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)

    def _world_to_frame(self, frame_id: int):
        if frame_id < 0:
            return self.pin.SE3.Identity()
        return self.data.oMf[frame_id]

    def forward_kinematics(
        self,
        q: np.ndarray,
        *,
        base_frame: str = "world",
        end_frame: str,
    ) -> Dict[str, Any]:
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape[0] != self.model.nq:
            raise ValueError(f"q must have length nq={self.model.nq}, got {q.shape[0]}")

        self._update_kinematics(q)
        base_id = self._frame_id(base_frame)
        end_id = self._frame_id(end_frame)
        oMb = self._world_to_frame(base_id)
        oMe = self._world_to_frame(end_id)
        bMe = oMb.inverse() * oMe
        return {
            "base_frame": base_frame,
            "end_frame": end_frame,
            "base_to_end": _se3_to_dict(bMe),
            "world_to_base": _se3_to_dict(oMb),
            "world_to_end": _se3_to_dict(oMe),
        }

    def transform_from_joint_map(
        self,
        q_values: Mapping[str, float],
        *,
        base_frame: str = "world",
        end_frame: str,
    ) -> np.ndarray:
        fk = self.forward_kinematics(self.q_from_map(q_values), base_frame=base_frame, end_frame=end_frame)
        return np.asarray(fk["base_to_end"]["homogeneous"], dtype=float)

    def pose_from_joint_map(
        self,
        q_values: Mapping[str, float],
        *,
        base_frame: str = "world",
        end_frame: str,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        T = self.transform_from_joint_map(q_values, base_frame=base_frame, end_frame=end_frame)
        return np.asarray(T[:3, 3], dtype=float).reshape(3), phi_tool_from_transform(T), T
