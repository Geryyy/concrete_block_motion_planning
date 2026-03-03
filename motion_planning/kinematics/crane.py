from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


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
    """Pinocchio-based FK utilities for the crane model."""

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

    def frame_names(self) -> List[str]:
        return [f.name for f in self.model.frames]

    def joint_names(self) -> List[str]:
        return [str(self.model.names[jid]) for jid in range(1, self.model.njoints)]

    def body_frame_names(self) -> List[str]:
        out: List[str] = []
        for f in self.model.frames:
            if int(f.type) == int(self.pin.FrameType.BODY):
                out.append(f.name)
        return out

    def _frame_id(self, name: str) -> int:
        if name == "world":
            return -1
        if not self.model.existFrame(name):
            raise KeyError(
                f"Frame '{name}' not found. Available frame names include: {self.frame_names()[:20]} ..."
            )
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

    def model_info(self, q: Optional[np.ndarray] = None) -> Dict[str, Any]:
        pin = self.pin
        if q is None:
            q = self.neutral()
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape[0] != self.model.nq:
            raise ValueError(f"q must have length nq={self.model.nq}, got {q.shape[0]}")
        self._update_kinematics(q)

        joints = []
        for jid in range(1, self.model.njoints):
            j = self.model.joints[jid]
            parent = int(self.model.parents[jid])
            joints.append(
                {
                    "id": int(jid),
                    "name": str(self.model.names[jid]),
                    "parent_joint_id": parent,
                    "parent_joint_name": "world" if parent == 0 else str(self.model.names[parent]),
                    "nq": int(j.nq),
                    "nv": int(j.nv),
                    "idx_q": int(j.idx_q),
                    "idx_v": int(j.idx_v),
                    "joint_placement_parent_to_joint": _se3_to_dict(self.model.jointPlacements[jid]),
                    "world_to_joint": _se3_to_dict(self.data.oMi[jid]),
                }
            )

        frames = []
        frame_type_names = {
            int(pin.FrameType.OP_FRAME): "OP_FRAME",
            int(pin.FrameType.JOINT): "JOINT",
            int(pin.FrameType.FIXED_JOINT): "FIXED_JOINT",
            int(pin.FrameType.BODY): "BODY",
            int(pin.FrameType.SENSOR): "SENSOR",
        }
        for fid, f in enumerate(self.model.frames):
            frames.append(
                {
                    "id": int(fid),
                    "name": str(f.name),
                    "type": frame_type_names.get(int(f.type), str(int(f.type))),
                    "parent_joint_id": int(f.parentJoint),
                    "parent_joint_name": "world"
                    if int(f.parentJoint) == 0
                    else str(self.model.names[int(f.parentJoint)]),
                    "placement_parent_joint_to_frame": _se3_to_dict(f.placement),
                    "world_to_frame": _se3_to_dict(self.data.oMf[fid]),
                }
            )

        return {
            "urdf_path": str(self.urdf_path),
            "nq": int(self.model.nq),
            "nv": int(self.model.nv),
            "njoints": int(self.model.njoints),
            "nframes": len(self.model.frames),
            "joints": joints,
            "frames": frames,
            "joint_names": self.joint_names(),
            "frame_names": self.frame_names(),
            "body_frame_names": self.body_frame_names(),
        }

    def print_model_info(self, q: Optional[np.ndarray] = None) -> None:
        info = self.model_info(q=q)
        print(f"URDF: {info['urdf_path']}")
        print(f"nq={info['nq']} nv={info['nv']} njoints={info['njoints']} nframes={info['nframes']}")
        print("Joints:")
        for j in info["joints"]:
            print(
                f"  - [{j['id']:>2}] {j['name']:<24} parent={j['parent_joint_name']:<16} "
                f"(nq={j['nq']}, nv={j['nv']}, idx_q={j['idx_q']}, idx_v={j['idx_v']})"
            )
        print("Frames:")
        for f in info["frames"]:
            print(
                f"  - [{f['id']:>3}] {f['name']:<32} type={f['type']:<12} "
                f"parent_joint={f['parent_joint_name']}"
            )
