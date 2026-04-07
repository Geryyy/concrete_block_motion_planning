from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .config import AnalyticModelConfig


_CRANE_URDF = Path(__file__).resolve().parents[1] / "data" / "crane.urdf"
_AXES = {
    "RX": np.array([1.0, 0.0, 0.0]),
    "RY": np.array([0.0, 1.0, 0.0]),
    "RZ": np.array([0.0, 0.0, 1.0]),
    "PX": np.array([1.0, 0.0, 0.0]),
    "PY": np.array([0.0, 1.0, 0.0]),
    "PZ": np.array([0.0, 0.0, 1.0]),
}
_FRAME_TYPES = {
    "OP_FRAME": "OP_FRAME",
    "JOINT": "JOINT",
    "FIXED_JOINT": "FIXED_JOINT",
    "BODY": "BODY",
    "SENSOR": "SENSOR",
}


def create_crane_config() -> AnalyticModelConfig:
    return AnalyticModelConfig(
        urdf_path=str(_CRANE_URDF),
        gravity=[0.0, 0.0, -9.81],
        base_frame="K0_mounting_base",
        target_frame="K8_tool_center_point",
        actuated_joints=[
            "theta1_slewing_joint",
            "theta2_boom_joint",
            "theta3_arm_joint",
            "q4_big_telescope",
            "theta8_rotator_joint",
        ],
        passive_joints=["theta6_tip_joint", "theta7_tilt_joint"],
        dynamic_joints=[
            "theta1_slewing_joint",
            "theta2_boom_joint",
            "theta3_arm_joint",
            "q4_big_telescope",
            "theta6_tip_joint",
            "theta7_tilt_joint",
            "theta8_rotator_joint",
        ],
        tied_joints={"q5_small_telescope": "q4_big_telescope"},
        joint_position_overrides={"theta3_arm_joint": (None, float(np.pi / 2.0))},
        locked_joints=[
            "truck_pitch",
            "truck_roll",
            "q9_left_rail_joint",
            "q11_right_rail_joint",
        ],
    )


def _se3_to_mat(se3) -> np.ndarray:
    return np.asarray(se3.homogeneous, dtype=float)


def _joint_type(j) -> str:
    shortname = j.shortname()
    for token in ("RevoluteUnaligned", "PrismaticUnaligned", *_AXES, "FreeFlyer", "Spherical"):
        if token in shortname:
            return token
    return shortname


def _joint_axis(j) -> np.ndarray:
    joint_type = _joint_type(j)
    if joint_type in _AXES:
        return _AXES[joint_type]
    axis = np.asarray(getattr(j, "axis", [0.0, 0.0, 1.0]), dtype=float)
    norm = float(np.linalg.norm(axis))
    return axis / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])


class ModelDescription:
    def __init__(self, config: AnalyticModelConfig) -> None:
        import pinocchio as pin

        self._pin = pin
        self._cfg = config
        self.model = pin.buildModelFromUrdf(config.urdf_path)
        self.data = self.model.createData()
        q0 = np.asarray(pin.neutral(self.model), dtype=float)
        pin.forwardKinematics(self.model, self.data, q0)
        pin.updateFramePlacements(self.model, self.data)

    def joint_info(self) -> List[Dict[str, Any]]:
        model = self.model
        return [
            {
                "id": jid,
                "name": str(model.names[jid]),
                "type": _joint_type(joint),
                "is_revolute": _joint_type(joint).startswith("R") or _joint_type(joint) == "RevoluteUnaligned",
                "is_prismatic": _joint_type(joint).startswith("P") or _joint_type(joint) == "PrismaticUnaligned",
                "motion_axis_local": _joint_axis(joint),
                "nq": int(joint.nq),
                "nv": int(joint.nv),
                "idx_q": int(joint.idx_q),
                "parent_id": parent_id,
                "parent_name": "universe" if parent_id == 0 else str(model.names[parent_id]),
                "placement_parent_to_joint": _se3_to_mat(model.jointPlacements[jid]),
                "world_to_joint_neutral": _se3_to_mat(self.data.oMi[jid]),
                "link_mass_kg": float(model.inertias[jid].mass),
                "link_com_local": np.asarray(model.inertias[jid].lever, dtype=float),
                "link_inertia_local": np.asarray(model.inertias[jid].inertia, dtype=float),
            }
            for jid in range(1, model.njoints)
            for joint in [model.joints[jid]]
            for parent_id in [int(model.parents[jid])]
        ]

    def frame_info(self) -> List[Dict[str, Any]]:
        frame_types = {
            int(getattr(self._pin.FrameType, name)): value for name, value in _FRAME_TYPES.items()
        }
        return [
            {
                "id": fid,
                "name": str(frame.name),
                "type": frame_types.get(int(frame.type), str(int(frame.type))),
                "parent_joint_id": parent_id,
                "parent_joint_name": "universe" if parent_id == 0 else str(self.model.names[parent_id]),
                "placement_joint_to_frame": _se3_to_mat(frame.placement),
                "world_to_frame_neutral": _se3_to_mat(self.data.oMf[fid]),
            }
            for fid, frame in enumerate(self.model.frames)
            for parent_id in [int(frame.parentJoint)]
        ]

    def joint_chain_ids(self, base_frame: str, end_frame: str) -> List[int]:
        base_id = self._frame_parent_joint(base_frame)
        chain = []
        cur = self._frame_parent_joint(end_frame)
        while cur:
            chain.append(cur)
            cur = int(self.model.parents[cur])
        chain.reverse()
        if base_id == 0:
            return chain
        if base_id not in chain:
            raise ValueError(f"'{base_frame}' is not an ancestor of '{end_frame}'.")
        return chain[chain.index(base_id) + 1 :]

    def get_link_inertia_data(self, joint_names: List[str]) -> List[Dict[str, Any]]:
        model = self.model
        missing = [name for name in joint_names if not model.existJointName(name)]
        if missing:
            raise KeyError(f"Joint '{missing[0]}' not found in model.")
        return [
            {
                "joint_name": name,
                "joint_id": jid,
                "mass": float(model.inertias[jid].mass),
                "com_local": np.asarray(model.inertias[jid].lever, dtype=float),
                "inertia_local": np.asarray(model.inertias[jid].inertia, dtype=float),
            }
            for name in joint_names
            for jid in [int(model.getJointId(name))]
        ]

    def print_info(self) -> None:
        model = self.model
        print("=" * 70)
        print(f"Model: {model.name}")
        print(f"  URDF: {self._cfg.urdf_path}")
        print(f"  nq={model.nq}  nv={model.nv}  njoints={model.njoints}  nframes={len(model.frames)}")
        print("\nJoints (non-universe):")
        width = 26
        header = f"  {'id':>3}  {'name':<{width}}  {'type':<22}  {'axis (local)':<18}  {'parent':<{width}}  mass [kg]"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for joint in self.joint_info():
            name = joint["name"]
            if name in self._cfg.actuated_joints:
                name += " [ACT]"
            elif name in self._cfg.passive_joints:
                name += " [PAS]"
            elif name in self._cfg.locked_joints:
                name += " [LCK]"
            axis = "[{:.0f} {:.0f} {:.0f}]".format(*joint["motion_axis_local"])
            print(
                f"  {joint['id']:>3}  {name:<{width + 6}}  {joint['type']:<22}  "
                f"{axis:<18}  {joint['parent_name']:<{width}}  {joint['link_mass_kg']:.2f}"
            )
        print("\nFrames:")
        header = f"  {'id':>4}  {'name':<38}  {'type':<14}  parent_joint"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for frame in self.frame_info():
            print(f"  {frame['id']:>4}  {frame['name']:<38}  {frame['type']:<14}  {frame['parent_joint_name']}")
        print("=" * 70)

    def _frame_parent_joint(self, frame_name: str) -> int:
        if frame_name in {"world", "universe"}:
            return 0
        if self.model.existFrame(frame_name):
            return int(self.model.frames[int(self.model.getFrameId(frame_name))].parentJoint)
        if self.model.existJointName(frame_name):
            return int(self.model.getJointId(frame_name))
        raise KeyError(f"Frame '{frame_name}' not found.")

    def frame_placement(self, frame_name: str) -> np.ndarray:
        if frame_name in {"world", "universe"}:
            return np.eye(4)
        if not self.model.existFrame(frame_name):
            raise KeyError(f"Frame '{frame_name}' not found.")
        return _se3_to_mat(self.model.frames[int(self.model.getFrameId(frame_name))].placement)
