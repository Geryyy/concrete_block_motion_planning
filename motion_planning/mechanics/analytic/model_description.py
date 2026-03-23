"""Pinocchio model loader with structured inspection output.

Provides :class:`ModelDescription` which:
* loads the URDF via pinocchio,
* exposes joint / body / frame metadata needed by the symbolic layers,
* prints a human-readable summary of the model.

Also provides :func:`create_crane_config` — the authoritative programmatic
definition of the crane model config (source of truth for the YAML).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .config import AnalyticModelConfig

# ------------------------------------------------------------------ #
# Canonical crane config factory
# ------------------------------------------------------------------ #

_CRANE_URDF = Path(__file__).resolve().parents[3] / "crane_urdf" / "crane.urdf"


def create_crane_config() -> AnalyticModelConfig:
    """Return the canonical :class:`AnalyticModelConfig` for the crane model.

    This is the authoritative programmatic definition.  Run
    ``python motion_planning/mechanics/model_information.py`` to regenerate
    ``crane_config.yaml`` from this function.
    """
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
        passive_joints=[
            "theta6_tip_joint",
            "theta7_tilt_joint",
        ],
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


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #

def _se3_to_mat(se3) -> np.ndarray:
    return np.asarray(se3.homogeneous, dtype=float)


def _joint_type_str(j) -> str:
    """Return a short, human-readable joint type string."""
    sn: str = j.shortname()
    # pinocchio returns e.g. 'JointModelRZ' or just 'RZ' depending on version
    for token in ("RevoluteUnaligned", "PrismaticUnaligned",
                  "RX", "RY", "RZ", "PX", "PY", "PZ",
                  "FreeFlyer", "Spherical"):
        if token in sn:
            return token
    return sn


def _joint_motion_axis(j, model=None, jid: int = 0) -> np.ndarray:
    """Return the joint motion axis in the local joint frame (unit vector).

    For principal-axis joints (RX/RY/RZ/PX/PY/PZ) this is implicit in
    the joint type.  For unaligned joints the axis is stored on the joint
    object.
    """
    sn = _joint_type_str(j)
    _AXIS_MAP = {
        "RX": np.array([1.0, 0.0, 0.0]),
        "RY": np.array([0.0, 1.0, 0.0]),
        "RZ": np.array([0.0, 0.0, 1.0]),
        "PX": np.array([1.0, 0.0, 0.0]),
        "PY": np.array([0.0, 1.0, 0.0]),
        "PZ": np.array([0.0, 0.0, 1.0]),
    }
    if sn in _AXIS_MAP:
        return _AXIS_MAP[sn]
    # For unaligned joints pinocchio exposes .axis
    if hasattr(j, "axis"):
        axis = np.asarray(j.axis, dtype=float)
        norm = np.linalg.norm(axis)
        if norm > 1e-12:
            return axis / norm
    return np.array([0.0, 0.0, 1.0])


def _is_revolute(j) -> bool:
    sn = _joint_type_str(j)
    return "R" in sn and "P" not in sn and "Free" not in sn and "Spherical" not in sn


def _is_prismatic(j) -> bool:
    sn = _joint_type_str(j)
    return "P" in sn and "R" not in sn and "Free" not in sn


# ------------------------------------------------------------------ #
# Public class
# ------------------------------------------------------------------ #

class ModelDescription:
    """Wraps a pinocchio model and provides structured metadata.

    Parameters
    ----------
    config:
        :class:`~.config.AnalyticModelConfig` — supplies the URDF path.
    """

    def __init__(self, config: AnalyticModelConfig) -> None:
        import pinocchio as pin  # lazy import so the module is importable without pinocchio

        self._pin = pin
        self._cfg = config
        self.model = pin.buildModelFromUrdf(config.urdf_path)
        self.data = self.model.createData()

        # neutral configuration for reference
        self._q0 = np.asarray(pin.neutral(self.model), dtype=float)
        pin.forwardKinematics(self.model, self.data, self._q0)
        pin.updateFramePlacements(self.model, self.data)

    # ------------------------------------------------------------------ #
    # Public introspection API
    # ------------------------------------------------------------------ #

    def joint_info(self) -> List[Dict[str, Any]]:
        """Structured info for every non-universe joint."""
        model = self.model
        out: List[Dict[str, Any]] = []
        for jid in range(1, model.njoints):
            j = model.joints[jid]
            parent_jid = int(model.parents[jid])
            placement = model.jointPlacements[jid]
            inertia = model.inertias[jid]
            out.append({
                "id": jid,
                "name": str(model.names[jid]),
                "type": _joint_type_str(j),
                "is_revolute": _is_revolute(j),
                "is_prismatic": _is_prismatic(j),
                "motion_axis_local": _joint_motion_axis(j),
                "nq": int(j.nq),
                "nv": int(j.nv),
                "idx_q": int(j.idx_q),
                "parent_id": parent_jid,
                "parent_name": "universe" if parent_jid == 0 else str(model.names[parent_jid]),
                "placement_parent_to_joint": _se3_to_mat(placement),   # 4×4 numpy
                "world_to_joint_neutral": _se3_to_mat(self.data.oMi[jid]),
                "link_mass_kg": float(inertia.mass),
                "link_com_local": np.asarray(inertia.lever, dtype=float),
                "link_inertia_local": np.asarray(inertia.inertia, dtype=float),
            })
        return out

    def frame_info(self) -> List[Dict[str, Any]]:
        """Structured info for every pinocchio frame."""
        pin = self._pin
        model = self.model
        _FTYPE = {
            int(pin.FrameType.OP_FRAME): "OP_FRAME",
            int(pin.FrameType.JOINT): "JOINT",
            int(pin.FrameType.FIXED_JOINT): "FIXED_JOINT",
            int(pin.FrameType.BODY): "BODY",
            int(pin.FrameType.SENSOR): "SENSOR",
        }
        out: List[Dict[str, Any]] = []
        for fid, f in enumerate(model.frames):
            pjid = int(f.parentJoint)
            out.append({
                "id": fid,
                "name": str(f.name),
                "type": _FTYPE.get(int(f.type), str(int(f.type))),
                "parent_joint_id": pjid,
                "parent_joint_name": "universe" if pjid == 0 else str(model.names[pjid]),
                "placement_joint_to_frame": _se3_to_mat(f.placement),
                "world_to_frame_neutral": _se3_to_mat(self.data.oMf[fid]),
            })
        return out

    def joint_chain_ids(self, base_frame: str, end_frame: str) -> List[int]:
        """Return joint ids on the path from *base_frame* to *end_frame*.

        For the crane (serial chain) this is simply the subset of joints
        between the ancestor of *base_frame* and the ancestor of
        *end_frame*.
        """
        model = self.model

        base_jid = self._frame_parent_joint(base_frame)
        end_jid = self._frame_parent_joint(end_frame)

        # Walk from end up to world, collecting path
        path_to_world: List[int] = []
        cur = end_jid
        while cur != 0:
            path_to_world.append(cur)
            cur = int(model.parents[cur])
        path_to_world.reverse()  # world → end

        if base_jid == 0:
            return path_to_world

        # Trim path: keep only joints AFTER base_jid
        try:
            idx = path_to_world.index(base_jid)
        except ValueError as exc:
            raise ValueError(
                f"'{base_frame}' (joint {base_jid}) is not an ancestor of "
                f"'{end_frame}' (joint {end_jid})."
            ) from exc
        return path_to_world[idx + 1:]

    def get_link_inertia_data(self, joint_names: List[str]) -> List[Dict[str, Any]]:
        """Return mass / CoM / inertia tensor for each listed joint's link."""
        model = self.model
        out: List[Dict[str, Any]] = []
        for jname in joint_names:
            if not model.existJointName(jname):
                raise KeyError(f"Joint '{jname}' not found in model.")
            jid = int(model.getJointId(jname))
            inertia = model.inertias[jid]
            out.append({
                "joint_name": jname,
                "joint_id": jid,
                "mass": float(inertia.mass),
                "com_local": np.asarray(inertia.lever, dtype=float),
                "inertia_local": np.asarray(inertia.inertia, dtype=float),
            })
        return out

    # ------------------------------------------------------------------ #
    # Print helpers
    # ------------------------------------------------------------------ #

    def print_info(self) -> None:
        """Print a structured summary of bodies, joints, and frames."""
        model = self.model
        print("=" * 70)
        print(f"Model: {model.name}")
        print(f"  URDF: {self._cfg.urdf_path}")
        print(f"  nq={model.nq}  nv={model.nv}  njoints={model.njoints}  nframes={len(model.frames)}")

        # ---- Joints ----
        print("\nJoints (non-universe):")
        _W = 26
        header = (
            f"  {'id':>3}  {'name':<{_W}}  {'type':<22}  "
            f"{'axis (local)':<18}  {'parent':<{_W}}  mass [kg]"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for j in self.joint_info():
            axis_str = "[{:.0f} {:.0f} {:.0f}]".format(*j["motion_axis_local"])
            role = ""
            jname = j["name"]
            if jname in self._cfg.actuated_joints:
                role = " [ACT]"
            elif jname in self._cfg.passive_joints:
                role = " [PAS]"
            elif jname in self._cfg.locked_joints:
                role = " [LCK]"
            print(
                f"  {j['id']:>3}  {jname + role:<{_W+6}}  {j['type']:<22}  "
                f"{axis_str:<18}  {j['parent_name']:<{_W}}  {j['link_mass_kg']:.2f}"
            )

        # ---- Frames ----
        print("\nFrames:")
        header2 = f"  {'id':>4}  {'name':<38}  {'type':<14}  parent_joint"
        print(header2)
        print("  " + "-" * (len(header2) - 2))
        for f in self.frame_info():
            print(
                f"  {f['id']:>4}  {f['name']:<38}  {f['type']:<14}  {f['parent_joint_name']}"
            )
        print("=" * 70)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _frame_parent_joint(self, frame_name: str) -> int:
        """Return the parent joint id for a named frame (0 for 'world')."""
        if frame_name in ("world", "universe"):
            return 0
        model = self.model
        if not model.existFrame(frame_name):
            # Maybe it's a joint name — look for its joint frame
            if model.existJointName(frame_name):
                return int(model.getJointId(frame_name))
            raise KeyError(
                f"Frame '{frame_name}' not found. "
                f"Available frame names (first 20): {[f.name for f in model.frames[:20]]}"
            )
        fid = int(model.getFrameId(frame_name))
        return int(model.frames[fid].parentJoint)

    def frame_placement(self, frame_name: str) -> np.ndarray:
        """Return the 4×4 fixed SE3 placement from the parent joint to *frame_name*."""
        if frame_name in ("world", "universe"):
            return np.eye(4)
        model = self.model
        if not model.existFrame(frame_name):
            raise KeyError(f"Frame '{frame_name}' not found.")
        fid = int(model.getFrameId(frame_name))
        return _se3_to_mat(model.frames[fid].placement)
