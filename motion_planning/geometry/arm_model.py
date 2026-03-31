"""hppfcl collision model for the CBS crane arm with a PZS100 proxy tool."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import hppfcl as _hppfcl

    _HPPFCL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HPPFCL_AVAILABLE = False

from .scene import Scene


@dataclass(frozen=True)
class _CapsuleDef:
    p1_frame: str
    p2_frame: str
    radius: float


@dataclass(frozen=True)
class CapsuleSegment:
    name: str
    p1_frame: str
    p2_frame: str
    radius: float
    p1: np.ndarray
    p2: np.ndarray


@dataclass
class ArmPathClearanceReport:
    arm_min_clearance_m: float
    payload_min_clearance_m: float
    combined_min_clearance_m: float
    collision_free: bool
    arm_worst_index: int
    payload_worst_index: int

    @property
    def message(self) -> str:
        tag = "clear" if self.collision_free else "COLLISION"
        return (
            f"{tag}: arm={self.arm_min_clearance_m*100:.1f} cm, "
            f"payload={self.payload_min_clearance_m*100:.1f} cm"
        )


_CAPSULES: tuple[_CapsuleDef, ...] = (
    _CapsuleDef("K0_mounting_base", "K1_slewing_column", 0.15),
    _CapsuleDef("K1_slewing_column", "K2_boom", 0.14),
    _CapsuleDef("K2_boom", "K3_arm", 0.10),
    _CapsuleDef("K3_arm", "K5_inner_telescope", 0.08),
    _CapsuleDef("K1_boom_cylinder_suspension", "boom_cylinder_piston", 0.06),
    _CapsuleDef("K5_inner_telescope", "K6_double_joint_link", 0.05),
    _CapsuleDef("K6_double_joint_link", "K8_tool_center_point", 0.08),
    _CapsuleDef("K11", "K9", 0.06),
    _CapsuleDef("K9", "K10_left_rail", 0.08),
    _CapsuleDef("K11", "K12_right_rail", 0.08),
)

DEFAULT_PZS100_RAIL_POSITION_M: float = 0.538
_DEG = np.pi / 180.0
_PSEUDO_FRAME_OFFSETS: dict[str, tuple[str, np.ndarray, np.ndarray]] = {
    "K10_left_rail": (
        "K9",
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([-np.pi / 2.0, 3.0 * _DEG, 0.0], dtype=float),
    ),
    "K12_right_rail": (
        "K11",
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([-np.pi / 2.0, 3.0 * _DEG, 0.0], dtype=float),
    ),
}


def _rotation_z_to(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    d = d / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0], dtype=float)
    axis = np.cross(z, d)
    sin_a = float(np.linalg.norm(axis))
    cos_a = float(np.dot(z, d))
    if sin_a < 1e-8:
        return np.eye(3) if cos_a > 0.0 else np.diag([1.0, -1.0, -1.0])
    axis = axis / sin_a
    k_mat = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=float,
    )
    return np.eye(3) + sin_a * k_mat + (1.0 - cos_a) * (k_mat @ k_mat)


def _rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def _make_capsule_shape_tf(p1: np.ndarray, p2: np.ndarray, radius: float):
    if not _HPPFCL_AVAILABLE:
        raise ImportError("hppfcl is required for arm collision checking")
    d = p2 - p1
    length = float(np.linalg.norm(d))
    if length < 1e-6:
        return None
    rot = _rotation_z_to(d / length)
    shape = _hppfcl.Capsule(float(radius), float(length))
    tf = _hppfcl.Transform3f()
    tf.setRotation(rot)
    tf.setTranslation((0.5 * (p1 + p2)).astype(float))
    return shape, tf


def _capsule_min_dist(
    cap_shape,
    cap_tf,
    scene: Scene,
    ignore_ids: Optional[List[str]],
) -> float:
    ignore = set(ignore_ids or [])
    req = _hppfcl.DistanceRequest()
    min_dist = np.inf
    for blk in scene.blocks:
        if blk.object_id is not None and blk.object_id in ignore:
            continue
        static_shape, static_tf = blk.hppfcl_shape_tf()
        res = _hppfcl.DistanceResult()
        d = float(_hppfcl.distance(cap_shape, cap_tf, static_shape, static_tf, req, res))
        min_dist = min(min_dist, d)
    return float(min_dist)


class CraneArmCollisionModel:
    """hppfcl collision model for the full CBS crane arm with a fixed PZS100 tool."""

    def __init__(self) -> None:
        self._pin = None
        self._model = None
        self._data = None
        self._act_names: list[str] = []
        self._frame_cache: dict[str, int] = {}
        self._steady_state = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import pinocchio as pin
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pinocchio is required for CraneArmCollisionModel") from exc
        from motion_planning.pipeline import JointGoalStage

        stage = JointGoalStage()
        self._pin = pin
        self._act_names = list(stage.config.actuated_joints)
        self._model = pin.buildModelFromUrdf(stage.config.urdf_path)
        self._data = self._model.createData()
        self._steady_state = stage._steady_state

    def _ensure_steady_state(self):
        self._ensure_loaded()
        return self._steady_state

    def complete_joint_map(
        self,
        q_act: np.ndarray,
        q_passive: Optional[Dict[str, float]] = None,
        rail_position: float = DEFAULT_PZS100_RAIL_POSITION_M,
    ) -> Dict[str, float]:
        self._ensure_loaded()
        q_map: Dict[str, float] = {name: float(q_act[i]) for i, name in enumerate(self._act_names)}
        if q_passive is not None:
            theta6 = float(q_passive.get("theta6_tip_joint", 0.0))
            theta7 = float(q_passive.get("theta7_tilt_joint", np.pi / 2))
        else:
            ss = self._ensure_steady_state()
            theta6, theta7 = ss.analytic_equilibrium(
                q_map["theta2_boom_joint"],
                q_map["theta3_arm_joint"],
                q_map["theta8_rotator_joint"],
            )

        q_map["theta6_tip_joint"] = theta6
        q_map["theta7_tilt_joint"] = theta7
        q_map["q9_left_rail_joint"] = float(rail_position)
        q_map["q11_right_rail_joint"] = float(rail_position)
        return q_map

    def _frame_pos(self, q_map: Dict[str, float], frame_name: str) -> np.ndarray:
        return np.asarray(self._frame_tf(q_map, frame_name)[:3, 3], dtype=float)

    def _frame_tf(self, q_map: Dict[str, float], frame_name: str) -> np.ndarray:
        from motion_planning.mechanics.analytic.pinocchio_utils import fk_homogeneous

        try:
            return fk_homogeneous(
                pin_model=self._model,
                pin_data=self._data,
                pin_module=self._pin,
                q_values=q_map,
                base_frame="K0_mounting_base",
                end_frame=frame_name,
                frame_cache=self._frame_cache,
            )
        except KeyError:
            pseudo = _PSEUDO_FRAME_OFFSETS.get(frame_name)
            if pseudo is None:
                raise
            parent_frame, xyz, rpy = pseudo
            parent_tf = fk_homogeneous(
                pin_model=self._model,
                pin_data=self._data,
                pin_module=self._pin,
                q_values=q_map,
                base_frame="K0_mounting_base",
                end_frame=parent_frame,
                frame_cache=self._frame_cache,
            )
            h_local = np.eye(4, dtype=float)
            h_local[:3, :3] = _rpy_to_rot(float(rpy[0]), float(rpy[1]), float(rpy[2]))
            h_local[:3, 3] = np.asarray(xyz, dtype=float)
            return np.asarray(parent_tf @ h_local, dtype=float)

    def capsule_segments(self, q_map: Dict[str, float]) -> list[CapsuleSegment]:
        self._ensure_loaded()
        return [
            CapsuleSegment(
                name=f"{cap.p1_frame}->{cap.p2_frame}",
                p1_frame=cap.p1_frame,
                p2_frame=cap.p2_frame,
                radius=float(cap.radius),
                p1=self._frame_pos(q_map, cap.p1_frame),
                p2=self._frame_pos(q_map, cap.p2_frame),
            )
            for cap in _CAPSULES
        ]

    def clearance(
        self,
        q_map: Dict[str, float],
        scene: Scene,
        ignore_ids: Optional[List[str]] = None,
    ) -> float:
        if not scene.blocks:
            return np.inf
        self._ensure_loaded()
        min_dist = np.inf
        for cap in self.capsule_segments(q_map):
            result = _make_capsule_shape_tf(cap.p1, cap.p2, cap.radius)
            if result is None:
                continue
            min_dist = min(min_dist, _capsule_min_dist(result[0], result[1], scene, ignore_ids))
        return float(min_dist)

    def payload_clearance(
        self,
        tcp_xyz: np.ndarray,
        tcp_yaw: float,
        block_size: Sequence[float],
        scene: Scene,
        ignore_ids: Optional[List[str]] = None,
    ) -> float:
        half = float(tcp_yaw) * 0.5
        quat = (0.0, 0.0, float(np.sin(half)), float(np.cos(half)))
        return scene.signed_distance_block(
            size=block_size,
            position=np.asarray(tcp_xyz, dtype=float),
            quat=quat,
            ignore_ids=ignore_ids,
        )

    def check_path(
        self,
        q_maps: List[Dict[str, float]],
        tcp_xyz_path: np.ndarray,
        tcp_yaw_path: np.ndarray,
        moving_block_size: Sequence[float],
        scene: Scene,
        arm_ignore_ids: Optional[List[str]] = None,
        payload_ignore_ids: Optional[List[str]] = None,
        safety_margin: float = 0.01,
    ) -> ArmPathClearanceReport:
        if arm_ignore_ids is None:
            arm_ignore_ids = ["table"]

        xyz_arr = np.asarray(tcp_xyz_path, dtype=float)
        yaw_arr = np.asarray(tcp_yaw_path, dtype=float)
        n_pts = len(q_maps)
        arm_dists = np.full(n_pts, np.inf, dtype=float)
        payload_dists = np.full(n_pts, np.inf, dtype=float)
        for i, q_map in enumerate(q_maps):
            arm_dists[i] = self.clearance(q_map, scene, ignore_ids=arm_ignore_ids)
            payload_dists[i] = self.payload_clearance(
                xyz_arr[i],
                float(yaw_arr[i]),
                moving_block_size,
                scene,
                ignore_ids=payload_ignore_ids,
            )

        arm_idx = int(np.argmin(arm_dists))
        payload_idx = int(np.argmin(payload_dists))
        arm_min = float(arm_dists[arm_idx])
        payload_min = float(payload_dists[payload_idx])
        return ArmPathClearanceReport(
            arm_min_clearance_m=arm_min,
            payload_min_clearance_m=payload_min,
            combined_min_clearance_m=float(min(arm_min, payload_min)),
            collision_free=float(min(arm_min, payload_min)) >= safety_margin,
            arm_worst_index=arm_idx,
            payload_worst_index=payload_idx,
        )


CraneArmFCLModel = CraneArmCollisionModel
