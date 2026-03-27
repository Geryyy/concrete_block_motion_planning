"""hppfcl collision model for the CBS crane arm.

Architecture
------------
Collision checking and joint-state completion are **decoupled**:

  clearance(q_map, scene)          — pure geometry, needs full joint state
  complete_joint_map(q_act, ...)   — planning adapter, fills in passive joints

This means the collision checker has no opinion about how passive joints are
determined.  The planning layer decides: analytic equilibrium (fast, default),
numeric equilibrium (precise, for final checks), or fixed values (simplest).

Full capsule model
------------------
  Segment                          Frames                             r [m]
  ────────────────────────────────────────────────────────────────────────
  Slewing column  K0_mounting_base → K1_slewing_column               0.15
  Boom            K1_slewing_column → K2_boom                        0.14
  Arm elbow       K2_boom → K3_arm                                   0.10
  Telescope       K3_arm → K5_inner_telescope                        0.08
  Boom cylinder   K1_boom_cylinder_suspension → boom_cylinder_piston 0.06
  Tip link        K5_inner_telescope → K6_double_joint_link          0.05
  Rotator body    K6_double_joint_link → K8_tool_center_point        0.08
  Gripper mount   K11 → K9                                           0.06
  Outer jaw       K9 → K10_outer_jaw                                 0.04
  Inner jaw       K11 → K12_inner_jaw                                0.04

Collision backend
-----------------
hppfcl returns proper signed distances including penetration depths as
negative values — no sentinel workarounds needed.

Usage
-----
    model = CraneArmCollisionModel()

    # Path planning: fill passive joints from analytic equilibrium, then check
    for q_act in waypoints:
        q_map = model.complete_joint_map(q_act)          # fast analytic
        d = model.clearance(q_map, scene)

    # Precise final check: numeric equilibrium from SteadyStateResult
    q_map = model.complete_joint_map(q_act, q_passive={"theta6_tip_joint": th6, ...})
    d = model.clearance(q_map, scene)

    # Combined path report
    report = model.check_path(q_maps, tcp_xyz_path, tcp_yaw_path,
                              moving_block_size, scene)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import hppfcl as _hppfcl
    _HPPFCL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HPPFCL_AVAILABLE = False

from .scene import Scene


# ---------------------------------------------------------------------------
# Capsule geometry definitions — full chain, single model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _CapsuleDef:
    p1_frame: str
    p2_frame: str
    radius: float  # metres


_CAPSULES: tuple[_CapsuleDef, ...] = (
    # ── Arm (K0→K5) — actuated joints ──────────────────────────────────────
    _CapsuleDef("K0_mounting_base",            "K1_slewing_column",       0.15),
    _CapsuleDef("K1_slewing_column",           "K2_boom",                 0.14),
    _CapsuleDef("K2_boom",                     "K3_arm",                  0.10),
    # K3 is 39 cm off the K2→K5 straight line — must not be skipped
    _CapsuleDef("K3_arm",                      "K5_inner_telescope",      0.08),
    _CapsuleDef("K1_boom_cylinder_suspension", "boom_cylinder_piston",    0.06),
    # ── Tool chain (K5→K8) — passive joints ────────────────────────────────
    _CapsuleDef("K5_inner_telescope",          "K6_double_joint_link",    0.05),
    _CapsuleDef("K6_double_joint_link",        "K8_tool_center_point",    0.08),
    # ── PZS100 gripper ──────────────────────────────────────────────────────
    _CapsuleDef("K11",                         "K9",                      0.06),
    _CapsuleDef("K9",                          "K10_outer_jaw",           0.04),
    _CapsuleDef("K11",                         "K12_inner_jaw",           0.04),
)

# Jaw angle that grips a 60×60×90 cm CBS concrete block [rad]
GRIP_ANGLE_60CM: float = 1.6101


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotation_z_to(d: np.ndarray) -> np.ndarray:
    """Rodrigues rotation: map z-axis onto unit vector d."""
    d = np.asarray(d, dtype=float)
    d = d / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0], dtype=float)
    axis = np.cross(z, d)
    sin_a = float(np.linalg.norm(axis))
    cos_a = float(np.dot(z, d))
    if sin_a < 1e-8:
        return np.eye(3) if cos_a > 0.0 else np.diag([1.0, -1.0, -1.0])
    axis = axis / sin_a
    K_mat = np.array(
        [[0.0, -axis[2], axis[1]],
         [axis[2], 0.0, -axis[0]],
         [-axis[1], axis[0], 0.0]],
        dtype=float,
    )
    return np.eye(3) + sin_a * K_mat + (1.0 - cos_a) * (K_mat @ K_mat)


def _make_capsule_shape_tf(p1: np.ndarray, p2: np.ndarray, radius: float):
    """Return (hppfcl.Capsule, hppfcl.Transform3f) or None if degenerate."""
    if not _HPPFCL_AVAILABLE:
        raise ImportError("hppfcl is required for arm collision checking")
    d = p2 - p1
    length = float(np.linalg.norm(d))
    if length < 1e-6:
        return None
    R = _rotation_z_to(d / length)
    shape = _hppfcl.Capsule(float(radius), float(length))
    tf = _hppfcl.Transform3f()
    tf.setRotation(R)
    tf.setTranslation((0.5 * (p1 + p2)).astype(float))
    return shape, tf


def _capsule_min_dist(cap_shape, cap_tf, scene: Scene,
                      ignore_ids: Optional[List[str]]) -> float:
    ignore = set(ignore_ids or [])
    req = _hppfcl.DistanceRequest()
    min_dist = np.inf
    for blk in scene.blocks:
        if blk.object_id is not None and blk.object_id in ignore:
            continue
        static_shape, static_tf = blk.hppfcl_shape_tf()
        res = _hppfcl.DistanceResult()
        d = float(_hppfcl.distance(cap_shape, cap_tf, static_shape, static_tf, req, res))
        if d < min_dist:
            min_dist = d
    return float(min_dist)


# ---------------------------------------------------------------------------
# Path check result
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Collision model
# ---------------------------------------------------------------------------

class CraneArmCollisionModel:
    """hppfcl collision model for the full CBS crane arm.

    The single ``clearance()`` method requires a **complete joint map** covering
    actuated joints, passive joints, and gripper jaw angles.  Use
    ``complete_joint_map()`` to build one from the 5 actuated joints alone
    (passive joints are filled by analytic equilibrium).

    Lazy-loads pinocchio and the CBS URDF on first use.
    """

    def __init__(self) -> None:
        self._pin = None
        self._model = None
        self._data = None
        self._act_names: list[str] = []
        self._frame_cache: dict[str, int] = {}
        self._steady_state = None  # lazy CraneSteadyState

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import pinocchio as pin
        except ImportError as e:
            raise ImportError("pinocchio is required for CraneArmCollisionModel") from e
        from motion_planning.pipeline import JointGoalStage
        stage = JointGoalStage()
        self._pin = pin
        self._act_names = list(stage.config.actuated_joints)
        self._model = pin.buildModelFromUrdf(stage.config.urdf_path)
        self._data = self._model.createData()
        # Reuse the steady-state solver already constructed by JointGoalStage
        self._steady_state = stage._steady_state

    def _ensure_steady_state(self):
        self._ensure_loaded()
        return self._steady_state

    # ------------------------------------------------------------------
    # Joint map builder — planning adapter
    # ------------------------------------------------------------------

    def complete_joint_map(
        self,
        q_act: np.ndarray,
        q_passive: Optional[Dict[str, float]] = None,
        jaw_angle: float = GRIP_ANGLE_60CM,
    ) -> Dict[str, float]:
        """Build a full joint map from the 5 actuated joints.

        Passive joints (theta6, theta7) are filled from *q_passive* if
        provided, otherwise from analytic equilibrium (fast, suitable for
        planning).  Gripper jaw angles are set to *jaw_angle*.

        Parameters
        ----------
        q_act     : (5,) actuated joints [theta1, theta2, theta3, q4, theta8]
        q_passive : optional dict with 'theta6_tip_joint' / 'theta7_tilt_joint'
                    — use this when you have a precise numeric equilibrium result
        jaw_angle : gripper jaw angle [rad]; GRIP_ANGLE_60CM for a 60 cm block
        """
        self._ensure_loaded()
        q_map: Dict[str, float] = {
            name: float(q_act[i]) for i, name in enumerate(self._act_names)
        }

        if q_passive is not None:
            theta6 = float(q_passive.get("theta6_tip_joint",  0.0))
            theta7 = float(q_passive.get("theta7_tilt_joint", np.pi / 2))
        else:
            ss = self._ensure_steady_state()
            theta6, theta7 = ss.analytic_equilibrium(
                q_map["theta2_boom_joint"],
                q_map["theta3_arm_joint"],
                q_map["theta8_rotator_joint"],
            )

        q_map["theta6_tip_joint"]        = theta6
        q_map["theta7_tilt_joint"]       = theta7
        q_map["theta10_outer_jaw_joint"] = float(jaw_angle)
        q_map["theta12_inner_jaw_joint"] = float(jaw_angle)
        return q_map

    # ------------------------------------------------------------------
    # FK
    # ------------------------------------------------------------------

    def _frame_pos(self, q_map: Dict[str, float], frame_name: str) -> np.ndarray:
        """Return position of *frame_name* in K0_mounting_base frame."""
        from motion_planning.mechanics.analytic.pinocchio_utils import fk_homogeneous
        H = fk_homogeneous(
            pin_model=self._model,
            pin_data=self._data,
            pin_module=self._pin,
            q_values=q_map,
            base_frame="K0_mounting_base",
            end_frame=frame_name,
            frame_cache=self._frame_cache,
        )
        return np.asarray(H[:3, 3], dtype=float)

    # ------------------------------------------------------------------
    # Core collision check
    # ------------------------------------------------------------------

    def clearance(
        self,
        q_map: Dict[str, float],
        scene: Scene,
        ignore_ids: Optional[List[str]] = None,
    ) -> float:
        """Minimum signed distance from the full arm+tool+gripper to scene blocks.

        Parameters
        ----------
        q_map     : complete joint map (actuated + passive + jaw angles).
                    Build one with ``complete_joint_map()`` if you only have
                    the 5 actuated joints.
        scene     : static scene
        ignore_ids: block IDs to skip (e.g. ``["table"]`` — the arm is
                    designed to extend below the table plane during placement)

        Returns
        -------
        float
            Positive = clearance [m], negative = penetration depth [m].
        """
        if not scene.blocks:
            return np.inf
        self._ensure_loaded()
        min_dist = np.inf
        for cap in _CAPSULES:
            p1 = self._frame_pos(q_map, cap.p1_frame)
            p2 = self._frame_pos(q_map, cap.p2_frame)
            result = _make_capsule_shape_tf(p1, p2, cap.radius)
            if result is None:
                continue
            d = _capsule_min_dist(result[0], result[1], scene, ignore_ids)
            if d < min_dist:
                min_dist = d
        return float(min_dist)

    # ------------------------------------------------------------------
    # Payload (moving block) clearance — separate concern
    # ------------------------------------------------------------------

    def payload_clearance(
        self,
        tcp_xyz: np.ndarray,
        tcp_yaw: float,
        block_size: Sequence[float],
        scene: Scene,
        ignore_ids: Optional[List[str]] = None,
    ) -> float:
        """Minimum signed distance from the payload box to scene blocks.

        The moving block is modelled as an axis-aligned box at TCP position
        with yaw-only rotation.  Delegates to ``scene.signed_distance_block()``.

        Parameters
        ----------
        tcp_xyz   : (3,) TCP position in K0 frame
        tcp_yaw   : TCP yaw angle [rad]
        block_size: (sx, sy, sz) of the moving block
        """
        half = float(tcp_yaw) * 0.5
        quat = (0.0, 0.0, float(np.sin(half)), float(np.cos(half)))
        return scene.signed_distance_block(
            size=block_size,
            position=np.asarray(tcp_xyz, dtype=float),
            quat=quat,
            ignore_ids=ignore_ids,
        )

    # ------------------------------------------------------------------
    # Path check
    # ------------------------------------------------------------------

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
        """Check collision clearance along a joint-space path.

        Parameters
        ----------
        q_maps           : list of N full joint maps (from ``complete_joint_map()``)
        tcp_xyz_path     : (N, 3) TCP positions in K0 frame
        tcp_yaw_path     : (N,) TCP yaw angles [rad]
        moving_block_size: (sx, sy, sz) of the block being moved
        scene            : static scene
        arm_ignore_ids   : IDs to skip for arm checks (default: skip "table")
        payload_ignore_ids: IDs to skip for payload checks
        safety_margin    : threshold below which collision_free=False [m]
        """
        if arm_ignore_ids is None:
            arm_ignore_ids = ["table"]

        xyz_arr = np.asarray(tcp_xyz_path, dtype=float)
        yaw_arr = np.asarray(tcp_yaw_path, dtype=float)
        N = len(q_maps)

        arm_dists = np.full(N, np.inf)
        payload_dists = np.full(N, np.inf)

        for i, q_map in enumerate(q_maps):
            arm_dists[i] = self.clearance(q_map, scene, ignore_ids=arm_ignore_ids)
            payload_dists[i] = self.payload_clearance(
                xyz_arr[i], float(yaw_arr[i]),
                moving_block_size, scene,
                ignore_ids=payload_ignore_ids,
            )

        arm_idx = int(np.argmin(arm_dists))
        pay_idx = int(np.argmin(payload_dists))
        arm_min = float(arm_dists[arm_idx])
        pay_min = float(payload_dists[pay_idx])

        return ArmPathClearanceReport(
            arm_min_clearance_m=arm_min,
            payload_min_clearance_m=pay_min,
            combined_min_clearance_m=float(min(arm_min, pay_min)),
            collision_free=float(min(arm_min, pay_min)) >= safety_margin,
            arm_worst_index=arm_idx,
            payload_worst_index=pay_idx,
        )


# ---------------------------------------------------------------------------
# Back-compat alias
# ---------------------------------------------------------------------------
CraneArmFCLModel = CraneArmCollisionModel
