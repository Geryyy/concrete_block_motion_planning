from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, minimize

from motion_planning.geometry.spline_opt import yaw_deg_to_quat


def _n(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, float).reshape(3)
    n = np.linalg.norm(v)
    return np.array([0.0, 0.0, -1.0]) if n <= 1e-12 else v / n


def _wrap(x: float) -> float:
    return math.atan2(math.sin(x), math.cos(x))


@dataclass(frozen=True)
class JointSpaceGlobalPathRequest:
    scene: Any
    moving_block_size: tuple[float, float, float]
    q_start: np.ndarray
    q_goal: np.ndarray
    start_approach_direction_world: tuple[float, float, float]
    goal_approach_direction_world: tuple[float, float, float]
    q_start_seed_map: Mapping[str, float] | None = None
    q_goal_seed_map: Mapping[str, float] | None = None
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class JointSpaceGlobalPathResult:
    success: bool
    message: str
    q_waypoints: np.ndarray
    via_points: np.ndarray
    tcp_xyz: np.ndarray
    tcp_yaw_rad: np.ndarray
    diagnostics: dict[str, Any]


class JointSpaceGlobalPathPlanner:
    def __init__(self, jsp) -> None:
        self.jsp, self.names = jsp, list(jsp._reduced_joint_names)

    @staticmethod
    def _sample_joint_bspline(ctrl: np.ndarray, u: np.ndarray) -> np.ndarray:
        p = np.asarray(ctrl, dtype=float).reshape(4, -1)
        t = np.asarray(u, dtype=float).reshape(-1, 1)
        omt = 1.0 - t
        return (
            (omt ** 3) * p[0]
            + 3.0 * (omt ** 2) * t * p[1]
            + 3.0 * omt * (t ** 2) * p[2]
            + (t ** 3) * p[3]
        )

    def plan(self, req: JointSpaceGlobalPathRequest) -> JointSpaceGlobalPathResult:
        q0, qN = np.asarray(req.q_start, float), np.asarray(req.q_goal, float)
        if q0.shape != qN.shape:
            z = np.zeros
            return JointSpaceGlobalPathResult(False, "start/goal joint dimensions do not match", z((0, q0.size)), z((0, q0.size)), z((0, 3)), z(0), {})
        c = {"sample_count": 31, "maxiter": 60, "safety_margin": 0.03, "approach_step_m": 0.08, "w_length": 1.0, "w_smooth": 0.25, "w_collision": 80.0, "w_penetration": 400.0, "w_approach": 20.0, **dict(req.config)}
        d0, dN = _n(req.start_approach_direction_world), _n(req.goal_approach_direction_world)
        s0 = None if req.q_start_seed_map is None else dict(req.q_start_seed_map)
        sN = None if req.q_goal_seed_map is None else dict(req.q_goal_seed_map)
        q1, q2 = self._step(q0, d0, c["approach_step_m"], s0), self._step(qN, -dN, c["approach_step_m"], sN)
        x0 = np.vstack([(2 * q1 + q2) / 3, (q1 + 2 * q2) / 3]).reshape(-1)
        lo, hi = zip(*[(self.jsp._joint_position_limits.get(n, (None, None))[0], self.jsp._joint_position_limits.get(n, (None, None))[1]) for _ in range(2) for n in self.names])
        bounds = Bounds(np.array([-np.inf if v is None else float(v) for v in lo]), np.array([np.inf if v is None else float(v) for v in hi]))

        def E(x: np.ndarray) -> dict[str, Any]:
            a = self._clip(np.vstack([q0, np.asarray(x, float).reshape(2, -1), qN])); a[0], a[-1] = q0, qN
            q = self._clip(self._sample_joint_bspline(a, np.linspace(0.0, 1.0, max(5, int(c["sample_count"])))))
            q[0], q[-1] = q0, qN
            xyz, yaw, maps, via_xyz = [], [], [], []
            seed = dict(s0) if s0 is not None else {n: float(q0[i]) for i, n in enumerate(self.names)}
            for qi in q:
                xi, yi, seed = self.jsp.fk_world_pose(qi, q_seed=seed)
                xyz.append(xi); yaw.append(yi); maps.append(dict(seed))
            seed = dict(s0) if s0 is not None else {n: float(q0[i]) for i, n in enumerate(self.names)}
            for qi in a[1:-1]:
                xi, _, seed = self.jsp.fk_world_pose(qi, q_seed=seed)
                via_xyz.append(xi)
            xyz, yaw, via_xyz = np.asarray(xyz, float), np.asarray(yaw, float), np.asarray(via_xyz, float).reshape(-1, 3)
            d = np.asarray([req.scene.signed_distance_block(req.moving_block_size, xyz[i], yaw_deg_to_quat(float(np.degrees(yaw[i])))) for i in range(len(q))], float)
            dq, ddq = np.diff(q, axis=0), np.diff(q, n=2, axis=0)
            a0 = float(np.degrees(np.arccos(np.clip(np.dot(_n(xyz[1] - xyz[0]), d0), -1.0, 1.0))))
            aN = float(np.degrees(np.arccos(np.clip(np.dot(_n(xyz[-1] - xyz[-2]), dN), -1.0, 1.0))))
            return {"q": q, "ctrl": a, "vias": a[1:-1], "xyz": xyz, "yaw": yaw, "maps": maps, "via_xyz": via_xyz, "j": float(c["w_length"] * np.sum(np.linalg.norm(dq, axis=1)) + c["w_smooth"] * np.sum(ddq * ddq) + c["w_collision"] * np.sum(np.maximum(0.0, c["safety_margin"] - d) ** 2) + c["w_penetration"] * np.sum(np.maximum(0.0, -d) ** 2) + c["w_approach"] * ((a0 / 180.0) ** 2 + (aN / 180.0) ** 2)), "min_d": float(np.min(d)), "mean_d": float(np.mean(d)), "len": float(np.sum(np.linalg.norm(dq, axis=1))), "smooth": float(np.sum(ddq * ddq)), "a0": a0, "aN": aN}

        def ok(r: dict[str, Any], dmin: float) -> bool:
            return r["min_d"] >= dmin

        best = {"r": E(x0)}
        opt = None if ok(best["r"], max(-1e-3, c["safety_margin"] - 5e-3)) else minimize(lambda x: best.__setitem__("r", E(x)) or best["r"]["j"], x0=x0, method="Powell", bounds=bounds, options={"maxiter": int(c["maxiter"]), "disp": False})
        r, success = best["r"], ok(best["r"], -1e-3)
        return JointSpaceGlobalPathResult(success, f"joint-space global via-2 path {'ok' if success else 'needs review'}: min clearance={r['min_d']:+.3f} m", np.asarray(r["q"], float), np.asarray(r["vias"], float), np.asarray(r["xyz"], float), np.asarray(r["yaw"], float), {"via_point_count": 2.0, "optimizer_success": bool(True if opt is None else opt.success), "optimizer_iterations": float(0 if opt is None else getattr(opt, "nit", 0) or 0), "objective_final": float(r["j"]), "min_signed_distance_m": float(r["min_d"]), "mean_signed_distance_m": float(r["mean_d"]), "path_length_joint": float(r["len"]), "smoothness_cost": float(r["smooth"]), "start_approach_alignment_deg": float(r["a0"]), "goal_approach_alignment_deg": float(r["aN"]), "q_maps_path": list(r["maps"]), "q_control_points": np.asarray(r["ctrl"], float), "via_points": np.asarray(r["vias"], float), "via_tcp_xyz": np.asarray(r["via_xyz"], float), "reference_path_backend": "joint_space_global_via2", "reference_path_fallback_used": 0.0, "joint_anchor_fallback_used": 0.0})

    def _step(self, q: np.ndarray, direction: np.ndarray, step_m: float, q_seed: Mapping[str, float] | None = None) -> np.ndarray:
        q = np.asarray(q, float).reshape(-1)
        xyz0, _, seed = self.jsp.fk_world_pose(q, q_seed=q_seed)
        J = np.zeros((3, q.size), float)
        for j in range(q.size):
            q1 = q.copy(); q1[j] += 1e-4
            xyz1, _, seed = self.jsp.fk_world_pose(self._clip(q1), q_seed=seed)
            J[:, j] = (xyz1 - xyz0) / 1e-4
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), _n(direction) * float(step_m))
        return self._clip(q + dq)

    def _clip(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, float).copy()
        for j, n in enumerate(self.names):
            lo, hi = self.jsp._joint_position_limits.get(n, (None, None))
            if lo is not None: q[..., j] = np.maximum(q[..., j], float(lo))
            if hi is not None: q[..., j] = np.minimum(q[..., j], float(hi))
        return q
