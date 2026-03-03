from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


def yaw_deg_to_quat(yaw_deg: float) -> Tuple[float, float, float, float]:
    """Quaternion [x,y,z,w] for a pure yaw rotation about +z."""
    half = 0.5 * np.deg2rad(float(yaw_deg))
    return (0.0, 0.0, float(np.sin(half)), float(np.cos(half)))


@dataclass(frozen=True)
class PathEvalContext:
    scene: Any
    goal: np.ndarray
    moving_block_size: Tuple[float, float, float]
    start_yaw_deg: float
    goal_yaw_deg: float
    goal_normals: np.ndarray
    config: Dict[str, Any]


def make_eval_context(
    *,
    scene: Any,
    goal: np.ndarray,
    moving_block_size: Tuple[float, float, float],
    start_yaw_deg: float,
    goal_yaw_deg: float,
    goal_normals: np.ndarray,
    config: Dict[str, Any],
) -> PathEvalContext:
    return PathEvalContext(
        scene=scene,
        goal=np.asarray(goal, dtype=float).reshape(3),
        moving_block_size=tuple(float(x) for x in moving_block_size),
        start_yaw_deg=float(start_yaw_deg),
        goal_yaw_deg=float(goal_yaw_deg),
        goal_normals=np.asarray(goal_normals, dtype=float).reshape(-1, 3),
        config=dict(config),
    )


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def goal_approach_alignment_cost(P: np.ndarray, goal_normals: np.ndarray, terminal_fraction: float) -> float:
    if P.shape[0] < 3 or goal_normals.size == 0:
        return 0.0
    tail_n = max(3, int(np.ceil(float(terminal_fraction) * P.shape[0])))
    v = normalize(np.sum(np.diff(P[-tail_n:], axis=0), axis=0))
    if not np.any(v):
        return 0.0

    N = np.asarray(goal_normals, dtype=float).reshape(-1, 3)
    Nn = np.array([normalize(n) for n in N], dtype=float)
    s = normalize(np.sum(Nn, axis=0)) if Nn.size else np.zeros(3, dtype=float)
    if not np.any(s) and Nn.size:
        s = normalize(Nn[0])
    if not np.any(s):
        return 0.0

    c = float(np.dot(v, -s))
    return float((1.0 - np.clip(c, -1.0, 1.0)) ** 2)


def path_length(P: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))


def curvature_cost(P: np.ndarray) -> float:
    P = np.asarray(P, dtype=float)
    if P.shape[0] < 3:
        return 0.0
    du = 1.0 / float(P.shape[0] - 1)
    d1 = np.gradient(P, du, axis=0)
    d2 = np.gradient(d1, du, axis=0)
    speed = np.linalg.norm(d1, axis=1)
    cross = np.linalg.norm(np.cross(d1, d2), axis=1)
    kappa = cross / np.maximum(speed, 1e-9) ** 3
    return float(np.sum((kappa * kappa) * speed) * du)


def mean_turn_angle_deg(P: np.ndarray) -> float:
    dP = np.diff(P, axis=0)
    if dP.shape[0] < 2:
        return 0.0
    a = dP[:-1]
    b = dP[1:]
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    valid = (an > 1e-12) & (bn > 1e-12)
    if not np.any(valid):
        return 0.0
    cosang = np.sum(a[valid] * b[valid], axis=1) / (an[valid] * bn[valid])
    ang = np.arccos(np.clip(cosang, -1.0, 1.0))
    return float(np.degrees(np.mean(ang)))


def distance_profile(scene, P: np.ndarray, moving_block_size: Tuple[float, float, float], yaw_samples: np.ndarray) -> np.ndarray:
    return np.array(
        [
            scene.signed_distance_block(
                size=moving_block_size,
                position=P[i],
                quat=yaw_deg_to_quat(float(yaw_samples[i])),
            )
            for i in range(P.shape[0])
        ],
        dtype=float,
    )


def evaluated_clearance_subset(
    P: np.ndarray,
    d: np.ndarray,
    goal: np.ndarray,
    contact_window_fraction: float,
    goal_contact_radius: float = 0.08,
) -> np.ndarray:
    """Return distances used for clearance/success evaluation.

    We ignore:
    1) the configured terminal contact window in parametric time, and
    2) terminal samples already within goal tolerance (allows intended placement contact).
    """
    n = P.shape[0]
    us = np.linspace(0.0, 1.0, n)
    mask_contact = us < (1.0 - float(contact_window_fraction))
    goal_dist = np.linalg.norm(np.asarray(P, dtype=float) - np.asarray(goal, dtype=float).reshape(1, 3), axis=1)
    mask_not_goal_contact = goal_dist > float(goal_contact_radius)
    mask = mask_contact & mask_not_goal_contact
    out = d[mask]
    if out.size == 0:
        out = d[mask_contact] if np.any(mask_contact) else d[:-1]
    if out.size == 0:
        out = d
    return out


def evaluate_path_metrics(
    ctx: PathEvalContext,
    P: np.ndarray,
    message: str,
    nit: int,
    yaw_samples_deg: np.ndarray | None = None,
    solver_success: bool | None = None,
) -> Dict[str, Any]:
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    if yaw_samples_deg is None:
        yaw_samples = np.linspace(ctx.start_yaw_deg, ctx.goal_yaw_deg, n)
    else:
        yaw_samples = np.asarray(yaw_samples_deg, dtype=float).reshape(-1)
        if yaw_samples.shape[0] != n:
            raise ValueError("yaw_samples_deg must have same length as trajectory samples.")
    d = distance_profile(ctx.scene, P, ctx.moving_block_size, yaw_samples)

    required = float(ctx.config.get("safety_margin", 0.0))
    def_req = np.maximum(0.0, required - d)
    j_safe = float(np.sum(def_req * def_req))

    contact_window_fraction = float(ctx.config.get("contact_window_fraction", 0.1))
    d_approach = evaluated_clearance_subset(
        P=P,
        d=d,
        goal=ctx.goal,
        contact_window_fraction=contact_window_fraction,
        goal_contact_radius=0.08,
    )
    col_approach = np.maximum(0.0, -d_approach)
    j_approach_col = float(np.sum(col_approach * col_approach))

    j_len = path_length(P)
    j_curv = curvature_cost(P)
    j_goal_normal = goal_approach_alignment_cost(
        P,
        goal_normals=ctx.goal_normals,
        terminal_fraction=float(ctx.config.get("goal_approach_window_fraction", 0.1)),
    )

    fun = (
        float(ctx.config.get("w_len", 1.0)) * j_len
        + float(ctx.config.get("w_curv", 0.1)) * j_curv
        + float(ctx.config.get("w_safe", 50.0)) * j_safe
        + float(ctx.config.get("w_approach_collision", 0.0)) * j_approach_col
        + float(ctx.config.get("w_goal_approach_normal", 0.0)) * j_goal_normal
    )
    goal_err = float(np.linalg.norm(P[-1] - ctx.goal))
    # Evaluate clearance over the non-contact approach segment to avoid
    # false negative penalties from terminal goal-contact samples.
    min_clear_eval = float(np.min(d_approach))
    mean_clear_eval = float(np.mean(d_approach))
    success = bool(min_clear_eval >= -1e-3 and goal_err <= 0.08)
    if solver_success is not None:
        success = bool(success and bool(solver_success))

    return {
        "success": success,
        "message": message,
        "fun": float(fun),
        "length": float(j_len),
        "curvature_cost": float(j_curv),
        "yaw_smoothness_cost": 0.0,
        "safety_cost": float(j_safe),
        "preferred_safety_cost": 0.0,
        "approach_rebound_cost": 0.0,
        "goal_clearance_cost": 0.0,
        "goal_clearance_target_cost": 0.0,
        "approach_clearance_cost": 0.0,
        "approach_collision_cost": float(j_approach_col),
        "via_deviation_cost": 0.0,
        "yaw_deviation_cost": 0.0,
        "yaw_monotonic_cost": 0.0,
        "yaw_schedule_cost": 0.0,
        "goal_approach_normal_cost": float(j_goal_normal),
        "min_clearance": min_clear_eval,
        "mean_clearance": mean_clear_eval,
        "min_clearance_raw": float(np.min(d)),
        "mean_clearance_raw": float(np.mean(d)),
        "turn_angle_mean_deg": mean_turn_angle_deg(P),
        "nit": int(nit),
    }


def scenario_score(info: Dict[str, Any], runtime_s: float) -> float:
    min_clear = float(info.get("min_clearance", -1.0))
    collision_penalty = 50_000.0 * max(0.0, -min_clear) ** 2
    success_penalty = 10_000.0 if not bool(info.get("success", False)) else 0.0
    return float(info["fun"]) + collision_penalty + success_penalty + 0.15 * runtime_s


def aggregate_numeric(per_scenario: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    keys = [
        "score",
        "runtime_s",
        "fun",
        "length",
        "path_efficiency",
        "curvature_cost",
        "turn_angle_mean_deg",
        "yaw_smoothness_cost",
        "safety_cost",
        "preferred_safety_cost",
        "approach_rebound_cost",
        "goal_clearance_cost",
        "goal_clearance_target_cost",
        "approach_clearance_cost",
        "approach_collision_cost",
        "goal_approach_normal_cost",
        "min_clearance",
        "mean_clearance",
        "min_clearance_raw",
        "mean_clearance_raw",
        "nit",
    ]
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = np.asarray([float(r.get(k, 0.0)) for r in per_scenario], dtype=float)
        out[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return out
