from __future__ import annotations

import math

import numpy as np

from .types import PlanEvaluation, StandalonePlanResult


def _wrap_to_pi(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _point_to_polyline_distance(point: np.ndarray, polyline: np.ndarray) -> float:
    p = np.asarray(point, dtype=float).reshape(3)
    line = np.asarray(polyline, dtype=float).reshape(-1, 3)
    if line.shape[0] <= 1:
        return float(np.linalg.norm(p - line[0])) if line.shape[0] == 1 else 0.0
    best = float("inf")
    for a, b in zip(line[:-1], line[1:]):
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            cand = float(np.linalg.norm(p - a))
        else:
            t = float(np.dot(p - a, ab) / denom)
            t = max(0.0, min(1.0, t))
            proj = a + t * ab
            cand = float(np.linalg.norm(p - proj))
        best = min(best, cand)
    return best


def evaluate_plan(result: StandalonePlanResult) -> PlanEvaluation:
    tcp_xyz = np.asarray(result.tcp_xyz, dtype=float).reshape(-1, 3)
    ref_xyz = np.asarray(result.reference_xyz, dtype=float).reshape(-1, 3)
    tcp_yaw = np.asarray(result.tcp_yaw_rad, dtype=float).reshape(-1)
    ref_yaw = np.asarray(result.reference_yaw_rad, dtype=float).reshape(-1)

    pos_err = np.linalg.norm(tcp_xyz - ref_xyz, axis=1)
    yaw_err_deg = np.degrees(np.abs([_wrap_to_pi(float(a - b)) for a, b in zip(tcp_yaw, ref_yaw)]))
    path_len = 0.0
    if tcp_xyz.shape[0] > 1:
        path_len = float(np.linalg.norm(np.diff(tcp_xyz, axis=0), axis=1).sum())
    max_dev = 0.0
    if ref_xyz.shape[0] > 0:
        max_dev = max(_point_to_polyline_distance(point, ref_xyz) for point in tcp_xyz)

    evaluation = PlanEvaluation(
        final_position_error_m=float(pos_err[-1]),
        final_yaw_error_deg=float(yaw_err_deg[-1]),
        max_position_error_m=float(np.max(pos_err)),
        mean_position_error_m=float(np.mean(pos_err)),
        max_path_deviation_m=float(max_dev),
        path_length_m=float(path_len),
        metadata={},
    )
    result.evaluation = evaluation
    return evaluation
