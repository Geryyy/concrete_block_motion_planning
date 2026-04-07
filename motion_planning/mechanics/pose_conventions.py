from __future__ import annotations

import numpy as np


def phi_tool_from_rotation(R: np.ndarray) -> float:
    R_arr = np.asarray(R, dtype=float).reshape(3, 3)
    return float(np.arctan2(R_arr[1, 1], R_arr[0, 1]))


def phi_tool_from_transform(T: np.ndarray) -> float:
    T_arr = np.asarray(T, dtype=float).reshape(4, 4)
    return phi_tool_from_rotation(T_arr[:3, :3])


def pose_from_pos_yaw(pos: np.ndarray, yaw: float) -> np.ndarray:
    rot_z = float(yaw) - 0.5 * np.pi
    cy, sy = float(np.cos(rot_z)), float(np.sin(rot_z))
    T = np.eye(4, dtype=float)
    T[0, 0] = cy
    T[0, 1] = -sy
    T[1, 0] = sy
    T[1, 1] = cy
    T[:3, 3] = np.asarray(pos, dtype=float).reshape(3)
    return T
