import numpy as np
from typing import Tuple
from .scene import Scene
from .utils import quat_to_rot

def goal_and_via_for_placement(
    scene: Scene,
    base_id,
    direction: str,
    gap: float = 0.01,
    lateral_offset: Tuple[float, float] = (0.0, 0.0),
    interlock_x: float = 0.0,
    via_shift_along_normal: float = 0.05,
    via_offset_local: Tuple[float, float, float] = (0.0, 0.0, 0.25),
):
    """
    Compute goal and via for a repeated wall-assembly placement.

    Conventions (local block frame):
    - front: +x normal; lateral axes = (y, z)
    - back:  -x normal; lateral axes = (y, z)
    - left:  +y normal; lateral axes = (x, z)
    - right: -y normal; lateral axes = (x, z)
    - top:   +z normal; lateral axes = (x, y), plus interlock along local +x

    Args:
        scene: Scene instance.
        base_id: object_id or index of the base block.
        direction: one of {'front', 'back', 'left', 'right', 'top'}.
        gap: distance from the base face to the goal along the face normal.
        lateral_offset: (u, v) offset along the two face tangents (see above).
        interlock_x: extra shift along local +x applied for 'top' placements only.
        via_shift_along_normal: how much to shift the via along the face normal from the goal.via_shift_along_normal
        via_offset_local: fixed via offset expressed in the block's local frame (dx, dy, dz).

    Returns:
        goal (3,), via (3,)
    """
    b = scene.get_block(base_id)
    R = quat_to_rot(b.quat)           # columns are local x,y,z in world
    c = np.asarray(b.position, float) # block center in world
    hx, hy, hz = 0.5 * np.asarray(b.size, float)  # half extents in local frame

    direction = direction.lower()
    face_normal = None

    if direction == "front":
        n_hat = R[:, 0]               # +x
        u_hat, v_hat = R[:, 1], R[:, 2]  # (y, z)
        h_n = hx
        u_off, v_off = lateral_offset
        goal = c + n_hat * (h_n + gap) + u_hat * u_off + v_hat * v_off
        face_normal = n_hat

    elif direction == "back":
        n_hat = -R[:, 0]              # -x
        u_hat, v_hat = R[:, 1], R[:, 2]  # (y, z)
        h_n = hx
        u_off, v_off = lateral_offset
        goal = c + n_hat * (h_n + gap) + u_hat * u_off + v_hat * v_off
        face_normal = n_hat

    elif direction == "left":
        n_hat = R[:, 1]               # +y
        u_hat, v_hat = R[:, 0], R[:, 2]  # (x, z)
        h_n = hy
        u_off, v_off = lateral_offset
        goal = c + n_hat * (h_n + gap) + u_hat * u_off + v_hat * v_off
        face_normal = n_hat

    elif direction == "right":
        n_hat = -R[:, 1]              # -y
        u_hat, v_hat = R[:, 0], R[:, 2]  # (x, z)
        h_n = hy
        u_off, v_off = lateral_offset
        goal = c + n_hat * (h_n + gap) + u_hat * u_off + v_hat * v_off
        face_normal = n_hat

    elif direction == "top":
        n_hat = R[:, 2]               # +z
        u_hat, v_hat = R[:, 0], R[:, 1]  # (x, y)
        h_n = hz
        # Apply interlock along local +x (added to u component)
        u_off = lateral_offset[0] + interlock_x
        v_off = lateral_offset[1]
        goal = c + n_hat * (h_n + gap) + u_hat * u_off + v_hat * v_off
        face_normal = n_hat

    else:
        raise ValueError("direction must be one of {'front','back','left','right','top'}")

    # Fixed via offset in local frame, mapped to world
    via = goal + (R @ np.asarray(via_offset_local, float)) + face_normal * via_shift_along_normal

    return goal, via
