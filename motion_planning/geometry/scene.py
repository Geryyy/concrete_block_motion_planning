from typing import List, Tuple, Optional, Union
import numpy as np
import hppfcl

from .blocks import Block
from .utils import quat_to_rot

IdLike = Union[int, str]  # allow index (int) or object_id (str)

class Scene:
    def __init__(self):
        self.blocks: List[Block] = []
        self._id_to_index: dict[str, int] = {}
        self._auto_id_counter: int = 0

    def _ensure_object_id(self, object_id: Optional[str]) -> str:
        if object_id is None:
            oid = f"obj_{self._auto_id_counter}"
            self._auto_id_counter += 1
            return oid
        if object_id in self._id_to_index:
            raise ValueError(f"object_id '{object_id}' already exists in scene.")
        return object_id

    def _index_from_id(self, id_or_index: IdLike) -> int:
        if isinstance(id_or_index, int):
            if id_or_index < 0 or id_or_index >= len(self.blocks):
                raise IndexError(f"Block index {id_or_index} out of range.")
            return id_or_index
        # string id
        if id_or_index not in self._id_to_index:
            raise KeyError(f"object_id '{id_or_index}' not found.")
        return self._id_to_index[id_or_index]

    def add_block(self, size, position, quat=(0.0, 0.0, 0.0, 1.0), object_id: Optional[str] = None) -> str:
        """Add a block and return its object_id."""
        oid = self._ensure_object_id(object_id)
        self.blocks.append(Block(size=size, position=position, quat=quat, object_id=oid))
        self._id_to_index[oid] = len(self.blocks) - 1
        return oid

    def get_block(self, id_or_index: IdLike) -> Block:
        """Retrieve a block by integer index or by object_id (string)."""
        idx = self._index_from_id(id_or_index)
        return self.blocks[idx]

    def update_block(
        self,
        id_or_index: IdLike,
        *,
        position: Optional[Tuple[float, float, float]] = None,
        quat: Optional[Tuple[float, float, float, float]] = None,
        size: Optional[Tuple[float, float, float]] = None,
    ) -> Block:
        """Update position, orientation, or size of an existing block in place."""
        idx = self._index_from_id(id_or_index)
        b = self.blocks[idx]
        self.blocks[idx] = Block(
            size=size if size is not None else b.size,
            position=position if position is not None else b.position,
            quat=quat if quat is not None else b.quat,
            object_id=b.object_id,
        )
        return self.blocks[idx]

    def remove_block(self, id_or_index: IdLike) -> None:
        """Remove a block from the scene."""
        idx = self._index_from_id(id_or_index)
        self.blocks.pop(idx)
        # Rebuild id→index map since indices shifted.
        self._id_to_index = {
            b.object_id: i
            for i, b in enumerate(self.blocks)
            if b.object_id is not None
        }

    def fcl_objects(self):
        return [b.hppfcl_shape_tf() for b in self.blocks]

    def signed_distance(self, p: np.ndarray, point_radius: float = 1e-6) -> float:
        """Minimum signed distance from point p to the union of blocks.
        Positive outside, negative inside any block.
        """
        # Use analytic point-to-OBB signed distance to avoid backend-specific
        # sentinel values (e.g. -1 on overlap) from FCL distance queries.
        p_arr = np.asarray(p, dtype=float).reshape(3)
        min_sdf = np.inf
        for b in self.blocks:
            sdf = self._point_signed_distance_to_block(p_arr, b)
            if sdf < min_sdf:
                min_sdf = sdf
        return float(min_sdf)

    def signed_distance_block(
        self,
        size: Tuple[float, float, float],
        position: np.ndarray,
        quat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        ignore_ids: Optional[List[str]] = None,
    ) -> float:
        """Minimum signed distance from a moving box to scene blocks.

        Positive = separated [m], negative = penetration depth [m].
        Uses hppfcl which returns proper signed distances (no sentinel values).
        """
        sx, sy, sz = map(float, size)
        R = quat_to_rot(quat)
        T = np.asarray(position, dtype=float).reshape(3)
        moving_shape = hppfcl.Box(sx, sy, sz)
        moving_tf = hppfcl.Transform3f()
        moving_tf.setRotation(R)
        moving_tf.setTranslation(T)

        req = hppfcl.DistanceRequest()
        ignore = set(ignore_ids or [])
        min_dist = np.inf

        for b in self.blocks:
            if b.object_id is not None and b.object_id in ignore:
                continue
            static_shape, static_tf = b.hppfcl_shape_tf()
            res = hppfcl.DistanceResult()
            d = float(hppfcl.distance(moving_shape, moving_tf, static_shape, static_tf, req, res))
            if d < min_dist:
                min_dist = d

        return float(min_dist) if np.isfinite(min_dist) else np.inf

    def _sample_box_points(
        self,
        size: Tuple[float, float, float],
        position: np.ndarray,
        quat: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """Sample center, face-centers and corners of the oriented box."""
        sx, sy, sz = map(float, size)
        hx, hy, hz = 0.5 * sx, 0.5 * sy, 0.5 * sz
        local_pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [hx, 0.0, 0.0],
                [-hx, 0.0, 0.0],
                [0.0, hy, 0.0],
                [0.0, -hy, 0.0],
                [0.0, 0.0, hz],
                [0.0, 0.0, -hz],
                [hx, hy, hz],
                [hx, hy, -hz],
                [hx, -hy, hz],
                [hx, -hy, -hz],
                [-hx, hy, hz],
                [-hx, hy, -hz],
                [-hx, -hy, hz],
                [-hx, -hy, -hz],
            ],
            dtype=float,
        )
        R = quat_to_rot(quat)
        T = np.asarray(position, dtype=float).reshape(3)
        return (R @ local_pts.T).T + T

    def _point_signed_distance_to_block(self, p: np.ndarray, b: Block) -> float:
        """Exact signed distance from point to oriented box (negative inside)."""
        R = quat_to_rot(b.quat)
        c = np.asarray(b.position, dtype=float).reshape(3)
        h = 0.5 * np.asarray(b.size, dtype=float).reshape(3)
        q = np.abs(R.T @ (p - c)) - h
        outside = np.linalg.norm(np.maximum(q, 0.0))
        inside = min(max(float(q[0]), float(q[1]), float(q[2])), 0.0)
        return float(outside + inside)

    def sample_sdf_grid(self, bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
                        dims: Tuple[int, int, int]):
        """Sample SDF on a regular grid over bounds with shape dims."""
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
        nx, ny, nz = dims
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        zs = np.linspace(zmin, zmax, nz)
        sdf = np.empty((nx, ny, nz), dtype=float)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                for k, z in enumerate(zs):
                    sdf[i, j, k] = self.signed_distance(np.array([x, y, z], dtype=float))
        return (xs, ys, zs), sdf

    # ---------- Face-based stacking helpers ----------
    # Convention (block local frame):
    # +x: front, -x: back, +y: left, -y: right, +z: top, -z: bottom.

    def _axes_center_half_extents(self, b: Block):
        """Return (R, c, h) where R has columns [x_hat, y_hat, z_hat],
        c is center in world, and h = (hx, hy, hz) half extents.
        """
        R = quat_to_rot(b.quat)
        c = np.asarray(b.position, dtype=float)
        h = 0.5 * np.asarray(b.size, dtype=float)
        return R, c, h

    def get_stack_point_on_face(self,
                                base: IdLike,
                                new_size: Tuple[float, float, float],
                                face: str,
                                gap: float = 0.0,
                                tangential_offset: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        """Compute center position for a new block of size new_size placed against a face of 'base'.

        Args:
            base: object_id (str) or index (int) of the base block.
            new_size: (sx, sy, sz) of the new block.
            face: one of {"top","bottom","front","back","right","left"} in base's local frame.
            gap: extra separation between faces (>0 leaves a gap).
            tangential_offset: offsets along the two tangential axes of the chosen face,
                               expressed in base local coordinates (u, v).

        Returns:
            pos (3,) world coordinates for the new block's center.
        """
        idx = self._index_from_id(base)
        b = self.blocks[idx]
        R, c, h_base = self._axes_center_half_extents(b)
        h_new = 0.5 * np.asarray(new_size, dtype=float)

        face = face.lower()
        # Determine normal axis and sign for the chosen face
        if face == "top":
            n_axis, sign = 2, +1   # +z
            tang_axes = (0, 1)      # x, y
            sep = h_base[2] + h_new[2] + gap
        elif face == "bottom":
            n_axis, sign = 2, -1    # -z
            tang_axes = (0, 1)
            sep = h_base[2] + h_new[2] + gap
        elif face == "front":
            n_axis, sign = 0, +1    # +x
            tang_axes = (1, 2)      # y, z
            sep = h_base[0] + h_new[0] + gap
        elif face == "back":
            n_axis, sign = 0, -1    # -x
            tang_axes = (1, 2)
            sep = h_base[0] + h_new[0] + gap
        elif face == "right":
            n_axis, sign = 1, -1    # -y
            tang_axes = (0, 2)      # x, z
            sep = h_base[1] + h_new[1] + gap
        elif face == "left":
            n_axis, sign = 1, +1    # +y
            tang_axes = (0, 2)
            sep = h_base[1] + h_new[1] + gap
        else:
            raise ValueError("face must be one of: top, bottom, front, back, right, left")

        # World normal and tangential axes
        n_hat = R[:, n_axis] * sign
        u_hat = R[:, tang_axes[0]]
        v_hat = R[:, tang_axes[1]]

        u_off, v_off = tangential_offset
        pos = c + n_hat * sep + u_hat * u_off + v_hat * v_off
        return pos

    # Convenience wrappers
    def get_top_point(self, base: IdLike, new_size, gap: float = 0.0, xy_offset: Tuple[float, float] = (0.0, 0.0)):
        return self.get_stack_point_on_face(base, new_size, face="top", gap=gap, tangential_offset=xy_offset)

    def get_bottom_point(self, base: IdLike, new_size, gap: float = 0.0, xy_offset: Tuple[float, float] = (0.0, 0.0)):
        return self.get_stack_point_on_face(base, new_size, face="bottom", gap=gap, tangential_offset=xy_offset)

    def get_front_point(self, base: IdLike, new_size, gap: float = 0.0, xz_offset: Tuple[float, float] = (0.0, 0.0)):
        return self.get_stack_point_on_face(base, new_size, face="front", gap=gap, tangential_offset=xz_offset)

    def get_back_point(self, base: IdLike, new_size, gap: float = 0.0, xz_offset: Tuple[float, float] = (0.0, 0.0)):
        return self.get_stack_point_on_face(base, new_size, face="back", gap=gap, tangential_offset=xz_offset)

    def get_right_point(self, base: IdLike, new_size, gap: float = 0.0, yz_offset: Tuple[float, float] = (0.0, 0.0)):
        return self.get_stack_point_on_face(base, new_size, face="right", gap=gap, tangential_offset=yz_offset)

    def get_left_point(self, base: IdLike, new_size, gap: float = 0.0, yz_offset: Tuple[float, float] = (0.0, 0.0)):
        return self.get_stack_point_on_face(base, new_size, face="left", gap=gap, tangential_offset=yz_offset)

    # Backward-compatible stackers
    def stack_on(self, base: IdLike, size, xy_offset=(0.0, 0.0), quat=(0.0, 0.0, 0.0, 1.0), gap: float = 0.0,
                 object_id: Optional[str] = None) -> str:
        """Place a new block on the 'top' face of base (arbitrary base orientation supported)."""
        pos = self.get_top_point(base, size, gap=gap, xy_offset=xy_offset)
        return self.add_block(size=size, position=tuple(pos.tolist()), quat=quat, object_id=object_id)

    def stack_on_face(self, base: IdLike, size, face: str, tangential_offset=(0.0, 0.0),
                      quat=(0.0, 0.0, 0.0, 1.0), gap: float = 0.0, object_id: Optional[str] = None) -> str:
        """General face-based stacking."""
        pos = self.get_stack_point_on_face(base, size, face, gap=gap, tangential_offset=tangential_offset)
        return self.add_block(size=size, position=tuple(pos.tolist()), quat=quat, object_id=object_id)
