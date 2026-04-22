"""
Composite rigid body inertia for a PZS-style gripper (three boxes).

Geometry model (Palfinger Epsilon PZS-like):
    - Box 0: central body / housing       (attached to the crane at the origin)
    - Box 1: jaw A (one side)
    - Box 2: jaw B (mirrored copy of jaw A)

World frame:
    Origin at the crane attachment point.
    +z points down along the gripper's main axis (toward the tip / load).
    x-y plane is horizontal; the jaws open/close along x.

What you define per box:
    - mass                   [kg]
    - half-extents (hx,hy,hz) along the box's local axes [m]
    - position of the box center in the world frame [m]
    - orientation as (roll, pitch, yaw) in radians (intrinsic ZYX)

For jaw B you don't define anything — it is auto-mirrored across the x=0 plane.

Outputs:
    - total mass
    - composite CoM in world frame
    - 3x3 inertia tensor about the composite CoM, expressed in the world frame
    - a matplotlib 3D plot with box edges, per-box CoM markers (size scaled by mass),
      and the composite CoM in a distinct color.
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# ---------------------------------------------------------------------------
# Rigid body primitives
# ---------------------------------------------------------------------------


def rotation_matrix(rpy):
    """Intrinsic ZYX (yaw-pitch-roll) rotation matrix.

    rpy = (roll, pitch, yaw)  in radians.
    Returns R such that x_world = R @ x_local + translation.
    """
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


@dataclass
class Box:
    name: str
    mass: float  # kg
    half_extents: np.ndarray  # (hx, hy, hz) in local frame, in meters
    position: np.ndarray  # center in world frame, meters
    rpy: np.ndarray  # (roll, pitch, yaw), radians

    @property
    def R(self):
        return rotation_matrix(self.rpy)

    def inertia_local(self):
        """Inertia tensor of a solid box about its own center, in its local frame."""
        hx, hy, hz = self.half_extents
        # full side lengths
        a, b, c = 2 * hx, 2 * hy, 2 * hz
        m = self.mass
        Ixx = (1.0 / 12.0) * m * (b * b + c * c)
        Iyy = (1.0 / 12.0) * m * (a * a + c * c)
        Izz = (1.0 / 12.0) * m * (a * a + b * b)
        return np.diag([Ixx, Iyy, Izz])

    def inertia_world_about_own_com(self):
        """Inertia tensor about the box center, but expressed in world axes."""
        R = self.R
        return R @ self.inertia_local() @ R.T

    def corners_world(self):
        """Return the 8 corners of the box in world coordinates, shape (8,3)."""
        hx, hy, hz = self.half_extents
        signs = np.array(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
            dtype=float,
        )
        local = signs * np.array([hx, hy, hz])
        return (self.R @ local.T).T + self.position

    def edges_world(self):
        """Return the 12 edges as pairs of world-frame points, shape (12, 2, 3)."""
        C = self.corners_world()
        # corner indexing follows the signs order above:
        # 0: ---  1: --+  2: -+-  3: -++  4: +--  5: +-+  6: ++-  7: +++
        edge_idx = [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),  # along z
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7),  # along y
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # along x
        ]
        return np.array([[C[i], C[j]] for i, j in edge_idx])


def mirror_box_x(box: Box, new_name: str) -> Box:
    """Mirror a box across the x=0 plane, producing a physically-valid twin.

    A pure reflection inverts orientation, so we rebuild a proper rotation
    by flipping the sign of x on the translated center and negating the
    yaw and roll components that would otherwise produce a left-handed frame.
    For the common PZS case (jaws are axis-aligned or only yawed), this
    gives the expected symmetric jaw.
    """
    pos = box.position.copy()
    pos[0] = -pos[0]
    r, p, y = box.rpy
    # reflect rotation across x=0: negate yaw and roll, keep pitch
    mirrored_rpy = np.array([-r, p, -y])
    return Box(
        name=new_name,
        mass=box.mass,
        half_extents=box.half_extents.copy(),
        position=pos,
        rpy=mirrored_rpy,
    )


# ---------------------------------------------------------------------------
# Composite inertia
# ---------------------------------------------------------------------------


def composite_inertia(boxes):
    """Compute total mass, CoM, and inertia about the CoM in world axes."""
    masses = np.array([b.mass for b in boxes])
    positions = np.array([b.position for b in boxes])
    M = masses.sum()
    com = (masses[:, None] * positions).sum(axis=0) / M

    I_total = np.zeros((3, 3))
    for b in boxes:
        I_b = b.inertia_world_about_own_com()
        d = b.position - com  # offset from composite CoM
        # parallel axis theorem: I += m (d.d * I3  -  d d^T)
        I_shift = b.mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        I_total += I_b + I_shift
    return M, com, I_total


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_gripper(boxes, com, inertia, show=True):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    box_colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]

    # mass scaling for per-box CoM markers
    masses = np.array([b.mass for b in boxes])
    m_min, m_max = masses.min(), masses.max()

    def marker_size(m):
        if np.isclose(m_max, m_min):
            return 90.0
        return 40.0 + 220.0 * (m - m_min) / (m_max - m_min)

    # draw box edges
    for i, b in enumerate(boxes):
        edges = b.edges_world()
        lc = Line3DCollection(
            edges, colors=box_colors[i % len(box_colors)], linewidths=1.6, alpha=0.9
        )
        ax.add_collection3d(lc)
        ax.scatter(
            *b.position,
            s=marker_size(b.mass),
            color=box_colors[i % len(box_colors)],
            edgecolor="k",
            linewidth=0.6,
            label=f"{b.name} CoM  ({b.mass:.2f} kg)",
        )

    # composite CoM
    ax.scatter(
        *com,
        s=260,
        color="gold",
        marker="*",
        edgecolor="k",
        linewidth=1.0,
        zorder=10,
        label=f"composite CoM  ({masses.sum():.2f} kg)",
    )

    # crane attachment (origin)
    ax.scatter(
        0, 0, 0, s=80, color="black", marker="x", label="crane attachment (origin)"
    )

    # equal-aspect bounding box
    pts = np.vstack(
        [b.corners_world() for b in boxes] + [com[None, :], np.zeros((1, 3))]
    )
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = (maxs - mins).max() * 0.55 + 1e-6
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)

    # z-down convention: invert z-axis so the gripper hangs naturally
    ax.invert_zaxis()

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]  (down)")
    ax.set_title("PZS-style gripper — composite inertia")
    ax.legend(loc="upper left", fontsize=8)

    # textbox with results
    info = (
        f"total mass : {masses.sum():.3f} kg\n"
        f"CoM (world): [{com[0]:+.4f}, {com[1]:+.4f}, {com[2]:+.4f}] m\n"
        "\nInertia about CoM  [kg m^2]:\n"
        f"  Ixx={inertia[0, 0]:+.4f}  Ixy={inertia[0, 1]:+.4f}  Ixz={inertia[0, 2]:+.4f}\n"
        f"  Iyx={inertia[1, 0]:+.4f}  Iyy={inertia[1, 1]:+.4f}  Iyz={inertia[1, 2]:+.4f}\n"
        f"  Izx={inertia[2, 0]:+.4f}  Izy={inertia[2, 1]:+.4f}  Izz={inertia[2, 2]:+.4f}"
    )
    fig.text(
        0.02,
        0.02,
        info,
        family="monospace",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.9),
    )

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# Example configuration — edit these numbers for your actual gripper
# ---------------------------------------------------------------------------


def build_default_gripper():
    """A placeholder PZS-like geometry. Replace the numbers with your own."""
    # Central body: sits directly below the crane attachment.
    # The top face of this box is at z = 0 (the attachment point).
    body_half = np.array([0.35, 0.35, 0.10])  # 0.24 x 0.30 x 0.40 m
    body = Box(
        name="central body",
        mass=56.0,
        half_extents=body_half,
        position=np.array([0.0, 0.0, body_half[2]]),  # center is hz below origin
        rpy=np.array([0.0, 0.0, 0.0]),
    )

    # Jaw A: hangs below the body, offset in +x. Jaw B will be the mirror.
    jaw_half = np.array([0.05, 0.35, 0.5])  # 0.08 x 0.24 x 0.36 m
    jaw_a = Box(
        name="jaw A (+x)",
        mass=150.0,
        half_extents=jaw_half,
        # centered laterally at +x, sitting just under the body
        position=np.array([0.300, 0.0, 2 * body_half[2] + jaw_half[2]]),
        rpy=np.array([0.0, 0.0, 0.0]),
    )

    jaw_b = mirror_box_x(jaw_a, new_name="jaw B (-x)")
    return [body, jaw_a, jaw_b]


def main():
    boxes = build_default_gripper()
    M, com, I = composite_inertia(boxes)

    print(f"total mass      : {M:.4f} kg")
    print(f"CoM (world)     : {com}")
    print("inertia about CoM (world axes) [kg m^2]:")
    with np.printoptions(precision=5, suppress=True):
        print(I)

    plot_gripper(boxes, com, I)


if __name__ == "__main__":
    main()
