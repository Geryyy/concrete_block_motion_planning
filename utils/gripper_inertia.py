"""Composite rigid-body inertia of the PZS gripper, driven by the URDF.

Reads the crane xacro (defaults to the PZS100 configuration used in sim),
walks the kinematic subtree rooted at K8_tool_center_point out to the two
rails, and reports composite mass / CoM / inertia in the K8 frame. Also
draws each link as an equivalent-inertia box so the URDF parametrisation
can be visually sanity-checked against the physical gripper.

Requires a ROS 2 environment with the workspace sourced so xacro can
resolve $(find pzs100_description) and $(find epsilon_crane_description).

Usage:
    python3 gripper_inertia.py                      # PZS100 at q9 = 0
    python3 gripper_inertia.py --q9 0.2             # rails open to 0.2 m
    python3 gripper_inertia.py --urdf path.urdf     # skip xacro, load URDF
"""

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_SCRIPT = Path(__file__).resolve()
_WORKSPACE = _SCRIPT.parents[4]  # .../ros2_baustelle_ws

DEFAULT_XACRO = (
    _WORKSPACE
    / "src/epsilon_crane_description/urdf/timber_loader_AIT.urdf.xacro"
)
DEFAULT_XACRO_ARGS = {
    "gazebo": "true",
    "tool": "pzs100_description",
    "load_ros2_control": "false",
}
DEFAULT_ROOT = "K8_tool_center_point"
DEFAULT_RAILS = ("K10_left_rail", "K12_right_rail")
Q_JOINT_DEFAULTS = {
    "q9_left_rail_joint": 0.0,
    "q11_right_rail_joint": 0.0,
}


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------


def rotation_matrix(rpy):
    """URDF rpy → rotation matrix (extrinsic XYZ = Rz · Ry · Rx)."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def homogeneous(xyz, rpy):
    H = np.eye(4)
    H[:3, :3] = rotation_matrix(rpy)
    H[:3, 3] = xyz
    return H


def prismatic_step(axis, q):
    H = np.eye(4)
    H[:3, 3] = np.asarray(axis, dtype=float) * q
    return H


def revolute_step(axis, q):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    R = np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)
    H = np.eye(4)
    H[:3, :3] = R
    return H


# ---------------------------------------------------------------------------
# URDF parsing
# ---------------------------------------------------------------------------


@dataclass
class LinkInertial:
    name: str
    mass: float
    com_xyz: np.ndarray  # in link frame
    com_rpy: np.ndarray  # rpy of inertial frame expressed in link frame
    I_inertial: np.ndarray  # 3x3, about CoM, in the inertial-origin frame


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: Optional[np.ndarray]


@dataclass
class LinkVisual:
    filename: str      # original "package://..." URI from URDF
    scale: np.ndarray  # 3-vector
    origin_xyz: np.ndarray  # visual origin in link frame
    origin_rpy: np.ndarray


def _floats(text, default):
    if text is None or text.strip() == "":
        return np.array(default, dtype=float)
    return np.array([float(x) for x in text.split()], dtype=float)


def parse_urdf(urdf_str):
    root = ET.fromstring(urdf_str)
    links: dict[str, LinkInertial] = {}
    visuals: dict[str, LinkVisual] = {}
    joints: dict[str, Joint] = {}

    for link in root.findall("link"):
        name = link.get("name")
        inertial = link.find("inertial")
        if inertial is not None:
            mass_elem = inertial.find("mass")
            mass = float(mass_elem.get("value")) if mass_elem is not None else 0.0
            origin = inertial.find("origin")
            com_xyz = _floats(origin.get("xyz") if origin is not None else None, [0, 0, 0])
            com_rpy = _floats(origin.get("rpy") if origin is not None else None, [0, 0, 0])
            I_elem = inertial.find("inertia")
            ixx = float(I_elem.get("ixx", 0))
            ixy = float(I_elem.get("ixy", 0))
            ixz = float(I_elem.get("ixz", 0))
            iyy = float(I_elem.get("iyy", 0))
            iyz = float(I_elem.get("iyz", 0))
            izz = float(I_elem.get("izz", 0))
            links[name] = LinkInertial(
                name=name,
                mass=mass,
                com_xyz=com_xyz,
                com_rpy=com_rpy,
                I_inertial=np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]),
            )

        # capture the first <visual> with a <mesh> for overlay / uniform-density analysis
        for visual in link.findall("visual"):
            geom = visual.find("geometry")
            mesh_elem = geom.find("mesh") if geom is not None else None
            if mesh_elem is None:
                continue
            vorigin = visual.find("origin")
            scale = _floats(mesh_elem.get("scale"), [1, 1, 1])
            if scale.size == 1:
                scale = np.repeat(scale, 3)
            visuals[name] = LinkVisual(
                filename=mesh_elem.get("filename"),
                scale=scale,
                origin_xyz=_floats(vorigin.get("xyz") if vorigin is not None else None, [0, 0, 0]),
                origin_rpy=_floats(vorigin.get("rpy") if vorigin is not None else None, [0, 0, 0]),
            )
            break

    for joint in root.findall("joint"):
        name = joint.get("name")
        jtype = joint.get("type")
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        origin = joint.find("origin")
        origin_xyz = _floats(origin.get("xyz") if origin is not None else None, [0, 0, 0])
        origin_rpy = _floats(origin.get("rpy") if origin is not None else None, [0, 0, 0])
        axis_elem = joint.find("axis")
        axis = _floats(axis_elem.get("xyz"), [1, 0, 0]) if axis_elem is not None else None
        joints[name] = Joint(name, jtype, parent, child, origin_xyz, origin_rpy, axis)

    return links, joints, visuals


def resolve_package_uri(uri: str) -> Optional[Path]:
    """Resolve package://pkg/... to an absolute path, searching ament then src/."""
    if not uri.startswith("package://"):
        p = Path(uri)
        return p if p.exists() else None
    pkg_name, _, relative = uri[len("package://"):].partition("/")
    try:
        from ament_index_python.packages import get_package_share_directory

        candidate = Path(get_package_share_directory(pkg_name)) / relative
        if candidate.exists():
            return candidate
    except Exception:
        pass
    for pkgxml in _WORKSPACE.glob("src/**/package.xml"):
        try:
            r = ET.parse(pkgxml).getroot()
            nm = r.find("name")
            if nm is not None and nm.text == pkg_name:
                candidate = pkgxml.parent / relative
                if candidate.exists():
                    return candidate
        except ET.ParseError:
            continue
    return None


def load_mesh_in_link_frame(visual: LinkVisual):
    """Load the STL, apply scale and visual origin so vertices are in link frame."""
    if not HAS_TRIMESH:
        return None
    path = resolve_package_uri(visual.filename)
    if path is None:
        print(f"WARN: cannot resolve {visual.filename}")
        return None
    try:
        mesh = trimesh.load_mesh(path, force="mesh")
    except Exception as e:
        print(f"WARN: failed to load {path}: {e}")
        return None
    mesh.apply_scale(visual.scale)
    mesh.apply_transform(homogeneous(visual.origin_xyz, visual.origin_rpy))
    return mesh


def uniform_density_inertia(mesh, target_mass: float):
    """Return (com_in_mesh_frame, I_about_com_in_mesh_frame) scaled to target_mass.

    Returns None if the mesh isn't watertight (volume undefined).
    """
    if not mesh.is_watertight:
        return None
    volume = mesh.volume
    if volume <= 0:
        return None
    density = target_mass / volume
    # trimesh computes moment_inertia with density=1 by default; linear in density.
    return mesh.center_mass.copy(), mesh.moment_inertia * density


def run_xacro(path: Path, args: dict[str, str]) -> str:
    cmd = ["xacro", str(path)] + [f"{k}:={v}" for k, v in args.items()]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        sys.exit("xacro executable not found — source your ROS 2 environment first.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"xacro failed:\n{e.stderr}")
    return res.stdout


# ---------------------------------------------------------------------------
# Kinematic tree
# ---------------------------------------------------------------------------


def find_parent_joint(joints: dict[str, Joint], child: str) -> Optional[Joint]:
    """Return the unique joint whose child link is `child`, or None."""
    matches = [j for j in joints.values() if j.child == child]
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"multiple joints claim {child!r} as child")
    return matches[0]


def downstream_path(joints: dict[str, Joint], start: str, end: str) -> list[Joint]:
    children: dict[str, list[Joint]] = {}
    for j in joints.values():
        children.setdefault(j.parent, []).append(j)
    frontier = deque([(start, [])])
    while frontier:
        node, acc = frontier.popleft()
        if node == end:
            return acc
        for j in children.get(node, []):
            frontier.append((j.child, acc + [j]))
    raise RuntimeError(f"no downstream path from {start!r} to {end!r}")


def resolve_transform(path: list[Joint], q_values: dict[str, float]) -> np.ndarray:
    H = np.eye(4)
    for j in path:
        H = H @ homogeneous(j.origin_xyz, j.origin_rpy)
        if j.type == "prismatic":
            H = H @ prismatic_step(j.axis, q_values.get(j.name, 0.0))
        elif j.type in ("revolute", "continuous"):
            H = H @ revolute_step(j.axis, q_values.get(j.name, 0.0))
        # fixed joints contribute only the origin transform
    return H


# ---------------------------------------------------------------------------
# Body (per-link, in root frame)
# ---------------------------------------------------------------------------


@dataclass
class Body:
    name: str
    mass: float
    com_in_root: np.ndarray
    I_com_in_root: np.ndarray   # 3x3, root-frame axes, about own CoM
    R_box_in_root: np.ndarray   # principal-axes rotation (box orientation)
    half_extents: np.ndarray    # equivalent solid-box half-extents


def link_to_body(link: LinkInertial, H_root_to_link: np.ndarray) -> Body:
    # CoM: link frame → root frame
    com_root = (H_root_to_link @ np.append(link.com_xyz, 1.0))[:3]

    # Inertia: inertial-origin frame → link frame → root frame (about CoM)
    R_inert_in_link = rotation_matrix(link.com_rpy)
    I_link = R_inert_in_link @ link.I_inertial @ R_inert_in_link.T
    R_link_in_root = H_root_to_link[:3, :3]
    I_root = R_link_in_root @ I_link @ R_link_in_root.T

    # Equivalent solid-box half-extents along inertia principal axes.
    # For a uniform box with half-extents (a, b, c):
    #   Ix = m/3 (b^2 + c^2), Iy = m/3 (a^2 + c^2), Iz = m/3 (a^2 + b^2)
    eigvals, eigvecs = np.linalg.eigh(I_root)
    if link.mass < 1e-6:
        half = np.full(3, 1e-3)
    else:
        I1, I2, I3 = eigvals
        m = link.mass
        a2 = (3 / (2 * m)) * (I2 + I3 - I1)
        b2 = (3 / (2 * m)) * (I1 + I3 - I2)
        c2 = (3 / (2 * m)) * (I1 + I2 - I3)
        half = np.sqrt(np.clip([a2, b2, c2], 1e-8, None))

    return Body(
        name=link.name,
        mass=link.mass,
        com_in_root=com_root,
        I_com_in_root=I_root,
        R_box_in_root=eigvecs,
        half_extents=half,
    )


def link_to_body_from_mesh_aabb(
    name: str,
    mass: float,
    mesh_in_link: "trimesh.Trimesh",
    H_root_to_link: np.ndarray,
) -> Body:
    """Build a Body from the mesh AABB + URDF mass (uniform-density solid box).

    The box is axis-aligned in the link frame and sized to enclose the mesh;
    CoM sits at the AABB center; inertia follows the solid-box formula for a
    body of the URDF-declared mass.
    """
    aabb_min, aabb_max = mesh_in_link.bounds  # (2, 3) in link frame
    half_ext = (aabb_max - aabb_min) / 2.0
    center_link = (aabb_min + aabb_max) / 2.0

    com_root = (H_root_to_link @ np.append(center_link, 1.0))[:3]

    hx, hy, hz = half_ext
    I_link = np.diag(
        [
            mass / 3.0 * (hy ** 2 + hz ** 2),
            mass / 3.0 * (hx ** 2 + hz ** 2),
            mass / 3.0 * (hx ** 2 + hy ** 2),
        ]
    )
    R_lr = H_root_to_link[:3, :3]
    return Body(
        name=name,
        mass=mass,
        com_in_root=com_root,
        I_com_in_root=R_lr @ I_link @ R_lr.T,
        R_box_in_root=R_lr,
        half_extents=half_ext,
    )


def composite(bodies: list[Body]) -> tuple[float, np.ndarray, np.ndarray]:
    M = sum(b.mass for b in bodies)
    com = sum(b.mass * b.com_in_root for b in bodies) / M
    inertia = np.zeros((3, 3))
    for b in bodies:
        d = b.com_in_root - com
        shift = b.mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        inertia += b.I_com_in_root + shift
    return M, com, inertia


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def box_edges(body: Body) -> np.ndarray:
    signs = np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=float,
    )
    local = signs * body.half_extents
    world = (body.R_box_in_root @ local.T).T + body.com_in_root
    # corners: ---  --+  -+-  -++  +--  +-+  ++-  +++
    edge_idx = [
        (0, 1), (2, 3), (4, 5), (6, 7),  # along c
        (0, 2), (1, 3), (4, 6), (5, 7),  # along b
        (0, 4), (1, 5), (2, 6), (3, 7),  # along a
    ]
    return np.array([[world[i], world[j]] for i, j in edge_idx])


def plot_gripper(
    bodies: list[Body],
    com: np.ndarray,
    inertia: np.ndarray,
    root_name: str,
    attachment_in_root: Optional[np.ndarray] = None,
    attachment_name: Optional[str] = None,
    meshes_in_root: Optional[list[tuple[str, "trimesh.Trimesh"]]] = None,
    source_label: str = "URDF-derived",
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#8c564b"]

    masses = np.array([b.mass for b in bodies])
    m_min = masses.min()
    m_max = max(masses.max(), m_min + 1e-9)

    def marker_size(m):
        return 40.0 + 220.0 * (m - m_min) / (m_max - m_min)

    # Draw meshes first so box edges / markers overlay on top
    mesh_by_name = {n: m for n, m in (meshes_in_root or [])}

    for i, b in enumerate(bodies):
        color = colors[i % len(colors)]
        if b.name in mesh_by_name:
            m = mesh_by_name[b.name]
            triangles = m.vertices[m.faces]
            ax.add_collection3d(
                Poly3DCollection(
                    triangles, alpha=0.12, facecolor=color, edgecolor="none"
                )
            )
        ax.add_collection3d(
            Line3DCollection(box_edges(b), colors=color, linewidths=1.6, alpha=0.9)
        )
        ax.scatter(
            *b.com_in_root,
            s=marker_size(b.mass),
            color=color,
            edgecolor="k",
            linewidth=0.6,
            label=f"{b.name}  ({b.mass:.2f} kg)",
        )

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
    ax.scatter(0, 0, 0, s=80, color="black", marker="x", label=f"{root_name} (TCP)")
    if attachment_in_root is not None:
        ax.scatter(
            *attachment_in_root,
            s=140,
            color="crimson",
            marker="P",
            edgecolor="k",
            linewidth=0.8,
            label=f"attachment: {attachment_name}",
        )
        ax.plot(
            [0, attachment_in_root[0]],
            [0, attachment_in_root[1]],
            [0, attachment_in_root[2]],
            color="crimson",
            linestyle=":",
            linewidth=1.2,
        )

    # equal-aspect bounding (consider mesh vertices too if present)
    extra_pts = [com[None, :], np.zeros((1, 3))]
    if attachment_in_root is not None:
        extra_pts.append(attachment_in_root[None, :])
    for _, m in meshes_in_root or []:
        extra_pts.append(m.vertices)
    pts = np.vstack(
        [box_edges(b).reshape(-1, 3) for b in bodies] + extra_pts
    )
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = (maxs - mins).max() * 0.55 + 1e-6
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.invert_zaxis()

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m] (down)")
    ax.set_title(f"gripper inertia ({source_label}) — frame: {root_name}")
    ax.legend(loc="upper left", fontsize=8)

    info = (
        f"total mass   : {masses.sum():.3f} kg\n"
        f"CoM ({root_name}) : [{com[0]:+.4f}, {com[1]:+.4f}, {com[2]:+.4f}]\n"
        "\nInertia about CoM  [kg m^2]:\n"
        f"  Ixx={inertia[0,0]:+.4f}  Ixy={inertia[0,1]:+.4f}  Ixz={inertia[0,2]:+.4f}\n"
        f"  Iyx={inertia[1,0]:+.4f}  Iyy={inertia[1,1]:+.4f}  Iyz={inertia[1,2]:+.4f}\n"
        f"  Izx={inertia[2,0]:+.4f}  Izy={inertia[2,1]:+.4f}  Izz={inertia[2,2]:+.4f}"
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
    plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xacro", default=str(DEFAULT_XACRO), help="crane xacro file")
    ap.add_argument("--urdf", default=None, help="pre-expanded URDF (overrides --xacro)")
    ap.add_argument("--tool", default=DEFAULT_XACRO_ARGS["tool"], help="tool:= xacro arg")
    ap.add_argument("--q9", type=float, default=0.0, help="q9 displacement [m]")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--rails", nargs="+", default=list(DEFAULT_RAILS))
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument(
        "--from-mesh",
        action="store_true",
        help=(
            "derive each body's bounding box, CoM and inertia from its visual "
            "mesh (AABB + URDF mass, solid-box formula) instead of from the URDF "
            "<inertia> block"
        ),
    )
    args = ap.parse_args()

    if args.urdf:
        urdf_str = Path(args.urdf).read_text()
    else:
        xacro_args = dict(DEFAULT_XACRO_ARGS)
        xacro_args["tool"] = args.tool
        urdf_str = run_xacro(Path(args.xacro), xacro_args)

    links, joints, visuals = parse_urdf(urdf_str)

    if args.root not in links:
        sys.exit(f"root link {args.root!r} has no <inertial> in URDF")

    q_values = dict(Q_JOINT_DEFAULTS)
    q_values["q9_left_rail_joint"] = args.q9
    q_values["q11_right_rail_joint"] = args.q9  # URDF mimic is 1:1

    # link name -> transform from root to that link (for mesh overlay + CoM compare)
    H_root_to_link: dict[str, np.ndarray] = {args.root: np.eye(4)}

    def _make_body(name: str, H_rl: np.ndarray) -> Optional[Body]:
        """Build a Body from mesh AABB (if --from-mesh) else from URDF inertia."""
        if args.from_mesh:
            if not HAS_TRIMESH:
                sys.exit("--from-mesh needs `trimesh` installed")
            if name not in visuals:
                print(f"WARN: {name} has no visual mesh; falling back to URDF inertia")
            else:
                mesh_in_link = load_mesh_in_link_frame(visuals[name])
                if mesh_in_link is not None:
                    return link_to_body_from_mesh_aabb(
                        name, links[name].mass, mesh_in_link, H_rl
                    )
                print(f"WARN: mesh for {name} failed to load; falling back to URDF inertia")
        return link_to_body(links[name], H_rl)

    bodies = [_make_body(args.root, np.eye(4))]
    for rail in args.rails:
        if rail not in links:
            print(f"WARN: {rail} not in URDF, skipping")
            continue
        path = downstream_path(joints, args.root, rail)
        H_rl = resolve_transform(path, q_values)
        H_root_to_link[rail] = H_rl
        bodies.append(_make_body(rail, H_rl))

    M, com, inertia = composite(bodies)

    # Attachment frame = parent link of `root` via its mount joint, if any.
    attach_pos = None
    attach_name = None
    parent_j = find_parent_joint(joints, args.root)
    if parent_j is not None:
        H_parent_to_root = homogeneous(parent_j.origin_xyz, parent_j.origin_rpy)
        H_root_to_parent = np.linalg.inv(H_parent_to_root)
        attach_pos = H_root_to_parent[:3, 3]
        attach_name = parent_j.parent
        com_in_attach = (H_root_to_parent @ np.append(com, 1.0))[:3]

    source_desc = "mesh AABB + URDF mass" if args.from_mesh else "URDF <inertia> blocks"
    print(f"source       : {source_desc}")
    print(f"q9           : {args.q9:.4f} m")
    print(f"total mass   : {M:.4f} kg")
    print(f"CoM ({args.root}) : [{com[0]:+.6f}, {com[1]:+.6f}, {com[2]:+.6f}]")
    if attach_pos is not None:
        print(
            f"attachment   : {attach_name} at {attach_pos} in {args.root} "
            f"(via joint {parent_j.name})"
        )
        print(
            f"CoM ({attach_name}) : "
            f"[{com_in_attach[0]:+.6f}, {com_in_attach[1]:+.6f}, {com_in_attach[2]:+.6f}]"
        )
    print(f"inertia about CoM ({args.root} axes) [kg m^2]:")
    with np.printoptions(precision=5, suppress=True):
        print(inertia)

    # Per-link report, plus mesh-based uniform-density check for each
    meshes_in_root: list[tuple[str, "trimesh.Trimesh"]] = []
    for b in bodies:
        print(f"\n-- {b.name} --")
        print(f"   mass          : {b.mass:.3f} kg")
        print(f"   CoM in root   : {b.com_in_root}")
        print(f"   box half-ext. : {b.half_extents}  (principal-axes frame)")

        if b.name not in visuals:
            continue
        if not HAS_TRIMESH:
            print("   (trimesh not installed — skipping mesh analysis)")
            continue
        mesh_link = load_mesh_in_link_frame(visuals[b.name])
        if mesh_link is None:
            continue

        print(f"   mesh file     : {visuals[b.name].filename}")
        aabb_lo, aabb_hi = mesh_link.bounds
        print(f"   mesh AABB link: min {aabb_lo}, max {aabb_hi}")

        ud = uniform_density_inertia(mesh_link, b.mass)
        if ud is None:
            print("   (mesh not watertight — no uniform-density inertia)")
        else:
            com_mesh_link, I_mesh_link = ud
            # rotate into root frame for fair comparison against URDF values
            R_lr = H_root_to_link[b.name][:3, :3]
            com_mesh_root = (
                H_root_to_link[b.name] @ np.append(com_mesh_link, 1.0)
            )[:3]
            I_mesh_root = R_lr @ I_mesh_link @ R_lr.T
            print(f"   mesh CoM root : {com_mesh_root}")
            print(f"   URDF CoM root : {b.com_in_root}")
            print(f"   Δ CoM         : {com_mesh_root - b.com_in_root}")
            with np.printoptions(precision=4, suppress=True):
                print(f"   URDF inertia (root-frame, about URDF CoM):\n{b.I_com_in_root}")
                print(
                    "   mesh inertia (uniform density, "
                    f"ρ = {b.mass / mesh_link.volume:.2f} kg/m³, about mesh CoM):\n"
                    f"{I_mesh_root}"
                )

        # Transform mesh vertices to root frame for plotting
        mesh_root = mesh_link.copy()
        mesh_root.apply_transform(H_root_to_link[b.name])
        meshes_in_root.append((b.name, mesh_root))

    if not args.no_plot:
        plot_gripper(
            bodies,
            com,
            inertia,
            root_name=args.root,
            attachment_in_root=attach_pos,
            attachment_name=attach_name,
            meshes_in_root=meshes_in_root if meshes_in_root else None,
            source_label=("mesh-AABB-derived" if args.from_mesh else "URDF-derived"),
        )


if __name__ == "__main__":
    main()
