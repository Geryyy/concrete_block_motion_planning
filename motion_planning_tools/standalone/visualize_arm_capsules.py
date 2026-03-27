"""Visualize the CBS crane arm FK and its hppfcl capsule approximation.

Shows the full collision model: K0→K5 arm + K5→K8 tool chain (passive joints
at analytic equilibrium) + PZS100 gripper jaws at the 60 cm gripping angle.

Usage
-----
    python3 visualize_arm_capsules.py [--scenario SCENE] [--config CONFIG]

Arguments
---------
--scenario SCENE
    One of: step_01_first_on_ground, step_02_second_beside_first,
            step_03_third_on_top, step_04_between_two_blocks
    Default: step_04_between_two_blocks

--config hover|goal|both|<5 floats>
    Joint config to visualize.  'hover' and 'goal' use the validated configs
    from scenarios.py.  Pass 5 floats for a custom config:
      --config "0.05 -0.86 0.62 0.65 -0.05"
    Default: both

Examples
--------
    # Step 4, hover + goal side by side
    python3 visualize_arm_capsules.py

    # Step 2 at goal position
    python3 visualize_arm_capsules.py --scenario step_02_second_beside_first --config goal

    # Custom joint config
    python3 visualize_arm_capsules.py --config "0.05 -0.89 0.65 0.55 -0.05"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── path setup ─────────────────────────────────────────────────────────────
_PKG = Path(__file__).resolve().parents[2]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))


# ── capsule definitions (mirror of arm_model.py _CAPSULES) ─────────────────
# Each entry: (p1_frame, p2_frame, radius, label, color)
_CAPSULE_DEFS = [
    # Arm (K0→K5) — actuated
    ("K0_mounting_base",            "K1_slewing_column",       0.15, "Slewing column  (K0→K1, r=0.15 m)", "#4e79a7"),
    ("K1_slewing_column",           "K2_boom",                 0.14, "Boom            (K1→K2, r=0.14 m)", "#f28e2b"),
    ("K2_boom",                     "K3_arm",                  0.10, "Arm elbow       (K2→K3, r=0.10 m)", "#e15759"),
    ("K3_arm",                      "K5_inner_telescope",      0.08, "Telescope       (K3→K5, r=0.08 m)", "#76b7b2"),
    ("K1_boom_cylinder_suspension", "boom_cylinder_piston",    0.06, "Boom cylinder   (r=0.06 m)",         "#59a14f"),
    # Tool chain (K5→K8) — passive joints at equilibrium
    ("K5_inner_telescope",          "K6_double_joint_link",    0.05, "Tip link (K5→K6, r=0.05 m)",        "#b07aa1"),
    ("K6_double_joint_link",        "K8_tool_center_point",    0.08, "Rotator  (K6→K8, r=0.08 m)",        "#9c755f"),
    # PZS100 gripper
    ("K11",                         "K9",                      0.06, "Gripper mount (K11→K9, r=0.06 m)",  "#ff9da7"),
    ("K9",                          "K10_outer_jaw",           0.04, "Outer jaw (K9→K10, r=0.04 m)",      "#edc948"),
    ("K11",                         "K12_inner_jaw",           0.04, "Inner jaw (K11→K12, r=0.04 m)",     "#bab0ac"),
]

# Real CBS concrete block dimensions [m]
_CBS_BLOCK_SIZE = (0.60, 0.60, 0.90)

# ── validated configs (mirror of standalone/scenarios.py) ──────────────────
_HOVER = {
    "step_01_first_on_ground":    (0.05,  -0.80, 0.55, 0.40, -0.05),
    "step_02_second_beside_first":(0.25,  -0.80, 0.55, 0.40, -0.25),
    "step_03_third_on_top":       (0.05,  -0.80, 0.55, 0.40, -0.05),
    "step_04_between_two_blocks": (0.05,  -0.80, 0.55, 0.40, -0.05),
}
_GOAL = {
    "step_01_first_on_ground":    (0.05,  -0.92, 0.68, 0.65, -0.05),
    "step_02_second_beside_first":(0.25,  -0.92, 0.68, 0.65, -0.25),
    "step_03_third_on_top":       (0.05,  -0.86, 0.62, 0.65, -0.05),
    "step_04_between_two_blocks": (0.05,  -0.92, 0.68, 0.65, -0.05),
}


# ── 3-D drawing helpers ────────────────────────────────────────────────────

def _rotation_z_to(d: np.ndarray) -> np.ndarray:
    """Rodrigues: rotation matrix mapping z-axis onto unit vector d."""
    d = np.asarray(d, dtype=float); d /= np.linalg.norm(d)
    z = np.array([0., 0., 1.])
    axis = np.cross(z, d)
    sin_a = float(np.linalg.norm(axis))
    cos_a = float(np.dot(z, d))
    if sin_a < 1e-8:
        return np.eye(3) if cos_a > 0 else np.diag([1., -1., -1.])
    axis /= sin_a
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3) + sin_a*K + (1-cos_a)*(K@K)


def _capsule_mesh(p1: np.ndarray, p2: np.ndarray, radius: float,
                  n_phi: int = 24, n_cap: int = 10):
    """Return (x, y, z) surface arrays for a capsule between p1 and p2."""
    d = p2 - p1
    length = float(np.linalg.norm(d))
    R = _rotation_z_to(d / length)
    mid = 0.5 * (p1 + p2)
    phi = np.linspace(0, 2*np.pi, n_phi)

    def _cyl_ring(z_local):
        pts = radius * np.column_stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)])
        pts[:, 2] = z_local
        return (R @ pts.T).T + mid

    def _hemi(sign, n_cap=n_cap):
        theta = np.linspace(0, np.pi/2, n_cap)
        pts = []
        for th in theta:
            ring_r = radius * np.sin(th)
            ring_z = sign * (length/2 + radius * np.cos(th))
            ring = ring_r * np.column_stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)])
            ring[:, 2] = ring_z
            pts.append((R @ ring.T).T + mid)
        return pts

    # Cylinder body rings
    n_cyl = max(4, int(length / radius * 3))
    z_vals = np.linspace(-length/2, length/2, n_cyl)
    rings = [_cyl_ring(z) for z in z_vals]

    # Hemispheres
    top_rings = _hemi(+1)
    bot_rings = _hemi(-1)
    bot_rings = list(reversed(bot_rings))

    all_rings = bot_rings + rings + top_rings
    xs = np.array([[r[i,0] for i in range(n_phi)] for r in all_rings])
    ys = np.array([[r[i,1] for i in range(n_phi)] for r in all_rings])
    zs = np.array([[r[i,2] for i in range(n_phi)] for r in all_rings])
    return xs, ys, zs


def _box_faces(center, size, quat=(0,0,0,1)):
    """Return list of (4,3) vertex arrays for the 6 faces of an OBB."""
    from motion_planning.geometry.utils import quat_to_rot
    R = quat_to_rot(quat)
    c = np.asarray(center, dtype=float)
    hx, hy, hz = 0.5*np.asarray(size, dtype=float)
    # 8 corners in local frame
    corners_local = np.array([
        [-hx,-hy,-hz],[+hx,-hy,-hz],[+hx,+hy,-hz],[-hx,+hy,-hz],
        [-hx,-hy,+hz],[+hx,-hy,+hz],[+hx,+hy,+hz],[-hx,+hy,+hz],
    ])
    cv = (R @ corners_local.T).T + c
    faces_idx = [
        [0,1,2,3],[4,5,6,7],[0,1,5,4],
        [2,3,7,6],[0,3,7,4],[1,2,6,5],
    ]
    return [cv[f] for f in faces_idx]


def _set_axes_equal(ax):
    """Force equal aspect ratio on a 3-D axes."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = limits[:, 1] - limits[:, 0]
    centers = limits.mean(axis=1)
    max_span = spans.max() / 2
    ax.set_xlim3d([centers[0]-max_span, centers[0]+max_span])
    ax.set_ylim3d([centers[1]-max_span, centers[1]+max_span])
    ax.set_zlim3d([centers[2]-max_span, centers[2]+max_span])


# ── main visualization ─────────────────────────────────────────────────────

def _draw_arm(ax, model, q_map, frame_pos):
    """Draw all capsules and skeleton onto ax. frame_pos is extended in-place."""
    # Ensure all capsule frames are computed
    for p1f, p2f, _, _, _ in _CAPSULE_DEFS:
        for fn in (p1f, p2f):
            if fn not in frame_pos:
                frame_pos[fn] = model._frame_pos(q_map, fn)

    for p1f, p2f, radius, _, color in _CAPSULE_DEFS:
        p1 = frame_pos[p1f]
        p2 = frame_pos[p2f]
        if np.linalg.norm(p2 - p1) < 1e-4:
            continue
        xs, ys, zs = _capsule_mesh(p1, p2, radius)
        ax.plot_surface(xs, ys, zs, alpha=0.22, color=color, linewidth=0, zorder=2)
        ax.plot_wireframe(xs, ys, zs, alpha=0.12, color=color, linewidth=0.35,
                          rstride=5, cstride=5)

    # Arm skeleton K0→K1→K2→K3→K5
    skel_frames = ["K0_mounting_base","K1_slewing_column","K2_boom","K3_arm","K5_inner_telescope"]
    skel_pos = np.array([frame_pos[f] for f in skel_frames])
    ax.plot(skel_pos[:,0], skel_pos[:,1], skel_pos[:,2],
            "k-o", linewidth=2.5, markersize=5, zorder=5)

    # Tool chain skeleton K5→K6→K8
    tool_frames = ["K5_inner_telescope","K6_double_joint_link","K8_tool_center_point"]
    tool_pos = np.array([frame_pos[f] for f in tool_frames])
    ax.plot(tool_pos[:,0], tool_pos[:,1], tool_pos[:,2],
            color="#9c755f", linestyle="--", linewidth=2, marker="o", markersize=4, zorder=5)

    # Gripper skeleton K11→K9→K10 and K11→K12
    for path in [["K11","K9","K10_outer_jaw"], ["K11","K12_inner_jaw"]]:
        pts = np.array([frame_pos[f] for f in path])
        ax.plot(pts[:,0], pts[:,1], pts[:,2],
                color="#edc948", linestyle="-", linewidth=2, marker="s", markersize=3, zorder=5)

    # Frame labels
    label_frames = {
        "K0_mounting_base": "K0", "K1_slewing_column": "K1",
        "K2_boom": "K2", "K3_arm": "K3", "K5_inner_telescope": "K5",
        "K6_double_joint_link": "K6", "K8_tool_center_point": "K8",
        "K9": "K9", "K11": "K11",
    }
    for fname, lbl in label_frames.items():
        pt = frame_pos[fname]
        ax.scatter(*pt, s=45, color="black", zorder=10)
        ax.text(pt[0]+0.05, pt[1]+0.05, pt[2]+0.07, lbl, fontsize=7, fontweight="bold")


def visualize(scenario_name: str, q_act: np.ndarray, title_suffix: str = "") -> None:
    from motion_planning.scenarios import ScenarioLibrary
    from motion_planning.geometry.arm_model import CraneArmCollisionModel, GRIP_ANGLE_60CM

    print(f"Loading scenario '{scenario_name}'...")
    scene_cfg = ScenarioLibrary().build_scenario(scenario_name)
    scene = scene_cfg.scene

    print("Computing FK + analytic equilibrium...")
    model = CraneArmCollisionModel()
    model._ensure_loaded()

    q_map = model.complete_joint_map(q_act, jaw_angle=GRIP_ANGLE_60CM)
    frame_pos: dict[str, np.ndarray] = {}

    d_arm = model.clearance(q_map, scene, ignore_ids=["table"])
    d_pay = model.payload_clearance(
        scene_cfg.goal, scene_cfg.goal_yaw_deg * np.pi / 180,
        _CBS_BLOCK_SIZE, scene,
    )

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")
    _draw_arm(ax, model, q_map, frame_pos)

    for blk in scene.blocks:
        color = "saddlebrown" if blk.object_id == "table" else "gray"
        alpha = 0.35 if blk.object_id == "table" else 0.55
        poly = Poly3DCollection(_box_faces(blk.position, blk.size, blk.quat),
                                alpha=alpha, facecolor=color,
                                edgecolor="dimgray", linewidth=0.6)
        ax.add_collection3d(poly)
        c = np.asarray(blk.position)
        ax.text(c[0], c[1], c[2]+0.06, blk.object_id or "",
                ha="center", fontsize=6.5, color="black")

    goal_quat = (0, 0, np.sin(np.radians(scene_cfg.goal_yaw_deg) / 2),
                      np.cos(np.radians(scene_cfg.goal_yaw_deg) / 2))
    ax.add_collection3d(Poly3DCollection(
        _box_faces(scene_cfg.goal, _CBS_BLOCK_SIZE, goal_quat),
        alpha=0.20, facecolor="limegreen", edgecolor="green", linewidth=1.2))
    gp = np.asarray(scene_cfg.goal)
    ax.text(gp[0], gp[1], gp[2]-0.06, "goal (60×60×90 cm)",
            ha="center", fontsize=7, color="green")

    ax.scatter(0, 0, 0, s=120, marker="^", color="red", zorder=15)
    ax.set_xlabel("X [m] →"); ax.set_ylabel("Y [m] →"); ax.set_zlabel("Z [m] ↑")

    arm_ok = "✓" if d_arm >= 0.01 else "✗"
    pay_ok = "✓" if d_pay >= 0.01 else "✗"
    th6 = q_map["theta6_tip_joint"]
    th7 = q_map["theta7_tilt_joint"]
    ax.set_title(
        f"CBS Crane — {scenario_name}\n"
        f"q_act={np.round(q_act,3).tolist()}  θ6={th6:.3f} θ7={th7:.3f} (equilibrium)\n"
        f"{arm_ok} arm {d_arm*100:.1f} cm   {pay_ok} payload {d_pay*100:.1f} cm",
        fontsize=9,
    )

    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color=c, alpha=0.6, label=lbl)
        for _, _, _, lbl, c in _CAPSULE_DEFS
    ] + [
        mpatches.Patch(color="gray",        alpha=0.6, label="Scene blocks"),
        mpatches.Patch(color="saddlebrown", alpha=0.6, label="Table"),
        mpatches.Patch(color="limegreen",   alpha=0.5, label="Goal (60×60×90 cm)"),
        plt.Line2D([0],[0], color="black",   linewidth=2, marker="o", markersize=5,
                   label="Skeleton K0→K5"),
        plt.Line2D([0],[0], color="#9c755f", linewidth=2, marker="o", markersize=4,
                   linestyle="--", label="Tool chain K5→K8 (equilibrium)"),
        plt.Line2D([0],[0], color="#edc948", linewidth=2, marker="s", markersize=3,
                   label="Gripper jaws"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7, framealpha=0.8)
    _set_axes_equal(ax)
    ax.view_init(elev=20, azim=-50)
    plt.tight_layout()


def visualize_multi(scenario_name: str, configs: list[tuple], labels: list[str]) -> None:
    """Show hover + goal side by side in one figure."""
    from motion_planning.scenarios import ScenarioLibrary
    from motion_planning.geometry.arm_model import CraneArmCollisionModel, GRIP_ANGLE_60CM

    print(f"Loading scenario '{scenario_name}'...")
    scene_cfg = ScenarioLibrary().build_scenario(scenario_name)
    scene = scene_cfg.scene
    model = CraneArmCollisionModel()
    model._ensure_loaded()

    n = len(configs)
    fig = plt.figure(figsize=(7*n, 8))
    fig.suptitle(f"CBS Crane: {scenario_name}", fontsize=11, fontweight="bold")

    for col, (q_act, label) in enumerate(zip(configs, labels)):
        ax = fig.add_subplot(1, n, col+1, projection="3d")
        q_arr = np.asarray(q_act, dtype=float)
        q_map = model.complete_joint_map(q_arr, jaw_angle=GRIP_ANGLE_60CM)
        frame_pos: dict[str, np.ndarray] = {}

        d = model.clearance(q_map, scene, ignore_ids=["table"])
        _draw_arm(ax, model, q_map, frame_pos)

        for blk in scene.blocks:
            color = "saddlebrown" if blk.object_id == "table" else "gray"
            alpha = 0.35 if blk.object_id == "table" else 0.55
            ax.add_collection3d(Poly3DCollection(
                _box_faces(blk.position, blk.size, blk.quat),
                alpha=alpha, facecolor=color, edgecolor="dimgray", linewidth=0.5))

        ax.add_collection3d(Poly3DCollection(
            _box_faces(scene_cfg.goal, _CBS_BLOCK_SIZE),
            alpha=0.18, facecolor="limegreen", edgecolor="green", linewidth=1.0))

        ax.scatter(0, 0, 0, s=100, marker="^", color="red", zorder=15)
        th6 = q_map["theta6_tip_joint"]
        th7 = q_map["theta7_tilt_joint"]
        ok = "✓" if d >= 0.01 else "✗"
        ax.set_title(f"{label}\n{ok} arm {d*100:.1f} cm  θ6={th6:.2f} θ7={th7:.2f}", fontsize=9)
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
        _set_axes_equal(ax)
        ax.view_init(elev=20, azim=-50)

    plt.tight_layout()


# ── CLI ───────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="step_04_between_two_blocks",
                   choices=list(_HOVER.keys()) + ["all"],
                   help="Scenario to visualize (default: step_04_between_two_blocks)")
    p.add_argument("--config", default="both",
                   help="'hover', 'goal', 'both', or 5 floats e.g. '0.05 -0.86 0.62 0.65 -0.05'")
    p.add_argument("--save", metavar="FILE", help="Save figure to FILE instead of showing")
    return p.parse_args()


def main():
    args = _parse_args()

    scenarios = list(_HOVER.keys()) if args.scenario == "all" else [args.scenario]

    for sc in scenarios:
        cfg_str = args.config
        if cfg_str == "both":
            visualize_multi(sc,
                            configs=[_HOVER[sc], _GOAL[sc]],
                            labels=["Hover (start)", "Placement (goal)"])
        elif cfg_str == "hover":
            visualize(sc, np.array(_HOVER[sc]), title_suffix="hover")
        elif cfg_str == "goal":
            visualize(sc, np.array(_GOAL[sc]), title_suffix="goal")
        else:
            try:
                q = np.array([float(v) for v in cfg_str.split()])
                if q.shape != (5,):
                    raise ValueError
            except ValueError:
                print(f"ERROR: --config must be 'hover', 'goal', 'both', or 5 floats. Got: {cfg_str!r}")
                sys.exit(1)
            visualize(sc, q)

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
