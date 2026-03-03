from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin


@dataclass(frozen=True)
class UrdfInertial:
    mass: float
    com_link: np.ndarray
    inertia_com_link: np.ndarray


@dataclass(frozen=True)
class _FixedEdge:
    parent: str
    child: str
    R_pc: np.ndarray
    p_pc: np.ndarray


def _rpy_to_R(rpy: tuple[float, float, float]) -> np.ndarray:
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return Rz @ Ry @ Rx


def _parse_xyz(text: str | None, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=float)
    vals = [float(v) for v in text.strip().split()]
    if len(vals) != 3:
        raise ValueError(f"Expected xyz with 3 values, got: {text}")
    return np.asarray(vals, dtype=float)


def _parse_rpy(text: str | None) -> np.ndarray:
    if not text:
        return np.eye(3, dtype=float)
    vals = [float(v) for v in text.strip().split()]
    if len(vals) != 3:
        raise ValueError(f"Expected rpy with 3 values, got: {text}")
    return _rpy_to_R((vals[0], vals[1], vals[2]))


def _parse_urdf_inertials(urdf_path: Path) -> dict[str, UrdfInertial]:
    root = ET.parse(urdf_path).getroot()
    inertials: dict[str, UrdfInertial] = {}
    for link in root.findall("link"):
        name = str(link.attrib["name"])
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_el = inertial.find("mass")
        inertia_el = inertial.find("inertia")
        if mass_el is None or inertia_el is None:
            continue
        origin = inertial.find("origin")
        p = _parse_xyz(origin.attrib.get("xyz") if origin is not None else None)
        R = _parse_rpy(origin.attrib.get("rpy") if origin is not None else None)
        mass = float(mass_el.attrib["value"])
        ixx = float(inertia_el.attrib.get("ixx", "0"))
        iyy = float(inertia_el.attrib.get("iyy", "0"))
        izz = float(inertia_el.attrib.get("izz", "0"))
        ixy = float(inertia_el.attrib.get("ixy", "0"))
        ixz = float(inertia_el.attrib.get("ixz", "0"))
        iyz = float(inertia_el.attrib.get("iyz", "0"))
        I_inertial = np.array(
            [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]],
            dtype=float,
        )
        I_link = R @ I_inertial @ R.T
        inertials[name] = UrdfInertial(mass=mass, com_link=p, inertia_com_link=I_link)
    return inertials


def _parse_urdf_fixed_edges(urdf_path: Path) -> dict[str, list[_FixedEdge]]:
    root = ET.parse(urdf_path).getroot()
    out: dict[str, list[_FixedEdge]] = {}
    for joint in root.findall("joint"):
        if joint.attrib.get("type") != "fixed":
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        p_name = str(parent.attrib["link"])
        c_name = str(child.attrib["link"])
        origin = joint.find("origin")
        p_pc = _parse_xyz(origin.attrib.get("xyz") if origin is not None else None)
        R_pc = _parse_rpy(origin.attrib.get("rpy") if origin is not None else None)
        out.setdefault(p_name, []).append(_FixedEdge(parent=p_name, child=c_name, R_pc=R_pc, p_pc=p_pc))
    return out


def _resolve_inertial_for_body(
    body_name: str,
    inertials: dict[str, UrdfInertial],
    fixed_edges: dict[str, list[_FixedEdge]],
    max_depth: int = 16,
) -> UrdfInertial | None:
    if body_name in inertials:
        return inertials[body_name]

    # Traverse fixed edges and propagate child's inertial into the parent frame.
    # Supports chains like moved_with_q* -> dh_trans* -> K*.
    stack: list[tuple[str, np.ndarray, np.ndarray, int]] = [(body_name, np.eye(3), np.zeros(3), 0)]
    seen: set[str] = set()
    while stack:
        cur, R_acc, p_acc, depth = stack.pop()
        if cur in seen or depth > max_depth:
            continue
        seen.add(cur)

        if cur in inertials:
            src = inertials[cur]
            com = p_acc + R_acc @ src.com_link
            I = R_acc @ src.inertia_com_link @ R_acc.T
            return UrdfInertial(mass=src.mass, com_link=com, inertia_com_link=I)

        for edge in fixed_edges.get(cur, []):
            R_next = R_acc @ edge.R_pc
            p_next = p_acc + R_acc @ edge.p_pc
            stack.append((edge.child, R_next, p_next, depth + 1))
    return None


def compile_urdf_to_mjcf(urdf_path: str | Path, output_mjcf_path: str | Path) -> Path:
    import mujoco

    urdf = Path(urdf_path).expanduser().resolve()
    out = Path(output_mjcf_path).expanduser().resolve()
    if not urdf.exists():
        raise FileNotFoundError(f"URDF not found: {urdf}")
    out.parent.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(urdf))
    mujoco.mj_saveLastXML(str(out), model)
    return out


def synchronize_mjcf_inertials_from_urdf(
    urdf_path: str | Path,
    mjcf_path: str | Path,
    output_mjcf_path: str | Path | None = None,
    *,
    update_existing: bool = True,
) -> Path:
    urdf = Path(urdf_path).expanduser().resolve()
    mjcf = Path(mjcf_path).expanduser().resolve()
    out = Path(output_mjcf_path).expanduser().resolve() if output_mjcf_path else mjcf

    if not urdf.exists():
        raise FileNotFoundError(f"URDF not found: {urdf}")
    if not mjcf.exists():
        raise FileNotFoundError(f"MJCF not found: {mjcf}")

    inertials = _parse_urdf_inertials(urdf)
    fixed_edges = _parse_urdf_fixed_edges(urdf)

    tree = ET.parse(mjcf)
    root = tree.getroot()
    updated = 0

    for body in root.findall(".//body"):
        bname = body.attrib.get("name")
        if not bname:
            continue
        target = _resolve_inertial_for_body(bname, inertials, fixed_edges)
        if target is None:
            continue

        inertial_el = body.find("inertial")
        if inertial_el is None:
            inertial_el = ET.SubElement(body, "inertial")
        elif not update_existing:
            continue

        I = target.inertia_com_link
        inertial_el.attrib["mass"] = f"{target.mass:.12g}"
        inertial_el.attrib["pos"] = " ".join(f"{v:.12g}" for v in target.com_link.tolist())
        inertial_el.attrib["fullinertia"] = " ".join(
            f"{v:.12g}"
            for v in [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
        )
        # Avoid mixed specifications.
        inertial_el.attrib.pop("diaginertia", None)
        inertial_el.attrib.pop("quat", None)
        updated += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding="utf-8", xml_declaration=False)
    if updated == 0:
        raise RuntimeError("No MJCF body inertials were updated from URDF.")
    return out


def _pin_body_inertials_from_mjcf(mjcf_path: str | Path) -> dict[str, tuple[float, np.ndarray, np.ndarray]]:
    model = pin.buildModelFromMJCF(str(Path(mjcf_path).expanduser().resolve()))
    out: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    for frame in model.frames:
        if int(frame.type) != int(pin.FrameType.BODY):
            continue
        jid = int(frame.parentJoint)
        if jid <= 0 or jid >= model.njoints:
            continue
        I = model.inertias[jid]
        diag = np.array([I.inertia[0, 0], I.inertia[1, 1], I.inertia[2, 2]], dtype=float)
        out[str(frame.name)] = (float(I.mass), np.asarray(I.lever, dtype=float), diag)
    return out


def compare_urdf_inertials_to_mjcf(
    urdf_path: str | Path,
    mjcf_path: str | Path,
) -> dict[str, float]:
    urdf = Path(urdf_path).expanduser().resolve()
    mjcf = Path(mjcf_path).expanduser().resolve()
    inertials = _parse_urdf_inertials(urdf)
    fixed_edges = _parse_urdf_fixed_edges(urdf)
    pin_mj = _pin_body_inertials_from_mjcf(mjcf)

    common = [name for name in pin_mj if _resolve_inertial_for_body(name, inertials, fixed_edges) is not None]
    if not common:
        raise RuntimeError("No common inertial bodies between URDF and MJCF.")

    dmass = []
    dcom = []
    didiag = []
    for name in common:
        u = _resolve_inertial_for_body(name, inertials, fixed_edges)
        assert u is not None
        mass_mj, com_mj, diag_mj = pin_mj[name]
        diag_u = np.array(
            [u.inertia_com_link[0, 0], u.inertia_com_link[1, 1], u.inertia_com_link[2, 2]],
            dtype=float,
        )
        dmass.append(abs(u.mass - mass_mj))
        dcom.append(float(np.linalg.norm(u.com_link - com_mj)))
        didiag.append(float(np.linalg.norm(diag_u - diag_mj)))
    return {
        "num_common_bodies": float(len(common)),
        "mean_abs_mass_diff": float(np.mean(dmass)),
        "max_abs_mass_diff": float(np.max(dmass)),
        "mean_com_diff": float(np.mean(dcom)),
        "max_com_diff": float(np.max(dcom)),
        "mean_inertia_diag_diff": float(np.mean(didiag)),
        "max_inertia_diag_diff": float(np.max(didiag)),
    }


def _sample_common_state(pin_a: pin.Model, pin_b: pin.Model, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[str]]:
    a_names = {str(pin_a.names[j]): int(pin_a.joints[j].idx_v) for j in range(1, pin_a.njoints)}
    b_names = {str(pin_b.names[j]): int(pin_b.joints[j].idx_v) for j in range(1, pin_b.njoints)}
    common = sorted(set(a_names).intersection(b_names), key=lambda n: a_names[n])
    if len(common) != pin_a.nv or len(common) != pin_b.nv:
        raise RuntimeError("Models do not share the same 1-DoF joint name set.")

    qa = np.zeros(pin_a.nv, dtype=float)
    qda = np.zeros(pin_a.nv, dtype=float)
    for jn in common:
        ia = a_names[jn]
        lo = float(pin_a.lowerPositionLimit[ia])
        hi = float(pin_a.upperPositionLimit[ia])
        if np.isfinite(lo) and np.isfinite(hi):
            qa[ia] = float(rng.uniform(0.8 * lo, 0.8 * hi))
        else:
            qa[ia] = float(rng.uniform(-1.0, 1.0))
        vmax = float(pin_a.velocityLimit[ia])
        if not np.isfinite(vmax):
            vmax = 1.0
        qda[ia] = float(rng.uniform(-0.3 * vmax, 0.3 * vmax))
    return qa, qda, common


def compare_pin_models_kinematics(
    urdf_path: str | Path,
    mjcf_path: str | Path,
    *,
    samples: int = 20,
    seed: int = 0,
) -> dict[str, float]:
    model_u = pin.buildModelFromUrdf(str(Path(urdf_path).expanduser().resolve()))
    model_m = pin.buildModelFromMJCF(str(Path(mjcf_path).expanduser().resolve()))
    data_u = model_u.createData()
    data_m = model_m.createData()

    rng = np.random.default_rng(seed)
    pos_err = []
    rot_err = []
    frames_u = {str(f.name): i for i, f in enumerate(model_u.frames) if int(f.type) == int(pin.FrameType.BODY)}
    frames_m = {str(f.name): i for i, f in enumerate(model_m.frames) if int(f.type) == int(pin.FrameType.BODY)}
    common_frames = sorted(set(frames_u).intersection(frames_m))
    if not common_frames:
        raise RuntimeError("No common body frame names for kinematic comparison.")

    for _ in range(samples):
        q, dq, _ = _sample_common_state(model_u, model_m, rng)
        q_u = np.asarray(pin.neutral(model_u), dtype=float)
        q_m = np.asarray(pin.neutral(model_m), dtype=float)
        for jid in range(1, model_u.njoints):
            iq = int(model_u.joints[jid].idx_q)
            iv = int(model_u.joints[jid].idx_v)
            q_u[iq] = q[iv]
        for jid in range(1, model_m.njoints):
            iq = int(model_m.joints[jid].idx_q)
            iv = int(model_m.joints[jid].idx_v)
            q_m[iq] = q[iv]
        pin.forwardKinematics(model_u, data_u, q_u, dq)
        pin.forwardKinematics(model_m, data_m, q_m, dq)
        pin.updateFramePlacements(model_u, data_u)
        pin.updateFramePlacements(model_m, data_m)
        for fn in common_frames:
            Mu = data_u.oMf[frames_u[fn]]
            Mm = data_m.oMf[frames_m[fn]]
            pos_err.append(float(np.linalg.norm(np.asarray(Mu.translation) - np.asarray(Mm.translation))))
            R = np.asarray(Mu.rotation).T @ np.asarray(Mm.rotation)
            c = float(np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0))
            rot_err.append(float(np.arccos(c)))
    return {
        "num_samples": float(samples),
        "num_frames": float(len(common_frames)),
        "mean_pos_err_m": float(np.mean(pos_err)),
        "max_pos_err_m": float(np.max(pos_err)),
        "mean_rot_err_rad": float(np.mean(rot_err)),
        "max_rot_err_rad": float(np.max(rot_err)),
    }


def compare_pin_models_dynamics(
    urdf_path: str | Path,
    mjcf_path: str | Path,
    *,
    samples: int = 20,
    seed: int = 0,
) -> dict[str, float]:
    model_u = pin.buildModelFromUrdf(str(Path(urdf_path).expanduser().resolve()))
    model_m = pin.buildModelFromMJCF(str(Path(mjcf_path).expanduser().resolve()))
    data_u = model_u.createData()
    data_m = model_m.createData()

    rng = np.random.default_rng(seed)
    M_diff = []
    h_diff = []
    for _ in range(samples):
        q, dq, _ = _sample_common_state(model_u, model_m, rng)
        q_u = np.asarray(pin.neutral(model_u), dtype=float)
        q_m = np.asarray(pin.neutral(model_m), dtype=float)
        for jid in range(1, model_u.njoints):
            iq = int(model_u.joints[jid].idx_q)
            iv = int(model_u.joints[jid].idx_v)
            q_u[iq] = q[iv]
        for jid in range(1, model_m.njoints):
            iq = int(model_m.joints[jid].idx_q)
            iv = int(model_m.joints[jid].idx_v)
            q_m[iq] = q[iv]

        Mu = pin.crba(model_u, data_u, q_u)
        Mm = pin.crba(model_m, data_m, q_m)
        Mu = 0.5 * (Mu + Mu.T)
        Mm = 0.5 * (Mm + Mm.T)
        hu = pin.rnea(model_u, data_u, q_u, dq, np.zeros(model_u.nv, dtype=float))
        hm = pin.rnea(model_m, data_m, q_m, dq, np.zeros(model_m.nv, dtype=float))

        M_diff.append(float(np.linalg.norm(Mu - Mm)))
        h_diff.append(float(np.linalg.norm(hu - hm)))
    return {
        "num_samples": float(samples),
        "mean_M_diff": float(np.mean(M_diff)),
        "max_M_diff": float(np.max(M_diff)),
        "mean_h_diff": float(np.mean(h_diff)),
        "max_h_diff": float(np.max(h_diff)),
    }
