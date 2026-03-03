from __future__ import annotations

from typing import Mapping

import numpy as np


def joint_bounds(pin_model, joint_name: str) -> tuple[float, float]:
    jid = int(pin_model.getJointId(joint_name))
    j = pin_model.joints[jid]
    if int(j.nq) == 1:
        iq = int(j.idx_q)
        lo = float(pin_model.lowerPositionLimit[iq])
        hi = float(pin_model.upperPositionLimit[iq])
        return lo, hi
    if int(j.nq) == 2 and int(j.nv) == 1:
        return -np.inf, np.inf
    raise ValueError(f"Unsupported joint representation nq={j.nq}, nv={j.nv} for '{joint_name}'.")


def q_map_to_pin_q(pin_model, q_values: Mapping[str, float], pin_module) -> np.ndarray:
    q_pin = np.asarray(pin_module.neutral(pin_model), dtype=float)
    for jn, val in q_values.items():
        if not pin_model.existJointName(jn):
            continue
        jid = int(pin_model.getJointId(jn))
        j = pin_model.joints[jid]
        nq = int(j.nq)
        iq = int(j.idx_q)
        if nq == 1:
            q_pin[iq] = float(val)
        elif nq == 2 and int(j.nv) == 1:
            q_pin[iq] = float(np.cos(val))
            q_pin[iq + 1] = float(np.sin(val))
        else:
            raise ValueError(f"Unsupported joint representation nq={j.nq}, nv={j.nv} for '{jn}'.")
    return q_pin


def sample_dynamic_q_within_limits(pin_model, dynamic_joint_names: list[str], rng: np.random.Generator) -> dict[str, float]:
    q: dict[str, float] = {}
    for jn in dynamic_joint_names:
        jid = int(pin_model.getJointId(jn))
        joint = pin_model.joints[jid]
        nq = int(joint.nq)
        if nq == 1:
            iq = int(joint.idx_q)
            lo = float(pin_model.lowerPositionLimit[iq])
            hi = float(pin_model.upperPositionLimit[iq])
            if np.isfinite(lo) and np.isfinite(hi):
                q[jn] = float(rng.uniform(lo, hi))
            else:
                q[jn] = float(rng.uniform(-np.pi, np.pi))
        elif nq == 2 and int(joint.nv) == 1:
            q[jn] = float(rng.uniform(-np.pi, np.pi))
        else:
            raise ValueError(f"Unsupported joint representation for sampling: {jn} (nq={joint.nq}, nv={joint.nv})")
    return q


def frame_id(pin_model, frame_name: str, cache: dict[str, int]) -> int:
    if frame_name == "world":
        return -1
    cached = cache.get(frame_name)
    if cached is not None:
        return cached
    if pin_model.existFrame(frame_name):
        fid = int(pin_model.getFrameId(frame_name))
        cache[frame_name] = fid
        return fid
    raise KeyError(
        f"Frame '{frame_name}' not found. Available frame names include: "
        f"{[f.name for f in pin_model.frames[:20]]} ..."
    )


def fk_homogeneous(
    *,
    pin_model,
    pin_data,
    pin_module,
    q_values: Mapping[str, float],
    base_frame: str,
    end_frame: str,
    frame_cache: dict[str, int],
) -> np.ndarray:
    q_pin = q_map_to_pin_q(pin_model, q_values, pin_module)
    pin_module.forwardKinematics(pin_model, pin_data, q_pin)
    pin_module.updateFramePlacements(pin_model, pin_data)
    base_id = frame_id(pin_model, base_frame, frame_cache)
    end_id = frame_id(pin_model, end_frame, frame_cache)
    oMb = pin_module.SE3.Identity() if base_id < 0 else pin_data.oMf[base_id]
    oMe = pin_module.SE3.Identity() if end_id < 0 else pin_data.oMf[end_id]
    bMe = oMb.inverse() * oMe
    return np.asarray(bMe.homogeneous, dtype=float)
