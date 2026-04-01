#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin


def _repo_pythonpath() -> tuple[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    pkg_root = repo_root
    scripts_root = repo_root / "scripts"
    return str(pkg_root), str(scripts_root)


pkg_root, scripts_root = _repo_pythonpath()
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)
if scripts_root not in sys.path:
    sys.path.insert(0, scripts_root)

from motion_planning.mechanics.analytic import AnalyticModelConfig, CraneSteadyState, ModelDescription
from motion_planning.mechanics.analytic.pinocchio_utils import q_map_to_pin_q
from motion_planning.standalone.scenarios import make_default_scenarios


def _wrap_to_pi(angle: float) -> float:
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def _phi_tool_from_transform(T: np.ndarray) -> float:
    T_arr = np.asarray(T, dtype=float).reshape(4, 4)
    return float(math.atan2(T_arr[1, 1], T_arr[0, 1]))


def _fk_transform_pin(desc: ModelDescription, q_map: dict[str, float], *, base_frame: str, end_frame: str) -> np.ndarray:
    q_pin = q_map_to_pin_q(desc.model, q_map, pin)
    data = desc.model.createData()
    pin.forwardKinematics(desc.model, data, q_pin)
    pin.updateFramePlacements(desc.model, data)
    fid_base = desc.model.getFrameId(base_frame)
    fid_end = desc.model.getFrameId(end_frame)
    return np.asarray((data.oMf[fid_base].inverse() * data.oMf[fid_end]).homogeneous, dtype=float)


def _timber_geometry_from_urdf(urdf_path: Path) -> dict[str, float]:
    root = ET.fromstring(urdf_path.read_text(encoding="utf-8"))
    joints = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        if name.startswith("dh_trans"):
            origin = joint.find("origin")
            if origin is None:
                continue
            xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
            joints[name] = xyz
    return {
        "a1": float(joints["dh_trans1"][0]),
        "d1": float(joints["dh_trans1"][2]),
        "a2": float(joints["dh_trans2"][0]),
        "a3": float(joints["dh_trans3"][0]),
        "d4": float(joints["dh_trans4"][2]),
        "a6": float(joints["dh_trans6"][0]),
        "d8": float(joints["dh_trans8"][2]),
    }


def _timber_transform_0_8(q_dyn: dict[str, float], geometry: dict[str, float]) -> np.ndarray:
    theta1 = float(q_dyn["theta1_slewing_joint"])
    theta2 = float(q_dyn["theta2_boom_joint"])
    theta3 = float(q_dyn["theta3_arm_joint"])
    q4 = float(q_dyn["q4_big_telescope"])
    theta6 = float(q_dyn["theta6_tip_joint"])
    theta7 = float(q_dyn["theta7_tilt_joint"])
    theta8 = float(q_dyn["theta8_rotator_joint"])

    a1 = float(geometry["a1"])
    a2 = float(geometry["a2"])
    a3 = float(geometry["a3"])
    d1 = float(geometry["d1"])
    d4 = float(geometry["d4"])
    a6 = float(geometry["a6"])
    d8 = float(geometry["d8"])

    H = np.zeros((4, 4), dtype=float)
    H[0, 0] = (
        (-math.cos(theta7) * ((math.sin(theta6) * math.sin(theta3) - math.cos(theta6) * math.cos(theta3)) * math.cos(theta2)
        + math.sin(theta2) * (math.cos(theta6) * math.sin(theta3) + math.sin(theta6) * math.cos(theta3))) * math.cos(theta8)
        + math.sin(theta8) * ((math.cos(theta6) * math.sin(theta3) + math.sin(theta6) * math.cos(theta3)) * math.cos(theta2)
        + math.sin(theta2) * (math.cos(theta6) * math.cos(theta3) - math.sin(theta6) * math.sin(theta3)))) * math.cos(theta1)
        - math.cos(theta8) * math.sin(theta1) * math.sin(theta7)
    )
    H[0, 1] = (
        (math.cos(theta7) * ((math.sin(theta6) * math.sin(theta3) - math.cos(theta6) * math.cos(theta3)) * math.cos(theta2)
        + math.sin(theta2) * (math.cos(theta6) * math.sin(theta3) + math.sin(theta6) * math.cos(theta3))) * math.sin(theta8)
        + math.cos(theta8) * ((math.cos(theta6) * math.sin(theta3) + math.sin(theta6) * math.cos(theta3)) * math.cos(theta2)
        + math.sin(theta2) * (math.cos(theta6) * math.cos(theta3) - math.sin(theta6) * math.sin(theta3)))) * math.cos(theta1)
        + math.sin(theta8) * math.sin(theta1) * math.sin(theta7)
    )
    H[0, 2] = (
        math.sin(theta7) * ((math.sin(theta6) * math.sin(theta3) - math.cos(theta6) * math.cos(theta3)) * math.cos(theta2)
        + math.sin(theta2) * (math.cos(theta6) * math.sin(theta3) + math.sin(theta6) * math.cos(theta3))) * math.cos(theta1)
        - math.sin(theta1) * math.cos(theta7)
    )
    H[1, 0] = (
        (-math.cos(theta7) * ((math.sin(theta6) * math.sin(theta3) - math.cos(theta6) * math.cos(theta3)) * math.cos(theta2)
        + math.sin(theta2) * (math.cos(theta6) * math.sin(theta3) + math.sin(theta6) * math.cos(theta3))) * math.cos(theta8)
        + math.sin(theta8) * ((math.cos(theta6) * math.sin(theta3) + math.sin(theta6) * math.cos(theta3)) * math.cos(theta2)
        + math.sin(theta2) * (math.cos(theta6) * math.cos(theta3) - math.sin(theta6) * math.sin(theta3)))) * math.sin(theta1)
        + math.cos(theta8) * math.cos(theta1) * math.sin(theta7)
    )
    H[1, 1] = (
        (math.cos(theta7) * ((math.sin(theta6) * math.sin(theta3) - math.cos(theta6) * math.cos(theta3)) * math.cos(theta2)
        + math.sin(theta2) * (math.cos(theta6) * math.sin(theta3) + math.sin(theta6) * math.cos(theta3))) * math.sin(theta8)
        + math.cos(theta8) * ((math.cos(theta6) * math.sin(theta3) + math.sin(theta6) * math.cos(theta3)) * math.cos(theta2)
        + math.sin(theta2) * (math.cos(theta6) * math.cos(theta3) - math.sin(theta6) * math.sin(theta3)))) * math.sin(theta1)
        - math.sin(theta8) * math.cos(theta1) * math.sin(theta7)
    )
    H[1, 2] = (
        math.sin(theta7) * ((math.sin(theta6) * math.sin(theta3) - math.cos(theta6) * math.cos(theta3)) * math.cos(theta2)
        + math.sin(theta2) * (math.cos(theta6) * math.sin(theta3) + math.sin(theta6) * math.cos(theta3))) * math.sin(theta1)
        + math.cos(theta1) * math.cos(theta7)
    )
    H[2, 0] = (
        ((math.cos(theta8) * math.sin(theta6) * math.cos(theta7) - math.sin(theta8) * math.cos(theta6)) * math.cos(theta3)
        + math.sin(theta3) * (math.cos(theta8) * math.cos(theta6) * math.cos(theta7) + math.sin(theta8) * math.sin(theta6))) * math.cos(theta2)
        - math.sin(theta2) * ((-math.cos(theta8) * math.cos(theta6) * math.cos(theta7) - math.sin(theta8) * math.sin(theta6)) * math.cos(theta3)
        + math.sin(theta3) * (math.cos(theta8) * math.sin(theta6) * math.cos(theta7) - math.sin(theta8) * math.cos(theta6)))
    )
    H[2, 1] = (
        ((-math.sin(theta6) * math.sin(theta8) * math.cos(theta7) - math.cos(theta8) * math.cos(theta6)) * math.cos(theta3)
        + math.sin(theta3) * (-math.sin(theta8) * math.cos(theta7) * math.cos(theta6) + math.cos(theta8) * math.sin(theta6))) * math.cos(theta2)
        + math.sin(theta2) * ((-math.sin(theta8) * math.cos(theta7) * math.cos(theta6) + math.cos(theta8) * math.sin(theta6)) * math.cos(theta3)
        + math.sin(theta3) * (math.sin(theta6) * math.sin(theta8) * math.cos(theta7) + math.cos(theta8) * math.cos(theta6)))
    )
    H[2, 2] = -(((math.sin(theta2) * math.cos(theta3) + math.cos(theta2) * math.sin(theta3)) * math.cos(theta6)
        + (math.cos(theta2) * math.cos(theta3) - math.sin(theta2) * math.sin(theta3)) * math.sin(theta6)) * math.sin(theta7))
    H[0, 3] = (
        ((((-math.sin(theta7) * d8 + a6) * math.cos(theta6) + a3) * math.cos(theta3)
        + ((math.sin(theta7) * d8 - a6) * math.sin(theta6) + d4 + 2.0 * q4) * math.sin(theta3) + a2) * math.cos(theta2)
        + ((math.sin(theta7) * d8 - a6) * math.sin(theta6) + d4 + 2.0 * q4) * math.sin(theta2) * math.cos(theta3)
        + ((math.sin(theta7) * d8 - a6) * math.cos(theta6) - a3) * math.sin(theta3) * math.sin(theta2) + a1) * math.cos(theta1)
        - math.sin(theta1) * math.cos(theta7) * d8
    )
    H[1, 3] = (
        ((((-math.sin(theta7) * d8 + a6) * math.cos(theta6) + a3) * math.cos(theta3)
        + ((math.sin(theta7) * d8 - a6) * math.sin(theta6) + d4 + 2.0 * q4) * math.sin(theta3) + a2) * math.cos(theta2)
        + ((math.sin(theta7) * d8 - a6) * math.sin(theta6) + d4 + 2.0 * q4) * math.sin(theta2) * math.cos(theta3)
        + ((math.sin(theta7) * d8 - a6) * math.cos(theta6) - a3) * math.sin(theta3) * math.sin(theta2) + a1) * math.sin(theta1)
        + math.cos(theta7) * math.cos(theta1) * d8
    )
    H[2, 3] = (
        (((-math.sin(theta7) * d8 + a6) * math.cos(theta6) + a3) * math.cos(theta3)
        + ((math.sin(theta7) * d8 - a6) * math.sin(theta6) + d4 + 2.0 * q4) * math.sin(theta3) + a2) * math.sin(theta2)
        + (((-math.sin(theta7) * d8 + a6) * math.sin(theta6) - d4 - 2.0 * q4) * math.cos(theta3)
        - math.sin(theta3) * ((math.sin(theta7) * d8 - a6) * math.cos(theta6) - a3)) * math.cos(theta2) + d1
    )
    H[3, 3] = 1.0
    return H


def _build_samples(cfg: AnalyticModelConfig, sample_count: int, seed: int) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    scenarios = make_default_scenarios()
    q_samples: list[dict[str, float]] = []
    for sc in scenarios.values():
        for q_red in (sc.planner_start_q, sc.planner_goal_q):
            if q_red is None:
                continue
            q_samples.append(
                {
                    "theta1_slewing_joint": float(q_red[0]),
                    "theta2_boom_joint": float(q_red[1]),
                    "theta3_arm_joint": float(q_red[2]),
                    "q4_big_telescope": float(q_red[3]),
                    "theta8_rotator_joint": float(q_red[4]),
                }
            )

    act_names = list(cfg.actuated_joints)
    bounds = {}
    desc = ModelDescription(cfg)
    for jn in act_names:
        jid = int(desc.model.getJointId(jn))
        joint = desc.model.joints[jid]
        iq = int(joint.idx_q)
        if int(joint.nq) == 1:
            lo = float(desc.model.lowerPositionLimit[iq])
            hi = float(desc.model.upperPositionLimit[iq])
            ov = cfg.joint_position_overrides.get(jn)
            if ov is not None:
                lo = max(lo, ov[0]) if ov[0] is not None else lo
                hi = min(hi, ov[1]) if ov[1] is not None else hi
            bounds[jn] = (lo, hi)
        else:
            bounds[jn] = (-math.pi, math.pi)

    while len(q_samples) < sample_count:
        sample = {}
        for jn in act_names:
            lo, hi = bounds[jn]
            if jn == "theta8_rotator_joint":
                lo, hi = -1.0, 1.0
            sample[jn] = float(rng.uniform(lo, hi))
        q_samples.append(sample)
    return q_samples[:sample_count]


def run_verification(sample_count: int, seed: int) -> dict[str, object]:
    cfg = AnalyticModelConfig.default()
    desc = ModelDescription(cfg)
    ss = CraneSteadyState(desc, cfg)
    timber_geom = _timber_geometry_from_urdf(Path(cfg.urdf_path))

    planner_urdf = Path(cfg.urdf_path).resolve()
    groundtruth_urdf = planner_urdf.with_name("crane_groundtruth_pzs100.urdf")
    urdf_match = groundtruth_urdf.exists() and planner_urdf.read_bytes() == groundtruth_urdf.read_bytes()

    frame_names = {f.name for f in desc.model.frames}
    required_frames = {"K0_mounting_base", "K5_inner_telescope", "K8_tool_center_point"}
    missing_frames = sorted(required_frames - frame_names)

    samples = _build_samples(cfg, sample_count=sample_count, seed=seed)
    results: list[dict[str, object]] = []
    max_maple_pos_err = 0.0
    max_maple_phi_err = 0.0
    max_roundtrip_pos_err = 0.0
    max_roundtrip_phi_err = 0.0
    max_roundtrip_q_act_err = 0.0
    complete_failures = 0
    solve_failures = 0

    for idx, q_act in enumerate(samples):
        completed = ss.complete_from_actuated(q_act, q_seed=q_act)
        if not completed.success:
            complete_failures += 1
            results.append({"sample": idx, "stage": "complete_from_actuated", "success": False, "message": completed.message})
            continue

        q_dyn = dict(completed.q_dynamic)
        q_dyn["q5_small_telescope"] = float(q_dyn["q4_big_telescope"])

        T_pin = _fk_transform_pin(desc, q_dyn, base_frame=cfg.base_frame, end_frame=cfg.target_frame)
        T_maple = _timber_transform_0_8(q_dyn, timber_geom)
        maple_pos_err = float(np.linalg.norm(T_pin[:3, 3] - T_maple[:3, 3]))
        maple_phi_err = abs(_wrap_to_pi(_phi_tool_from_transform(T_pin) - _phi_tool_from_transform(T_maple)))
        max_maple_pos_err = max(max_maple_pos_err, maple_pos_err)
        max_maple_phi_err = max(max_maple_phi_err, maple_phi_err)

        target_pos = np.asarray(T_pin[:3, 3], dtype=float)
        target_phi = _phi_tool_from_transform(T_pin)
        solved = ss.compute(target_pos=target_pos, target_yaw=target_phi, q_seed=q_dyn)
        if not solved.success:
            solve_failures += 1
            results.append(
                {
                    "sample": idx,
                    "stage": "steady_state_compute",
                    "success": False,
                    "message": solved.message,
                    "target_pos": target_pos.tolist(),
                    "target_phi": target_phi,
                }
            )
            continue

        q_act_err = 0.0
        for jn in cfg.actuated_joints:
            diff = _wrap_to_pi(float(solved.q_actuated[jn]) - float(q_act[jn])) if "theta" in jn else float(solved.q_actuated[jn]) - float(q_act[jn])
            q_act_err = max(q_act_err, abs(diff))
        max_roundtrip_q_act_err = max(max_roundtrip_q_act_err, q_act_err)
        max_roundtrip_pos_err = max(max_roundtrip_pos_err, float(solved.fk_position_error_m))
        max_roundtrip_phi_err = max(max_roundtrip_phi_err, abs(float(solved.fk_yaw_error_rad)))
        results.append(
            {
                "sample": idx,
                "success": True,
                "maple_pos_err_m": maple_pos_err,
                "maple_phi_err_rad": maple_phi_err,
                "roundtrip_pos_err_m": float(solved.fk_position_error_m),
                "roundtrip_phi_err_rad": float(abs(solved.fk_yaw_error_rad)),
                "roundtrip_max_actuated_err": q_act_err,
            }
        )

    summary = {
        "pinocchio_model": {
            "urdf_path": str(planner_urdf),
            "groundtruth_urdf_path": str(groundtruth_urdf),
            "groundtruth_matches_planner_snapshot": bool(urdf_match),
            "required_frames_present": len(missing_frames) == 0,
            "missing_frames": missing_frames,
            "nq": int(desc.model.nq),
            "nv": int(desc.model.nv),
            "nframes": len(desc.model.frames),
        },
        "sample_count": len(samples),
        "complete_failures": complete_failures,
        "solve_failures": solve_failures,
        "max_maple_pos_err_m": max_maple_pos_err,
        "max_maple_phi_err_rad": max_maple_phi_err,
        "max_roundtrip_pos_err_m": max_roundtrip_pos_err,
        "max_roundtrip_phi_err_rad": max_roundtrip_phi_err,
        "max_roundtrip_actuated_err": max_roundtrip_q_act_err,
        "samples": results,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify standalone FK/IK/steady-state against timber conventions and Pinocchio.")
    parser.add_argument("--samples", type=int, default=12, help="Number of reachable samples to check.")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed for random sample generation.")
    parser.add_argument("--json", action="store_true", help="Print the full JSON payload.")
    args = parser.parse_args()

    summary = run_verification(sample_count=int(args.samples), seed=int(args.seed))
    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print("Pinocchio model check:")
    print(json.dumps(summary["pinocchio_model"], indent=2))
    print("Verification summary:")
    print(
        json.dumps(
            {
                "sample_count": summary["sample_count"],
                "complete_failures": summary["complete_failures"],
                "solve_failures": summary["solve_failures"],
                "max_maple_pos_err_m": summary["max_maple_pos_err_m"],
                "max_maple_phi_err_rad": summary["max_maple_phi_err_rad"],
                "max_roundtrip_pos_err_m": summary["max_roundtrip_pos_err_m"],
                "max_roundtrip_phi_err_rad": summary["max_roundtrip_phi_err_rad"],
                "max_roundtrip_actuated_err": summary["max_roundtrip_actuated_err"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
