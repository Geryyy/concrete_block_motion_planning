#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import yaml


def _parse_vec3(text: str) -> list[float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    try:
        return [float(parts[0]), float(parts[1]), float(parts[2])]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from motion_planning import JointGoalStage

    parser = argparse.ArgumentParser(
        description="Probe concrete joint-goal feasibility for pre-approach offsets."
    )
    parser.add_argument("--start", type=_parse_vec3, required=True, help="world XYZ start as x,y,z")
    parser.add_argument("--goal", type=_parse_vec3, required=True, help="world XYZ target as x,y,z")
    parser.add_argument("--yaw-deg", type=float, default=-180.0, help="tool yaw in degrees")
    parser.add_argument(
        "--offsets",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        help="candidate pre-approach offsets in meters",
    )
    parser.add_argument("--yaml", action="store_true", help="emit YAML instead of text")
    args = parser.parse_args()

    stage = JointGoalStage()
    results = stage.solve_preapproach_family(
        start_world=args.start,
        target_world=args.goal,
        target_yaw_rad=math.radians(args.yaw_deg),
        offsets_m=args.offsets,
    )

    rows = []
    for res in results:
        rows.append(
            {
                "offset_m": round(res.offset_m, 4),
                "success": bool(res.success),
                "message": res.message,
                "goal_world": [float(v) for v in res.goal_world.tolist()],
                "goal_base": [float(v) for v in res.goal_base.tolist()],
                "q4_big_telescope": float(res.q_dynamic.get("q4_big_telescope", float("nan"))),
                "theta2_boom_joint": float(res.q_dynamic.get("theta2_boom_joint", float("nan"))),
                "theta3_arm_joint": float(res.q_dynamic.get("theta3_arm_joint", float("nan"))),
                "passive_residual": float(res.passive_residual),
            }
        )

    if args.yaml:
        print(yaml.safe_dump({"results": rows}, sort_keys=False))
        return 0

    print("Joint goal feasibility probe")
    print(f"start={args.start} goal={args.goal} yaw_deg={args.yaw_deg}")
    for row in rows:
        print(
            f"offset={row['offset_m']:.2f}m success={row['success']} "
            f"q4={row['q4_big_telescope']:.4f} "
            f"theta2={row['theta2_boom_joint']:.4f} "
            f"theta3={row['theta3_arm_joint']:.4f} "
            f"msg={row['message']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
