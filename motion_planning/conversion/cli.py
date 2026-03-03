from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

from .urdf_to_mjcf import (
    compare_pin_models_dynamics,
    compare_pin_models_kinematics,
    compare_urdf_inertials_to_mjcf,
    compile_urdf_to_mjcf,
    synchronize_mjcf_inertials_from_urdf,
)


def _default_urdf() -> Path:
    return Path(__file__).resolve().parents[2] / "crane_urdf" / "crane.urdf"


def _default_mjcf() -> Path:
    return Path(__file__).resolve().parents[2] / "crane_urdf" / "crane.xml"


def _print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(title)
    for k, v in metrics.items():
        print(f"  {k}: {v:.6e}")


def _strip_to_model_core(root: ET.Element) -> ET.Element:
    keep_top = {"compiler", "asset", "worldbody"}
    for child in list(root):
        if child.tag not in keep_top:
            root.remove(child)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for child in list(worldbody):
            # Keep only the articulated model body tree.
            if child.tag != "body":
                worldbody.remove(child)

    def _strip_body(body: ET.Element) -> None:
        for child in list(body):
            # Ignore non-URDF extras (e.g., marker sites).
            if child.tag not in {"inertial", "joint", "geom", "body"}:
                body.remove(child)
                continue
            if child.tag == "body":
                _strip_body(child)

    if worldbody is not None:
        for body in worldbody.findall("body"):
            _strip_body(body)

    return root


def _canonicalize_xml(elem: ET.Element) -> str:
    def _canon(e: ET.Element) -> ET.Element:
        out = ET.Element(e.tag, attrib={k: e.attrib[k] for k in sorted(e.attrib)})
        out.text = None
        out.tail = None
        for c in list(e):
            out.append(_canon(c))
        return out

    return ET.tostring(_canon(elem), encoding="unicode")


def _model_core_is_synced(urdf: Path, mjcf: Path) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="motion_planning_synccheck_") as td:
        td_path = Path(td)
        compiled = compile_urdf_to_mjcf(urdf, td_path / "compiled.xml")
        generated = synchronize_mjcf_inertials_from_urdf(urdf, compiled, td_path / "synced.xml")

        lhs_root = _strip_to_model_core(ET.parse(mjcf).getroot())
        rhs_root = _strip_to_model_core(ET.parse(generated).getroot())
        lhs = _canonicalize_xml(lhs_root)
        rhs = _canonicalize_xml(rhs_root)
        if lhs == rhs:
            return True, "model core is synchronized"
        return (
            False,
            "model core differs from URDF-generated MJCF (ignoring scene/actuator/site extras).",
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="motion_planning.conversion",
        description="URDF -> MJCF conversion and URDF/MJCF consistency checks.",
    )
    sp = p.add_subparsers(dest="cmd", required=True)

    p_compile = sp.add_parser("compile", help="Compile URDF to MJCF using MuJoCo.")
    p_compile.add_argument("--urdf", type=Path, default=_default_urdf())
    p_compile.add_argument("--out", type=Path, required=True)

    p_sync = sp.add_parser("sync-inertias", help="Copy URDF inertials into MJCF body inertials.")
    p_sync.add_argument("--urdf", type=Path, default=_default_urdf())
    p_sync.add_argument("--mjcf", type=Path, required=True)
    p_sync.add_argument("--out", type=Path, required=True)
    p_sync.add_argument("--no-update-existing", action="store_true")

    p_ci = sp.add_parser("compare-inertia", help="Compare URDF inertials against MJCF inertials.")
    p_ci.add_argument("--urdf", type=Path, default=_default_urdf())
    p_ci.add_argument("--mjcf", type=Path, required=True)
    p_ci.add_argument("--json", action="store_true", help="Print JSON output.")

    p_ck = sp.add_parser("compare-kinematics", help="Compare URDF vs MJCF kinematics in Pinocchio.")
    p_ck.add_argument("--urdf", type=Path, default=_default_urdf())
    p_ck.add_argument("--mjcf", type=Path, required=True)
    p_ck.add_argument("--samples", type=int, default=20)
    p_ck.add_argument("--seed", type=int, default=0)
    p_ck.add_argument("--json", action="store_true", help="Print JSON output.")

    p_cd = sp.add_parser("compare-dynamics", help="Compare URDF vs MJCF dynamics in Pinocchio.")
    p_cd.add_argument("--urdf", type=Path, default=_default_urdf())
    p_cd.add_argument("--mjcf", type=Path, required=True)
    p_cd.add_argument("--samples", type=int, default=20)
    p_cd.add_argument("--seed", type=int, default=0)
    p_cd.add_argument("--json", action="store_true", help="Print JSON output.")

    p_pipe = sp.add_parser("pipeline", help="Compile URDF, sync inertials, and run all comparisons.")
    p_pipe.add_argument("--urdf", type=Path, default=_default_urdf())
    p_pipe.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/motion_planning_conversion"),
    )
    p_pipe.add_argument("--samples", type=int, default=20)
    p_pipe.add_argument("--seed", type=int, default=0)
    p_pipe.add_argument("--json", action="store_true", help="Print JSON output.")

    p_check = sp.add_parser(
        "check-sync",
        help="Check whether MJCF model core matches URDF-generated model (ignores scene/actuator/site extras).",
    )
    p_check.add_argument("--urdf", type=Path, default=_default_urdf())
    p_check.add_argument("--mjcf", type=Path, default=_default_mjcf())
    p_check.add_argument("--json", action="store_true", help="Print JSON output.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "compile":
        out = compile_urdf_to_mjcf(args.urdf, args.out)
        print(f"compiled_mjcf: {out}")
        return 0

    if args.cmd == "sync-inertias":
        out = synchronize_mjcf_inertials_from_urdf(
            args.urdf,
            args.mjcf,
            args.out,
            update_existing=not args.no_update_existing,
        )
        print(f"synced_mjcf: {out}")
        return 0

    if args.cmd == "compare-inertia":
        metrics = compare_urdf_inertials_to_mjcf(args.urdf, args.mjcf)
        if args.json:
            print(json.dumps(metrics, indent=2))
        else:
            _print_metrics("inertia_comparison", metrics)
        return 0

    if args.cmd == "compare-kinematics":
        metrics = compare_pin_models_kinematics(args.urdf, args.mjcf, samples=args.samples, seed=args.seed)
        if args.json:
            print(json.dumps(metrics, indent=2))
        else:
            _print_metrics("kinematics_comparison", metrics)
        return 0

    if args.cmd == "compare-dynamics":
        metrics = compare_pin_models_dynamics(args.urdf, args.mjcf, samples=args.samples, seed=args.seed)
        if args.json:
            print(json.dumps(metrics, indent=2))
        else:
            _print_metrics("dynamics_comparison", metrics)
        return 0

    if args.cmd == "pipeline":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        compiled = compile_urdf_to_mjcf(args.urdf, args.out_dir / "compiled.xml")
        synced = synchronize_mjcf_inertials_from_urdf(args.urdf, compiled, args.out_dir / "synced.xml")

        res = {
            "paths": {
                "compiled_mjcf": str(compiled),
                "synced_mjcf": str(synced),
            },
            "inertia_before": compare_urdf_inertials_to_mjcf(args.urdf, compiled),
            "inertia_after": compare_urdf_inertials_to_mjcf(args.urdf, synced),
            "kinematics_after": compare_pin_models_kinematics(
                args.urdf, synced, samples=args.samples, seed=args.seed
            ),
            "dynamics_before": compare_pin_models_dynamics(
                args.urdf, compiled, samples=args.samples, seed=args.seed
            ),
            "dynamics_after": compare_pin_models_dynamics(
                args.urdf, synced, samples=args.samples, seed=args.seed
            ),
        }
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"compiled_mjcf: {compiled}")
            print(f"synced_mjcf: {synced}")
            _print_metrics("inertia_before", res["inertia_before"])
            _print_metrics("inertia_after", res["inertia_after"])
            _print_metrics("kinematics_after", res["kinematics_after"])
            _print_metrics("dynamics_before", res["dynamics_before"])
            _print_metrics("dynamics_after", res["dynamics_after"])
        return 0

    if args.cmd == "check-sync":
        ok, msg = _model_core_is_synced(args.urdf, args.mjcf)
        if args.json:
            print(json.dumps({"ok": bool(ok), "message": msg}, indent=2))
        else:
            print(f"sync_ok: {ok}")
            print(f"message: {msg}")
        return 0 if ok else 1

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
