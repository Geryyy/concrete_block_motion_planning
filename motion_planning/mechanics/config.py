from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


_DEFAULT_YAML = Path(__file__).resolve().parent / "crane_config.yaml"


def resolve_existing_urdf_path(initial_path: str) -> str:
    path = Path(initial_path).expanduser()
    if path.exists():
        return str(path.resolve())
    here = Path(__file__).resolve()
    for parent in [Path.cwd(), here.parent, *here.parents]:
        urdf = parent / "crane_urdf" / "crane.urdf"
        if urdf.exists():
            return str(urdf.resolve())
        xacro = parent / "src" / "epsilon_crane_description" / "urdf" / "crane.urdf.xacro"
        if xacro.exists():
            out = Path(tempfile.gettempdir()) / "analytic_model_config_crane.urdf"
            out.write_text(
                subprocess.run(["xacro", str(xacro)], check=True, capture_output=True, text=True).stdout,
                encoding="utf-8",
            )
            return str(out.resolve())
    return str(path)


def _parse_joint_bounds(raw: object) -> tuple[Optional[float], Optional[float]] | None:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        lo_raw, hi_raw = raw
    elif isinstance(raw, dict):
        lo_raw = raw.get("lo", raw.get("min"))
        hi_raw = raw.get("hi", raw.get("max"))
    else:
        return None
    return (
        None if lo_raw is None else float(lo_raw),
        None if hi_raw is None else float(hi_raw),
    )


@dataclass
class AnalyticModelConfig:
    urdf_path: str
    actuated_joints: List[str]
    passive_joints: List[str]
    dynamic_joints: List[str]
    locked_joints: List[str]
    tied_joints: Dict[str, str] = field(default_factory=dict)
    joint_position_overrides: Dict[str, Tuple[Optional[float], Optional[float]]] = field(default_factory=dict)
    gravity: List[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    base_frame: str = "world"
    target_frame: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AnalyticModelConfig":
        path = Path(path).expanduser().resolve()
        with open(path) as fh:
            data = yaml.safe_load(fh)
        raw_overrides = data.get("joint_position_overrides", {})
        joint_position_overrides = (
            {
                str(name): parsed
                for name, raw in raw_overrides.items()
                if (parsed := _parse_joint_bounds(raw)) is not None
            }
            if isinstance(raw_overrides, dict)
            else {}
        )
        return cls(
            urdf_path=resolve_existing_urdf_path(str((path.parent / data.get("urdf_path", "")).resolve())),
            actuated_joints=list(data.get("actuated_joints", [])),
            passive_joints=list(data.get("passive_joints", [])),
            dynamic_joints=list(data.get("dynamic_joints", [])),
            locked_joints=list(data.get("locked_joints", [])),
            tied_joints=dict(data.get("tied_joints", {})),
            joint_position_overrides=joint_position_overrides,
            gravity=list(data.get("gravity", [0.0, 0.0, -9.81])),
            base_frame=str(data.get("base_frame", "world")),
            target_frame=str(data.get("target_frame", "")),
        )

    @classmethod
    def default(cls) -> "AnalyticModelConfig":
        return cls.from_yaml(_DEFAULT_YAML)

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.dump(
                {
                    "urdf_path": os.path.relpath(self.urdf_path, start=str(path.parent)),
                    "gravity": list(self.gravity),
                    "base_frame": self.base_frame,
                    "target_frame": self.target_frame,
                    "actuated_joints": list(self.actuated_joints),
                    "passive_joints": list(self.passive_joints),
                    "dynamic_joints": list(self.dynamic_joints),
                    "tied_joints": dict(self.tied_joints),
                    "joint_position_overrides": {
                        name: [lo, hi] for name, (lo, hi) in self.joint_position_overrides.items()
                    },
                    "locked_joints": list(self.locked_joints),
                },
                fh,
                default_flow_style=False,
                sort_keys=False,
            )
