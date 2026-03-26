from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


_DEFAULT_YAML = Path(__file__).resolve().parents[1] / "crane_config.yaml"


def _resolve_existing_urdf_path(initial_path: str) -> str:
    p = Path(initial_path).expanduser()
    if p.exists():
        return str(p.resolve())

    here = Path(__file__).resolve()
    for parent in [Path.cwd(), here.parent, *here.parents]:
        cand = parent / "crane_urdf" / "crane.urdf"
        if cand.exists():
            return str(cand.resolve())
        xacro_cand = parent / "src" / "epsilon_crane_description" / "urdf" / "crane.urdf.xacro"
        if xacro_cand.exists():
            urdf_path = Path(tempfile.gettempdir()) / "analytic_model_config_crane.urdf"
            proc = subprocess.run(
                ["xacro", str(xacro_cand)],
                check=True,
                capture_output=True,
                text=True,
            )
            urdf_path.write_text(proc.stdout, encoding="utf-8")
            return str(urdf_path.resolve())
    return str(p)


@dataclass
class AnalyticModelConfig:
    """Configuration for the analytic kinematics / dynamics module.

    Load from YAML via :meth:`from_yaml` or use the bundled crane
    default via :meth:`default`.
    """

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

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AnalyticModelConfig":
        """Load config from a YAML file.

        ``urdf_path`` in the YAML may be relative — it is resolved against
        the directory that contains the YAML file.
        """
        path = Path(path).expanduser().resolve()
        with open(path) as fh:
            data = yaml.safe_load(fh)

        urdf_raw = data.get("urdf_path", "")
        urdf_resolved = _resolve_existing_urdf_path(str((path.parent / urdf_raw).resolve()))
        joint_position_overrides: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        raw_overrides = data.get("joint_position_overrides", {})
        if isinstance(raw_overrides, dict):
            for jn, bounds in raw_overrides.items():
                lo: Optional[float] = None
                hi: Optional[float] = None
                if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    lo = None if bounds[0] is None else float(bounds[0])
                    hi = None if bounds[1] is None else float(bounds[1])
                elif isinstance(bounds, dict):
                    lo_raw = bounds.get("lo", bounds.get("min"))
                    hi_raw = bounds.get("hi", bounds.get("max"))
                    lo = None if lo_raw is None else float(lo_raw)
                    hi = None if hi_raw is None else float(hi_raw)
                else:
                    continue
                joint_position_overrides[str(jn)] = (lo, hi)

        return cls(
            urdf_path=urdf_resolved,
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
        """Load the bundled crane config (``crane_config.yaml``)."""
        return cls.from_yaml(_DEFAULT_YAML)

    def save_yaml(self, path: str | Path) -> None:
        """Serialise config to *path* as YAML.

        ``urdf_path`` is stored as a path relative to the output file's
        directory so the file is portable.
        """
        path = Path(path).expanduser().resolve()
        urdf_rel = os.path.relpath(self.urdf_path, start=str(path.parent))
        data = {
            "urdf_path": urdf_rel,
            "gravity": list(self.gravity),
            "base_frame": self.base_frame,
            "target_frame": self.target_frame,
            "actuated_joints": list(self.actuated_joints),
            "passive_joints": list(self.passive_joints),
            "dynamic_joints": list(self.dynamic_joints),
            "tied_joints": dict(self.tied_joints),
            "joint_position_overrides": {jn: [lo, hi] for jn, (lo, hi) in self.joint_position_overrides.items()},
            "locked_joints": list(self.locked_joints),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def all_joints(self) -> List[str]:
        """Union of dynamic, locked (and any remaining) joints."""
        seen: set[str] = set()
        out: List[str] = []
        for j in list(self.dynamic_joints) + list(self.locked_joints):
            if j not in seen:
                seen.add(j)
                out.append(j)
        return out
