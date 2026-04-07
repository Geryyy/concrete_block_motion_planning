from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

_CANONICAL_METHOD = {
    "POWELL": "Powell",
    "NELDER-MEAD": "Nelder-Mead",
    "NELDER_MEAD": "Nelder-Mead",
    "NELDERMEAD": "Nelder-Mead",
    "NELDER": "Nelder-Mead",
    "NM": "Nelder-Mead",
    "CEM": "CEM",
    "VP-STO": "VP-STO",
    "OMPL-RRT": "OMPL-RRT",
    "OMPL": "OMPL-RRT",
    "RRT": "OMPL-RRT",
}


def canonical_method_name(method: str) -> str:
    key = str(method).strip().upper()
    return _CANONICAL_METHOD.get(key, method)


def load_optimized_planner_params(path: str | Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load benchmark-optimized planner params keyed by canonical method name.

    Returns: {"Powell": {"config": {...}, "options": {...}}, ...}
    """
    p = Path(path)
    payload = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid optimized params file: {p}")
    methods = payload.get("methods", {})
    if not isinstance(methods, Mapping):
        raise ValueError("optimized params YAML must contain a 'methods' mapping")

    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for method_name, entry in methods.items():
        if not isinstance(entry, Mapping):
            continue
        canonical = canonical_method_name(str(method_name))
        cfg = dict(entry.get("config", {}) or {})
        opts = dict(entry.get("options", {}) or {})
        out[canonical] = {"config": cfg, "options": opts}
    return out
