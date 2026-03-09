from __future__ import annotations

import re
import uuid


_ID_SUFFIX_LEN = 8
_SAFE_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_]+")


def _suffix() -> str:
    return uuid.uuid4().hex[:_ID_SUFFIX_LEN]


def _sanitize_name(name: str) -> str:
    cleaned = _SAFE_NAME_PATTERN.sub("_", str(name).strip())
    return cleaned.strip("_") or "unnamed"


def make_geometric_plan_id() -> str:
    return f"geo_{_suffix()}"


def make_trajectory_id() -> str:
    return f"traj_{_suffix()}"


def make_named_trajectory_id(configuration_name: str) -> str:
    return f"named_{_sanitize_name(configuration_name)}_{_suffix()}"
