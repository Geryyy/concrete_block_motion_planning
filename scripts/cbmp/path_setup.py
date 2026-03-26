from __future__ import annotations

import sys
from pathlib import Path


def ensure_motion_planning_on_path() -> None:
    candidates = []

    here = Path(__file__).resolve()
    # Source layout: <pkg>/scripts/cbmp/path_setup.py -> <pkg>
    candidates.append(here.parents[2])

    # Installed layout: use package share directory if available.
    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.append(Path(get_package_share_directory("concrete_block_motion_planning")))
    except Exception:
        pass

    for candidate in candidates:
        motion_planning_dir = candidate / "motion_planning"
        if motion_planning_dir.is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return
