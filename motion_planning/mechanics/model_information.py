"""Generate / display crane model information.

Running this script regenerates ``mechanics/crane_config.yaml`` from the
authoritative programmatic config in :func:`create_crane_config`.
"""
from pathlib import Path

from motion_planning.mechanics.analytic import ModelDescription, create_crane_config

_YAML_OUT = Path(__file__).resolve().parent / "crane_config.yaml"


def main() -> None:
    cfg = create_crane_config()
    cfg.save_yaml(_YAML_OUT)
    print(f"Wrote {_YAML_OUT}")

    model_desc = ModelDescription(cfg)
    model_desc.print_info()


if __name__ == "__main__":
    main()
