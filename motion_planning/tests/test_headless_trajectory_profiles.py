from __future__ import annotations

from pathlib import Path

import yaml


def test_headless_trajectory_profiles_are_well_formed() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "config" / "headless_trajectory_profiles.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert "profiles" in data
    assert "slew_out_and_back" in data["profiles"]

    for profile_name, profile in data["profiles"].items():
        joint_names = profile["joint_names"]
        waypoints = profile["waypoints"]
        assert joint_names, profile_name
        assert len(waypoints) >= 2, profile_name
        for waypoint in waypoints:
            assert len(waypoint["deltas"]) == len(joint_names), profile_name


def test_oneway_profiles_are_present() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "config" / "headless_trajectory_profiles.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert "slew_oneway" in data["profiles"]
    assert "telescope_oneway" in data["profiles"]
