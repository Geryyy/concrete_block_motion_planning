from __future__ import annotations

from pathlib import Path

import pytest

from motion_planning import (
    compare_pin_models_dynamics,
    compare_pin_models_kinematics,
    compare_urdf_inertials_to_mjcf,
    compile_urdf_to_mjcf,
    synchronize_mjcf_inertials_from_urdf,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
URDF = REPO_ROOT / "crane_urdf" / "crane.urdf"


@pytest.mark.skipif(not URDF.exists(), reason="crane URDF missing")
def test_compile_urdf_to_mjcf(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    out = tmp_path / "compiled.xml"
    compiled = compile_urdf_to_mjcf(URDF, out)
    assert compiled.exists()
    text = compiled.read_text(encoding="utf-8")
    assert "<mujoco" in text
    assert "moved_with_q2" in text


@pytest.mark.skipif(not URDF.exists(), reason="crane URDF missing")
def test_inertia_sync_improves_mass_and_com_match(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")

    compiled = compile_urdf_to_mjcf(URDF, tmp_path / "compiled.xml")
    before = compare_urdf_inertials_to_mjcf(URDF, compiled)
    synced = synchronize_mjcf_inertials_from_urdf(URDF, compiled, tmp_path / "synced.xml")
    after = compare_urdf_inertials_to_mjcf(URDF, synced)

    assert after["mean_abs_mass_diff"] < before["mean_abs_mass_diff"]
    assert after["mean_com_diff"] < before["mean_com_diff"]
    assert after["mean_inertia_diag_diff"] < before["mean_inertia_diag_diff"]


@pytest.mark.skipif(not URDF.exists(), reason="crane URDF missing")
def test_kinematic_comparison_is_stable_after_sync(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")

    compiled = compile_urdf_to_mjcf(URDF, tmp_path / "compiled.xml")
    synced = synchronize_mjcf_inertials_from_urdf(URDF, compiled, tmp_path / "synced.xml")
    kin = compare_pin_models_kinematics(URDF, synced, samples=8, seed=1)

    assert kin["num_frames"] > 0
    assert kin["mean_pos_err_m"] >= 0.0
    assert kin["max_pos_err_m"] >= kin["mean_pos_err_m"]
    assert kin["mean_rot_err_rad"] >= 0.0
    assert kin["max_rot_err_rad"] >= kin["mean_rot_err_rad"]


@pytest.mark.skipif(not URDF.exists(), reason="crane URDF missing")
def test_dynamic_comparison_improves_after_sync(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")

    compiled = compile_urdf_to_mjcf(URDF, tmp_path / "compiled.xml")
    dyn_before = compare_pin_models_dynamics(URDF, compiled, samples=8, seed=2)
    synced = synchronize_mjcf_inertials_from_urdf(URDF, compiled, tmp_path / "synced.xml")
    dyn_after = compare_pin_models_dynamics(URDF, synced, samples=8, seed=2)

    assert dyn_before["mean_M_diff"] >= 0.0
    assert dyn_before["mean_h_diff"] >= 0.0
    assert dyn_after["mean_M_diff"] >= 0.0
    assert dyn_after["mean_h_diff"] >= 0.0
    # Inertia sync must produce a dynamics change relative to the compiled baseline.
    assert abs(dyn_after["mean_M_diff"] - dyn_before["mean_M_diff"]) > 1e-6
