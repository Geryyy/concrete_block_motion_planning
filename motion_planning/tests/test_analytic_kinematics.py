from __future__ import annotations

import numpy as np
import pytest

from motion_planning.kinematics.crane import CraneKinematics
from motion_planning.mechanics.analytic import AnalyticModelConfig, ModelDescription
from motion_planning.mechanics.analytic.pinocchio_utils import fk_homogeneous, joint_bounds, q_map_to_pin_q


@pytest.fixture(scope="module")
def cfg() -> AnalyticModelConfig:
    return AnalyticModelConfig.default()


@pytest.fixture(scope="module")
def desc(cfg: AnalyticModelConfig) -> ModelDescription:
    return ModelDescription(cfg)


@pytest.fixture(scope="module")
def pin_kin(cfg: AnalyticModelConfig) -> CraneKinematics:
    return CraneKinematics(cfg.urdf_path)


def test_config_default_loads(cfg: AnalyticModelConfig) -> None:
    assert cfg.urdf_path.endswith("crane.urdf")
    assert len(cfg.dynamic_joints) > 0


def test_model_description_joint_and_frame_info(desc: ModelDescription) -> None:
    joints = desc.joint_info()
    frames = desc.frame_info()
    assert len(joints) == desc.model.njoints - 1
    assert len(frames) == len(desc.model.frames)


def test_pinocchio_utils_q_map_to_pin_and_fk(cfg: AnalyticModelConfig, desc: ModelDescription, pin_kin: CraneKinematics) -> None:
    q_map = {jn: 0.0 for jn in cfg.dynamic_joints}
    q_pin = q_map_to_pin_q(desc.model, q_map, pin_kin.pin)
    fk_ref = pin_kin.forward_kinematics(q_pin, base_frame="world", end_frame="K8_tool_center_point")
    T_ref = np.asarray(fk_ref["base_to_end"]["homogeneous"], dtype=float)

    T = fk_homogeneous(
        pin_model=desc.model,
        pin_data=desc.model.createData(),
        pin_module=pin_kin.pin,
        q_values=q_map,
        base_frame="world",
        end_frame="K8_tool_center_point",
        frame_cache={},
    )
    assert T.shape == (4, 4)
    assert np.max(np.abs(T - T_ref)) < 1e-12


def test_joint_bounds_readable(desc: ModelDescription) -> None:
    lo, hi = joint_bounds(desc.model, "theta2_boom_joint")
    assert np.isfinite(lo) and np.isfinite(hi)
    assert hi > lo
