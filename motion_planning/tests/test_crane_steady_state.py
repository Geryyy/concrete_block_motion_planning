from __future__ import annotations

import numpy as np
import pytest
import pinocchio as pin

from motion_planning.kinematics.crane import CraneKinematics
from motion_planning.mechanics.analytic import AnalyticModelConfig, CraneSteadyState, ModelDescription
from motion_planning.mechanics.analytic.pinocchio_utils import q_map_to_pin_q


@pytest.fixture(scope="module")
def cfg() -> AnalyticModelConfig:
    return AnalyticModelConfig.default()


@pytest.fixture(scope="module")
def desc(cfg: AnalyticModelConfig) -> ModelDescription:
    return ModelDescription(cfg)


@pytest.fixture(scope="module")
def ss(cfg: AnalyticModelConfig, desc: ModelDescription) -> CraneSteadyState:
    return CraneSteadyState(desc, cfg)


@pytest.fixture(scope="module")
def pin_kin(cfg: AnalyticModelConfig) -> CraneKinematics:
    return CraneKinematics(cfg.urdf_path)

def test_steady_state_balances_passive_and_respects_theta3_cap(
    cfg: AnalyticModelConfig,
    ss: CraneSteadyState,
    pin_kin: CraneKinematics,
):
    q_seed = {
        "theta1_slewing_joint": 0.2,
        "theta2_boom_joint": 0.3,
        "theta3_arm_joint": 0.8,
        "q4_big_telescope": 0.25,
        "theta6_tip_joint": 0.0,
        "theta7_tilt_joint": 1.1,
        "theta8_rotator_joint": -0.2,
    }
    q_seed_pin = q_map_to_pin_q(pin_kin.model, q_seed, pin_module=pin)
    fk_seed = pin_kin.forward_kinematics(q_seed_pin, base_frame=cfg.base_frame, end_frame=cfg.target_frame)
    T_target = np.asarray(fk_seed["base_to_end"]["homogeneous"], dtype=float)
    p_target = T_target[:3, 3].copy()
    yaw_target = float(np.arctan2(T_target[1, 0], T_target[0, 0]))

    res = ss.compute(target_pos=p_target, target_yaw=yaw_target, q_seed=q_seed)
    assert res.success, res.message
    assert res.passive_residual < 1e-5

    q_theta3 = float(res.q_dynamic["theta3_arm_joint"])
    theta3_override = cfg.joint_position_overrides.get("theta3_arm_joint", (None, None))
    assert theta3_override[1] is not None
    assert q_theta3 <= float(theta3_override[1]) + 1e-9

    for jn in cfg.passive_joints:
        assert np.isfinite(float(res.q_dynamic[jn]))
        assert np.isfinite(float(res.q_passive[jn]))


def test_default_config_contains_theta3_upper_override(cfg: AnalyticModelConfig):
    assert "theta3_arm_joint" in cfg.joint_position_overrides
    lo, hi = cfg.joint_position_overrides["theta3_arm_joint"]
    assert lo is None
    assert hi is not None
    assert abs(float(hi) - (np.pi / 2.0)) < 1e-12


def test_steady_state_reports_failure_for_unreachable_target(ss: CraneSteadyState):
    res = ss.compute(target_pos=np.array([100.0, 100.0, 100.0]), target_yaw=0.0, q_seed={})
    assert not res.success
    assert "IK failed" in res.message
