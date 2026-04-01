from __future__ import annotations

import numpy as np
import pytest
import pinocchio as pin

from motion_planning.kinematics.crane import CraneKinematics
from motion_planning.mechanics.analytic import AnalyticModelConfig, CraneSteadyState, ModelDescription
from motion_planning.mechanics.analytic.pinocchio_utils import q_map_to_pin_q


def _phi_tool_from_transform(T: np.ndarray) -> float:
    T_arr = np.asarray(T, dtype=float).reshape(4, 4)
    return float(np.arctan2(T_arr[1, 1], T_arr[0, 1]))


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
    q_actuated = {
        "theta1_slewing_joint": 0.05,
        "theta2_boom_joint": -0.85,
        "theta3_arm_joint": 0.55,
        "q4_big_telescope": 0.35,
        "theta8_rotator_joint": -0.05,
    }
    completed = ss.complete_from_actuated(q_actuated, q_seed=q_actuated)
    assert completed.success, completed.message
    q_seed = dict(completed.q_dynamic)
    q_seed["q5_small_telescope"] = q_seed["q4_big_telescope"]
    q_seed_pin = q_map_to_pin_q(pin_kin.model, q_seed, pin_module=pin)
    fk_seed = pin_kin.forward_kinematics(q_seed_pin, base_frame=cfg.base_frame, end_frame=cfg.target_frame)
    T_target = np.asarray(fk_seed["base_to_end"]["homogeneous"], dtype=float)
    p_target = T_target[:3, 3].copy()
    yaw_target = _phi_tool_from_transform(T_target)

    res = ss.compute(target_pos=p_target, target_yaw=yaw_target, q_seed=q_seed)
    assert res.success, res.message
    assert res.passive_residual < 1e-5
    assert res.fk_position_error_m < 2e-2
    assert abs(res.fk_yaw_error_rad) < 1e-2

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
    assert ("IK failed" in res.message) or ("FK truth check" in res.message)
