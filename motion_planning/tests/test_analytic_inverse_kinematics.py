from __future__ import annotations

import pytest
import numpy as np

from motion_planning.mechanics.analytic import (
    AnalyticInverseKinematics,
    AnalyticModelConfig,
    ModelDescription,
)
from motion_planning.kinematics.crane import CraneKinematics


@pytest.fixture(scope="module")
def cfg() -> AnalyticModelConfig:
    return AnalyticModelConfig.default()


@pytest.fixture(scope="module")
def desc(cfg: AnalyticModelConfig) -> ModelDescription:
    return ModelDescription(cfg)


@pytest.fixture(scope="module")
def pin_kin(cfg: AnalyticModelConfig) -> CraneKinematics:
    return CraneKinematics(cfg.urdf_path)


@pytest.fixture(scope="module")
def ik(cfg: AnalyticModelConfig, desc: ModelDescription) -> AnalyticInverseKinematics:
    return AnalyticInverseKinematics(desc, cfg)


def _pin_q_from_joint_map(pin_model, q_joint: dict[str, float]) -> np.ndarray:
    import pinocchio as pin

    q = np.asarray(pin.neutral(pin_model), dtype=float)
    for jname, val in q_joint.items():
        if not pin_model.existJointName(jname):
            continue
        jid = int(pin_model.getJointId(jname))
        j = pin_model.joints[jid]
        iq, nq, nv = int(j.idx_q), int(j.nq), int(j.nv)
        if nq == 1:
            q[iq] = float(val)
        elif nq == 2 and nv == 1:
            q[iq] = float(np.cos(val))
            q[iq + 1] = float(np.sin(val))
    return q


def _assert_dynamic_joints_within_urdf_limits(model, q_dynamic: dict[str, float], *, tol: float = 1e-9) -> None:
    for jn, qv in q_dynamic.items():
        jid = int(model.getJointId(jn))
        joint = model.joints[jid]
        nq = int(joint.nq)
        if nq == 1:
            iq = int(joint.idx_q)
            lo = float(model.lowerPositionLimit[iq])
            hi = float(model.upperPositionLimit[iq])
            if np.isfinite(lo):
                assert qv >= lo - tol, f"{jn} below URDF lower limit: q={qv}, lo={lo}"
            if np.isfinite(hi):
                assert qv <= hi + tol, f"{jn} above URDF upper limit: q={qv}, hi={hi}"
        elif nq == 2 and int(joint.nv) == 1:
            # Continuous revolute joints use cosine/sine representation in Pinocchio and are unbounded in angle.
            continue
        else:
            raise AssertionError(f"Unsupported joint representation for limit check: {jn} (nq={joint.nq}, nv={joint.nv})")


def _sample_q_within_urdf_limits(model, cfg: AnalyticModelConfig, rng: np.random.Generator) -> dict[str, float]:
    q = {}
    for jn in cfg.dynamic_joints:
        jid = int(model.getJointId(jn))
        joint = model.joints[jid]
        nq = int(joint.nq)
        if nq == 1:
            iq = int(joint.idx_q)
            lo = float(model.lowerPositionLimit[iq])
            hi = float(model.upperPositionLimit[iq])
            if np.isfinite(lo) and np.isfinite(hi):
                q[jn] = float(rng.uniform(lo, hi))
            else:
                q[jn] = float(rng.uniform(-np.pi, np.pi))
        elif nq == 2 and int(joint.nv) == 1:
            q[jn] = float(rng.uniform(-np.pi, np.pi))
        else:
            raise AssertionError(f"Unsupported joint representation for sampling: {jn} (nq={joint.nq}, nv={joint.nv})")
    return q


def test_ik_recovers_pose_for_valid_target(cfg: AnalyticModelConfig, pin_kin: CraneKinematics, ik: AnalyticInverseKinematics):
    q_ref = {
        "theta1_slewing_joint": 0.2,
        "theta2_boom_joint": 0.35,
        "theta3_arm_joint": 0.8,
        "q4_big_telescope": 0.25,
        "theta6_tip_joint": 0.1,
        "theta7_tilt_joint": 1.0,
        "theta8_rotator_joint": 0.2,
    }
    q_pin = _pin_q_from_joint_map(pin_kin.model, q_ref)
    fk = pin_kin.forward_kinematics(q_pin, base_frame="world", end_frame="theta8_rotator_joint")
    T = np.asarray(fk["base_to_end"]["homogeneous"], dtype=float)
    res = ik.solve_pose(
        target_T_base_to_end=T,
        base_frame="world",
        end_frame="theta8_rotator_joint",
        q_seed=q_ref,
        actuated_joint_names=list(cfg.dynamic_joints),
        max_nfev=80,
    )
    assert res.success
    assert res.pos_error_m < 2e-4
    assert res.rot_error_rad < 2e-4
    _assert_dynamic_joints_within_urdf_limits(pin_kin.model, res.q_dynamic)


def test_ik_telescope_is_single_dof(cfg: AnalyticModelConfig, pin_kin: CraneKinematics, ik: AnalyticInverseKinematics):
    q_ref = {
        "theta1_slewing_joint": 0.0,
        "theta2_boom_joint": 0.2,
        "theta3_arm_joint": 0.6,
        "q4_big_telescope": 0.4,
        "theta6_tip_joint": 0.0,
        "theta7_tilt_joint": 1.2,
        "theta8_rotator_joint": 0.0,
    }
    q_pin = _pin_q_from_joint_map(pin_kin.model, q_ref)
    fk = pin_kin.forward_kinematics(q_pin, base_frame="world", end_frame="theta8_rotator_joint")
    T = np.asarray(fk["base_to_end"]["homogeneous"], dtype=float)
    res = ik.solve_pose(
        target_T_base_to_end=T,
        base_frame="world",
        end_frame="theta8_rotator_joint",
        q_seed=q_ref,
        actuated_joint_names=list(cfg.dynamic_joints),
        max_nfev=80,
    )
    assert res.success
    assert "q5_small_telescope" not in res.q_dynamic
    _assert_dynamic_joints_within_urdf_limits(pin_kin.model, res.q_dynamic)


def test_ik_solution_respects_urdf_limits_even_with_out_of_range_seed(
    cfg: AnalyticModelConfig,
    pin_kin: CraneKinematics,
    ik: AnalyticInverseKinematics,
):
    q_target = {
        "theta1_slewing_joint": 0.1,
        "theta2_boom_joint": 0.3,
        "theta3_arm_joint": 0.7,
        "q4_big_telescope": 0.2,
        "theta6_tip_joint": 0.0,
        "theta7_tilt_joint": 1.1,
        "theta8_rotator_joint": -0.2,
    }
    q_pin = _pin_q_from_joint_map(pin_kin.model, q_target)
    fk = pin_kin.forward_kinematics(q_pin, base_frame="world", end_frame="theta8_rotator_joint")
    T = np.asarray(fk["base_to_end"]["homogeneous"], dtype=float)
    q_bad_seed = {jn: 1e6 for jn in cfg.dynamic_joints}
    res = ik.solve_pose(
        target_T_base_to_end=T,
        base_frame="world",
        end_frame="theta8_rotator_joint",
        q_seed=q_bad_seed,
        actuated_joint_names=list(cfg.dynamic_joints),
        max_nfev=120,
    )
    assert res.success
    _assert_dynamic_joints_within_urdf_limits(pin_kin.model, res.q_dynamic)


def test_monte_carlo_fk_ik_within_urdf_joint_limits(
    cfg: AnalyticModelConfig,
    pin_kin: CraneKinematics,
    ik: AnalyticInverseKinematics,
):
    rng = np.random.default_rng(1234)
    n_samples = 30

    for _ in range(n_samples):
        q_ref = _sample_q_within_urdf_limits(pin_kin.model, cfg, rng)
        _assert_dynamic_joints_within_urdf_limits(pin_kin.model, q_ref)

        q_pin = _pin_q_from_joint_map(pin_kin.model, q_ref)
        fk = pin_kin.forward_kinematics(q_pin, base_frame="world", end_frame="theta8_rotator_joint")
        T = np.asarray(fk["base_to_end"]["homogeneous"], dtype=float)
        res = ik.solve_pose(
            target_T_base_to_end=T,
            base_frame="world",
            end_frame="theta8_rotator_joint",
            q_seed=q_ref,
            actuated_joint_names=list(cfg.dynamic_joints),
            max_nfev=100,
        )
        assert res.success
        assert res.pos_error_m < 5e-4
        assert res.rot_error_rad < 2e-4
        _assert_dynamic_joints_within_urdf_limits(pin_kin.model, res.q_dynamic)


def test_numeric_fallback_respects_config_joint_override(
    cfg: AnalyticModelConfig,
    pin_kin: CraneKinematics,
    ik: AnalyticInverseKinematics,
):
    q_ref = {
        "theta1_slewing_joint": 0.1,
        "theta2_boom_joint": 0.3,
        "theta3_arm_joint": 0.8,
        "q4_big_telescope": 0.25,
        "theta6_tip_joint": 0.0,
        "theta7_tilt_joint": 1.1,
        "theta8_rotator_joint": -0.2,
    }
    q_pin = _pin_q_from_joint_map(pin_kin.model, q_ref)
    fk = pin_kin.forward_kinematics(q_pin, base_frame="world", end_frame="theta8_rotator_joint")
    T = np.asarray(fk["base_to_end"]["homogeneous"], dtype=float)

    q_bad_seed = dict(q_ref)
    q_bad_seed["theta3_arm_joint"] = 3.0
    res = ik.solve_pose(
        target_T_base_to_end=T,
        base_frame="world",                  # force numeric fallback (analytic branch unavailable)
        end_frame="theta8_rotator_joint",
        q_seed=q_bad_seed,
        actuated_joint_names=list(cfg.dynamic_joints),
        max_nfev=120,
    )
    assert res.success
    theta3_hi = cfg.joint_position_overrides["theta3_arm_joint"][1]
    assert theta3_hi is not None
    assert float(res.q_dynamic["theta3_arm_joint"]) <= float(theta3_hi) + 1e-9
