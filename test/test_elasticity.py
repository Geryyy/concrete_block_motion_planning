"""The loaded bend as a goal correction (issue 300)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from motion_planning.mechanics.elasticity import (  # noqa: E402
    ElasticityCompensation,
    elasticity_dz,
)

#: mp_parameter_pzs100.yaml elasticityModel.p
P = [0.00750624, 0.0102337, 1.05088e-06]


def _compensation(enabled=True, payload_mass=880.0):
    """tau3 = 10 kN*m of structure plus 50 N*m per kg of payload."""
    return ElasticityCompensation(
        p=P,
        payload_mass=payload_mass,
        tau3_fn=lambda q_values, mass: 10000.0 + 50.0 * mass,
        enabled=enabled,
    )


def test_dz_is_the_fitted_polynomial() -> None:
    assert np.isclose(
        elasticity_dz(54000.0, 1.5, P),
        P[0] * 1.5 + P[1] * 1.5**2 + P[2] * 54000.0 * 1.5,
    )


def test_the_place_shift_is_the_change_in_bend_not_the_bend() -> None:
    """Issue 294 Session 3: the jaws closed with the pick's sag in the grip depth.

    Correcting the absolute sag would place the block that whole sag high (3-7 cm
    unloaded at these coefficients), so only the payload's own term is left when the
    place pose equals the pick pose.
    """
    compensation = _compensation()
    pose = {"q4_big_telescope": 1.5}

    compensation.store_pick(pose)
    assert np.isclose(compensation.dz_pick, elasticity_dz(10000.0, 1.5, P))
    assert compensation.dz_pick > 0.03, (
        "the unloaded sag is what would be double-counted"
    )

    shift = compensation.place_shift(pose)
    assert np.isclose(shift, P[2] * 50.0 * 880.0 * 1.5)
    assert not np.isclose(shift, compensation.dz(pose, 880.0))


def test_the_shift_follows_the_pose_the_place_ends_in() -> None:
    compensation = _compensation()
    compensation.store_pick({"q4_big_telescope": 1.0})

    out = compensation.place_shift({"q4_big_telescope": 2.5})
    near = compensation.place_shift({"q4_big_telescope": 1.0})
    assert out > near > 0.0


def test_the_flag_off_shifts_nothing() -> None:
    compensation = _compensation(enabled=False)
    compensation.store_pick({"q4_big_telescope": 1.0})

    # dz_pick is kept regardless, so switching the flag on mid-cycle still corrects
    # the difference and not the absolute sag.
    assert compensation.dz_pick > 0.0
    assert compensation.place_shift({"q4_big_telescope": 2.5}) == 0.0


def test_the_payload_raises_a_real_pose_by_a_few_centimetres() -> None:
    """The crane's own model, block mass at the TCP, against the sim-rigid nominal."""
    import pinocchio as pin
    from motion_planning.mechanics import ModelDescription, create_crane_config
    from motion_planning.mechanics.pinocchio_utils import arm_gravity_torque

    config = create_crane_config()
    description = ModelDescription(config)
    data = description.model.createData()
    frame_cache: dict[str, int] = {}

    def tau3(q_values, payload_mass):
        return arm_gravity_torque(
            pin_model=description.model,
            pin_data=data,
            pin_module=pin,
            q_values=q_values,
            payload_mass=payload_mass,
            frame_cache=frame_cache,
        )

    pose = {
        "theta1_slewing_joint": 0.0,
        "theta2_boom_joint": 0.3,
        "theta3_arm_joint": 1.2,
        "q4_big_telescope": 1.0,
        "q5_small_telescope": 1.0,
        "theta6_tip_joint": 0.0,
        "theta7_tilt_joint": 0.0,
        "theta8_rotator_joint": 0.0,
    }
    for name in config.locked_joints:
        pose.setdefault(name, 0.0)

    # The payload may only add torque: a sign slip here would correct downwards.
    assert tau3(pose, 880.0) > tau3(pose, 0.0) > 0.0

    compensation = ElasticityCompensation(
        p=P, payload_mass=880.0, tau3_fn=tau3, enabled=True
    )
    compensation.store_pick(pose)
    assert 0.01 < compensation.place_shift(pose) < 0.15
