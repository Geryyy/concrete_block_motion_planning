"""Loaded boom/telescope bend, as a correction on a grip target (issue 300).

Christoph's fit from the laser-tracker campaign (`+tools/polyval_elasticity.m`):
``dz = p1*q4 + p2*q4^2 + p3*tau3*q4``, with ``tau3`` the arm joint's gravity torque.
Fitted on empty runs, so the payload term extrapolates -- which is why the server
keeps it behind a flag.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

TELESCOPE_JOINT = "q4_big_telescope"


def elasticity_dz(tau3: float, q4: float, p: Sequence[float]) -> float:
    """Downward sag of the tool centre point [m]."""
    return p[0] * q4 + p[1] * q4 * q4 + p[2] * tau3 * q4


class ElasticityCompensation:
    """Sag at a pose, and the place shift it implies.

    The shift is the *change* since the pick, not the sag itself: the jaws close
    with the pick's sag already in the grip depth, so correcting the absolute sag
    would place the block that much high (3-7 cm, issue 294 Session 3).

    ``tau3_fn(q_values, payload_mass)`` returns the arm joint's gravity torque
    [N*m]; the model lives with the caller.
    """

    def __init__(
        self,
        *,
        p: Sequence[float],
        payload_mass: float,
        tau3_fn: Callable[[Mapping[str, float], float], float],
        enabled: bool,
    ) -> None:
        if len(p) != 3:
            raise ValueError(f"elasticity_p must have 3 coefficients, got {len(p)}")
        self._p = [float(value) for value in p]
        self.payload_mass = float(payload_mass)
        self._tau3_fn = tau3_fn
        self.enabled = bool(enabled)
        self.dz_pick = 0.0

    def dz(self, q_values: Mapping[str, float], payload_mass: float) -> float:
        return elasticity_dz(
            self._tau3_fn(q_values, payload_mass),
            float(q_values[TELESCOPE_JOINT]),
            self._p,
        )

    def store_pick(self, q_values: Mapping[str, float]) -> float:
        """Sag the jaws closed at; empty hook, so no payload term.

        Stored whatever ``enabled`` says, so switching the flag on between a pick and
        its place still corrects the difference and not the absolute sag.
        """
        self.dz_pick = self.dz(q_values, 0.0)
        return self.dz_pick

    def place_shift(self, q_values: Mapping[str, float]) -> float:
        """How much higher a loaded target has to be commanded [m]."""
        if not self.enabled:
            return 0.0
        return self.dz(q_values, self.payload_mass) - self.dz_pick
