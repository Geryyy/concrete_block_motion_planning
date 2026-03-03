from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pinocchio as pin

from .config import AnalyticModelConfig
from .pinocchio_utils import q_map_to_pin_q


@dataclass(frozen=True)
class PassiveAccelResult:
    qdd_passive: dict[str, float]
    q_full: dict[str, float]
    dq_full: dict[str, float]
    qdd_known_full: dict[str, float]


class ProjectedUnderactuatedDynamics:
    """Full-model passive acceleration via projected equations.

    Computes passive accelerations from full-model dynamics:
      M(q) qdd + h(q, dq) = tau
    with:
    - actuated accelerations provided as inputs,
    - tied joints enforced (follower = leader),
    - locked joints fixed at provided values with zero velocity/acceleration.
    """

    def __init__(self, config: AnalyticModelConfig) -> None:
        self._cfg = config
        self._model = pin.buildModelFromUrdf(config.urdf_path)
        self._data = self._model.createData()

        self._name_to_vidx = {
            str(self._model.names[jid]): int(self._model.joints[jid].idx_v)
            for jid in range(1, self._model.njoints)
        }
        self._act = [jn for jn in self._cfg.actuated_joints if jn in self._name_to_vidx]
        self._pas = [jn for jn in self._cfg.passive_joints if jn in self._name_to_vidx]
        self._dyn = [jn for jn in self._cfg.dynamic_joints if jn in self._name_to_vidx]
        self._locked = [jn for jn in self._cfg.locked_joints if jn in self._name_to_vidx]

        self._p_idx = [self._name_to_vidx[jn] for jn in self._pas]
        self._n_idx = [i for i in range(self._model.nv) if i not in set(self._p_idx)]
        self._damping = np.asarray(self._model.damping, dtype=float)

    @classmethod
    def from_model_path(cls, config: AnalyticModelConfig, model_path: str) -> "ProjectedUnderactuatedDynamics":
        self = cls.__new__(cls)
        self._cfg = config
        path = str(model_path)
        if path.lower().endswith(".urdf"):
            self._model = pin.buildModelFromUrdf(path)
        else:
            self._model = pin.buildModelFromMJCF(path)
        self._data = self._model.createData()
        self._name_to_vidx = {
            str(self._model.names[jid]): int(self._model.joints[jid].idx_v)
            for jid in range(1, self._model.njoints)
        }
        self._act = [jn for jn in self._cfg.actuated_joints if jn in self._name_to_vidx]
        self._pas = [jn for jn in self._cfg.passive_joints if jn in self._name_to_vidx]
        self._dyn = [jn for jn in self._cfg.dynamic_joints if jn in self._name_to_vidx]
        self._locked = [jn for jn in self._cfg.locked_joints if jn in self._name_to_vidx]
        self._p_idx = [self._name_to_vidx[jn] for jn in self._pas]
        self._n_idx = [i for i in range(self._model.nv) if i not in set(self._p_idx)]
        self._damping = np.asarray(self._model.damping, dtype=float)
        return self

    @property
    def model(self) -> pin.Model:
        return self._model

    def _build_full_state(
        self,
        *,
        q_act: Mapping[str, float],
        dq_act: Mapping[str, float],
        q_pas: Mapping[str, float],
        dq_pas: Mapping[str, float],
        qdd_act: Mapping[str, float],
        locked_q: Mapping[str, float] | None,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        q_full = {jn: 0.0 for jn in self._name_to_vidx}
        dq_full = {jn: 0.0 for jn in self._name_to_vidx}
        qdd_known = {jn: 0.0 for jn in self._name_to_vidx}

        if locked_q is not None:
            for jn in self._locked:
                if jn in locked_q:
                    q_full[jn] = float(locked_q[jn])
        for jn in self._act:
            if jn in q_act:
                q_full[jn] = float(q_act[jn])
            if jn in dq_act:
                dq_full[jn] = float(dq_act[jn])
            if jn in qdd_act:
                qdd_known[jn] = float(qdd_act[jn])
        for jn in self._pas:
            if jn in q_pas:
                q_full[jn] = float(q_pas[jn])
            if jn in dq_pas:
                dq_full[jn] = float(dq_pas[jn])

        # Fill remaining dynamic joints (if provided in passive maps etc.).
        for jn in self._dyn:
            if jn in q_act:
                q_full[jn] = float(q_act[jn])
            if jn in q_pas:
                q_full[jn] = float(q_pas[jn])
            if jn in dq_act:
                dq_full[jn] = float(dq_act[jn])
            if jn in dq_pas:
                dq_full[jn] = float(dq_pas[jn])

        # Enforce tied joints for q, dq, qdd.
        for follower, leader in self._cfg.tied_joints.items():
            if follower not in q_full or leader not in q_full:
                continue
            q_full[follower] = float(q_full[leader])
            dq_full[follower] = float(dq_full[leader])
            qdd_known[follower] = float(qdd_known[leader])
        return q_full, dq_full, qdd_known

    def compute_passive_acceleration(
        self,
        *,
        q_act: Mapping[str, float],
        dq_act: Mapping[str, float],
        q_pas: Mapping[str, float],
        dq_pas: Mapping[str, float],
        qdd_act: Mapping[str, float],
        locked_q: Mapping[str, float] | None = None,
    ) -> PassiveAccelResult:
        q_full_map, dq_full_map, qdd_known_map = self._build_full_state(
            q_act=q_act,
            dq_act=dq_act,
            q_pas=q_pas,
            dq_pas=dq_pas,
            qdd_act=qdd_act,
            locked_q=locked_q,
        )

        q_pin = q_map_to_pin_q(self._model, q_full_map, pin)
        dq_full = np.zeros(self._model.nv, dtype=float)
        qdd_known_full = np.zeros(self._model.nv, dtype=float)
        for jn, i in self._name_to_vidx.items():
            dq_full[i] = float(dq_full_map[jn])
            qdd_known_full[i] = float(qdd_known_map[jn])

        M = pin.crba(self._model, self._data, q_pin)
        M = 0.5 * (M + M.T)
        h = pin.nonLinearEffects(self._model, self._data, q_pin, dq_full) + self._damping * dq_full

        Mpp = M[np.ix_(self._p_idx, self._p_idx)]
        Mpn = M[np.ix_(self._p_idx, self._n_idx)]
        rhs = -(Mpn @ qdd_known_full[self._n_idx] + h[self._p_idx])
        qdd_p = np.linalg.solve(Mpp + 1e-8 * np.eye(len(self._p_idx)), rhs)

        qdd_passive = {jn: float(qdd_p[i]) for i, jn in enumerate(self._pas)}
        return PassiveAccelResult(
            qdd_passive=qdd_passive,
            q_full=q_full_map,
            dq_full=dq_full_map,
            qdd_known_full={jn: float(qdd_known_map[jn]) for jn in self._name_to_vidx},
        )
