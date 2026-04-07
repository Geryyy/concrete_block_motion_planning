from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pinocchio as pin

from .config import AnalyticModelConfig
from .pinocchio_utils import q_map_to_pin_q


@dataclass(frozen=True)
class SplitPassiveAccelResult:
    qdd_passive: dict[str, float]
    q_reduced: dict[str, float]
    dq_reduced: dict[str, float]


class SplitUnderactuatedDynamics:
    """Classic reduced split dynamics model for passive acceleration.

    Uses a reduced Pinocchio model with selected joints locked.
    """

    def __init__(
        self,
        config: AnalyticModelConfig,
        *,
        lock_tied_followers: bool = True,
    ) -> None:
        self._cfg = config
        full = pin.buildModelFromUrdf(config.urdf_path)
        name_to_jid = {str(full.names[jid]): jid for jid in range(1, full.njoints)}

        lock_names = list(config.locked_joints)
        if lock_tied_followers:
            for follower in config.tied_joints:
                if follower not in lock_names:
                    lock_names.append(follower)
        lock_ids = [name_to_jid[jn] for jn in lock_names if jn in name_to_jid]

        self._model = pin.buildReducedModel(full, lock_ids, pin.neutral(full))
        self._data = self._model.createData()
        self._name_to_vidx = {
            str(self._model.names[jid]): int(self._model.joints[jid].idx_v)
            for jid in range(1, self._model.njoints)
        }
        self._act = [jn for jn in self._cfg.actuated_joints if jn in self._name_to_vidx]
        self._pas = [jn for jn in self._cfg.passive_joints if jn in self._name_to_vidx]
        self._damping = np.asarray(self._model.damping, dtype=float)

        self._a_idx = [self._name_to_vidx[jn] for jn in self._act]
        self._p_idx = [self._name_to_vidx[jn] for jn in self._pas]

    @classmethod
    def from_model_path(
        cls,
        config: AnalyticModelConfig,
        model_path: str,
        *,
        lock_tied_followers: bool = True,
    ) -> "SplitUnderactuatedDynamics":
        self = cls.__new__(cls)
        self._cfg = config
        path = str(model_path)
        if path.lower().endswith(".urdf"):
            full = pin.buildModelFromUrdf(path)
        else:
            full = pin.buildModelFromMJCF(path)
        name_to_jid = {str(full.names[jid]): jid for jid in range(1, full.njoints)}
        lock_names = list(config.locked_joints)
        if lock_tied_followers:
            for follower in config.tied_joints:
                if follower not in lock_names:
                    lock_names.append(follower)
        lock_ids = [name_to_jid[jn] for jn in lock_names if jn in name_to_jid]
        self._model = pin.buildReducedModel(full, lock_ids, pin.neutral(full))
        self._data = self._model.createData()
        self._name_to_vidx = {
            str(self._model.names[jid]): int(self._model.joints[jid].idx_v)
            for jid in range(1, self._model.njoints)
        }
        self._act = [jn for jn in self._cfg.actuated_joints if jn in self._name_to_vidx]
        self._pas = [jn for jn in self._cfg.passive_joints if jn in self._name_to_vidx]
        self._damping = np.asarray(self._model.damping, dtype=float)
        self._a_idx = [self._name_to_vidx[jn] for jn in self._act]
        self._p_idx = [self._name_to_vidx[jn] for jn in self._pas]
        return self

    @property
    def model(self) -> pin.Model:
        return self._model

    def compute_passive_acceleration(
        self,
        *,
        q_act: Mapping[str, float],
        dq_act: Mapping[str, float],
        q_pas: Mapping[str, float],
        dq_pas: Mapping[str, float],
        qdd_act: Mapping[str, float],
    ) -> SplitPassiveAccelResult:
        q_map = {jn: 0.0 for jn in self._name_to_vidx}
        dq_map = {jn: 0.0 for jn in self._name_to_vidx}
        qdd_a = np.zeros(len(self._a_idx), dtype=float)

        for jn in self._act:
            if jn in q_act:
                q_map[jn] = float(q_act[jn])
            if jn in dq_act:
                dq_map[jn] = float(dq_act[jn])
        for jn in self._pas:
            if jn in q_pas:
                q_map[jn] = float(q_pas[jn])
            if jn in dq_pas:
                dq_map[jn] = float(dq_pas[jn])
        for i, jn in enumerate(self._act):
            qdd_a[i] = float(qdd_act.get(jn, 0.0))

        q_pin = q_map_to_pin_q(self._model, q_map, pin)
        dq = np.zeros(self._model.nv, dtype=float)
        for jn, i in self._name_to_vidx.items():
            dq[i] = float(dq_map[jn])

        M = pin.crba(self._model, self._data, q_pin)
        M = 0.5 * (M + M.T)
        h = pin.nonLinearEffects(self._model, self._data, q_pin, dq) + self._damping * dq

        Mpp = M[np.ix_(self._p_idx, self._p_idx)]
        Mpa = M[np.ix_(self._p_idx, self._a_idx)]
        rhs = -(Mpa @ qdd_a + h[self._p_idx])
        qdd_p = np.linalg.solve(Mpp + 1e-8 * np.eye(len(self._p_idx)), rhs)
        return SplitPassiveAccelResult(
            qdd_passive={jn: float(qdd_p[i]) for i, jn in enumerate(self._pas)},
            q_reduced=q_map,
            dq_reduced=dq_map,
        )
