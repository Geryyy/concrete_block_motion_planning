"""VP-STO + iLQR trajectory stack.

Pipeline:
  1. VP-STO  →  joint-space waypoints + total time T
  2. JointSplineReference  →  cubic spline over VP-STO waypoints
  3. iLQR  →  time-parameterised, smooth actuated trajectory
  4. SplitUnderactuatedDynamics  →  passive joint sway (theta6, theta7)
  5. FK  →  TCP path

The iLQR uses a simple double-integrator model for the 5 actuated joints.
Passive sway is propagated forward as a post-processing step.
"""

from __future__ import annotations

import numpy as np

from motion_planning.mechanics.analytic.config import AnalyticModelConfig
from motion_planning.mechanics.analytic.split_dynamics import SplitUnderactuatedDynamics
from motion_planning.pipeline import JointGoalStage, JointSpaceCartesianPlanner
from motion_planning.trajectory.ilqr import (
    DoubleIntegratorDynamics,
    ILQRSolver,
    JointSplineReference,
    TrackingCost,
)

from ..evaluate import evaluate_plan
from ..types import StandalonePlanResult, StandaloneScenario
from .vpsto_path_planning import plan_vpsto_path_planning, _fail as _vpsto_fail


# ---- iLQR hyper-parameters ----------------------------------------
_N_STEPS = 60          # number of iLQR steps
_Q_Q = 20.0            # joint position tracking weight
_Q_DQ = 0.5            # velocity regularisation
_R = 0.05              # acceleration cost
_QF_Q = 200.0          # terminal position weight
_QF_DQ = 5.0           # terminal velocity weight
# -------------------------------------------------------------------


def plan_vpsto_ilqr(scenario: StandaloneScenario) -> StandalonePlanResult:
    # ---- Step 1: VP-STO path ----
    vpsto = plan_vpsto_path_planning(scenario)
    if not vpsto.success:
        return _fail(f"VP-STO failed: {vpsto.message}")

    T = float(vpsto.diagnostics.get("vpsto_T", 5.0))
    q_waypoints = np.asarray(vpsto.q_waypoints, dtype=float)  # (M, 5)
    M, n_q = q_waypoints.shape

    if M < 2:
        return _fail("VP-STO returned too few waypoints for iLQR")

    # ---- Step 2: Reference spline ----
    ref = JointSplineReference(q_waypoints, T)
    Ts = T / _N_STEPS
    dyn = DoubleIntegratorDynamics(n_q=n_q, Ts=Ts)

    # ---- Build reference states x_refs (N+1, n_x) ----
    q_refs, dq_refs = ref.sample_uniform(_N_STEPS + 1)
    x_refs = np.hstack([q_refs, dq_refs])   # (N+1, 2*n_q)

    # ---- Initial state: q0 from VP-STO start, dq0 = 0 ----
    x0 = np.concatenate([q_waypoints[0], np.zeros(n_q)])

    # ---- Step 3: iLQR ----
    cost = TrackingCost(
        n_q=n_q,
        Q_q=_Q_Q,
        Q_dq=_Q_DQ,
        R=_R,
        Qf_q=_QF_Q,
        Qf_dq=_QF_DQ,
    )
    solver = ILQRSolver(dyn, cost, N=_N_STEPS, max_iter=30)
    ilqr_result = solver.solve(x0, x_refs)

    q_act = ilqr_result.q_traj       # (N+1, 5)
    dq_act = ilqr_result.dq_traj     # (N+1, 5)
    qdd_act = ilqr_result.qdd_traj   # (N, 5)
    time_s = ilqr_result.time_s      # (N+1,)

    # ---- Step 4: Passive joint propagation ----
    q_pas_traj, dq_pas_traj = _propagate_passive(
        q_act, dq_act, qdd_act, Ts, time_s.shape[0]
    )
    # q_pas_traj: (N+1, 2) = [theta6, theta7]

    # ---- Step 5: FK for TCP path ----
    stage = JointGoalStage()
    act_names = list(stage.config.actuated_joints)
    planner = JointSpaceCartesianPlanner(
        urdf_path=stage.config.urdf_path,
        target_frame=stage.config.target_frame,
        reduced_joint_names=act_names,
    )

    xyz_list = []
    yaw_list = []
    q_seed: dict[str, float] = {}
    for q in q_act:
        xyz_i, yaw_i, q_seed = planner.fk_world_pose(q, q_seed=q_seed)
        xyz_list.append(xyz_i)
        yaw_list.append(yaw_i)

    tcp_xyz = np.asarray(xyz_list, dtype=float)
    tcp_yaw = np.asarray(yaw_list, dtype=float)

    # ---- Build full 7-DOF q_waypoints for joint plot ----
    q_full = np.hstack([q_act[:, :4], q_pas_traj, q_act[:, 4:5]])
    # Order: [th1, th2, th3, q4, th6, th7, th8] — matches URDF dynamic_joints

    # Re-sample VP-STO reference to match iLQR output length
    n_out = tcp_xyz.shape[0]
    ref_xyz_vp = np.asarray(vpsto.reference_xyz, dtype=float).reshape(-1, 3)
    ref_yaw_vp = np.asarray(vpsto.reference_yaw_rad, dtype=float).ravel()
    t_vp = np.linspace(0.0, 1.0, ref_xyz_vp.shape[0])
    t_out = np.linspace(0.0, 1.0, n_out)
    ref_xyz_rs = np.column_stack([
        np.interp(t_out, t_vp, ref_xyz_vp[:, i]) for i in range(3)
    ])
    ref_yaw_rs = np.interp(t_out, t_vp, ref_yaw_vp)

    result = StandalonePlanResult(
        stack_name="vpsto_ilqr",
        success=True,
        message=(
            f"VP-STO+iLQR: T={T:.2f}s, {ilqr_result.iterations} iLQR iters, "
            f"cost={ilqr_result.cost:.3f}"
        ),
        q_waypoints=q_full,
        tcp_xyz=tcp_xyz,
        tcp_yaw_rad=tcp_yaw,
        reference_xyz=ref_xyz_rs,
        reference_yaw_rad=ref_yaw_rs,
        time_s=time_s,
        dq_waypoints=np.hstack([dq_act[:, :4], np.zeros((dq_act.shape[0], 2)), dq_act[:, 4:5]]),
        diagnostics={
            "vpsto_T": T,
            "ilqr_cost": ilqr_result.cost,
            "ilqr_iterations": float(ilqr_result.iterations),
            "waypoint_count": float(q_full.shape[0]),
            "passive_max_th6_deg": float(np.max(np.abs(q_pas_traj[:, 0])) * 180.0 / np.pi),
            "passive_max_th7_deg": float(np.max(np.abs(q_pas_traj[:, 1])) * 180.0 / np.pi),
        },
    )
    evaluate_plan(result)
    return result


def _compute_passive_equilibrium(
    dyn: SplitUnderactuatedDynamics,
    q_act_map: dict[str, float],
    pas_names: list[str],
    max_iter: int = 300,
    lr: float = 0.05,
) -> dict[str, float]:
    """Find static equilibrium of passive joints via gradient descent.

    Solves g_p(q_act, q_pas) = 0 by simulating a damped passive system
    from the pinocchio neutral position.  Returns the equilibrium q_pas.
    """
    q_pas = {jn: 0.0 for jn in pas_names}
    dq_pas = {jn: 0.0 for jn in pas_names}
    zero_act = {jn: 0.0 for jn in q_act_map}  # zero velocity
    zero_qdd = {jn: 0.0 for jn in q_act_map}

    for _ in range(max_iter):
        try:
            res = dyn.compute_passive_acceleration(
                q_act=q_act_map,
                dq_act=zero_act,
                q_pas=q_pas,
                dq_pas=dq_pas,
                qdd_act=zero_qdd,
            )
            qdd_pas = res.qdd_passive
        except Exception:
            break
        mag = max(abs(v) for v in qdd_pas.values()) if qdd_pas else 0.0
        if mag < 1e-6:
            break
        for jn in pas_names:
            q_pas[jn] += lr * qdd_pas.get(jn, 0.0)
    return q_pas


def _propagate_passive(
    q_act: np.ndarray,
    dq_act: np.ndarray,
    qdd_act: np.ndarray,
    Ts: float,
    N1: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-simulate passive joints (theta6, theta7) via split dynamics.

    Parameters
    ----------
    q_act:   (N+1, 5) actuated joint positions  [th1,th2,th3,q4,th8]
    dq_act:  (N+1, 5) actuated joint velocities
    qdd_act: (N,   5) actuated joint accelerations (controls)
    Ts:      timestep [s]
    N1:      N+1 (number of output rows)

    Returns
    -------
    q_pas_traj:  (N+1, 2)  passive positions  [th6, th7]
    dq_pas_traj: (N+1, 2)  passive velocities
    """
    try:
        config = AnalyticModelConfig.default()
        dyn = SplitUnderactuatedDynamics(config)
    except Exception:
        # If pinocchio / URDF unavailable, return zeros
        return np.zeros((N1, 2)), np.zeros((N1, 2))

    act_names = [
        "theta1_slewing_joint",
        "theta2_boom_joint",
        "theta3_arm_joint",
        "q4_big_telescope",
        "theta8_rotator_joint",
    ]
    pas_names = ["theta6_tip_joint", "theta7_tilt_joint"]

    N = qdd_act.shape[0]
    q_pas_traj = np.zeros((N1, 2))
    dq_pas_traj = np.zeros((N1, 2))

    # Start from pinocchio equilibrium (zeros — passive joints are near 0
    # in CBS; a more accurate start could be computed from steady_state)
    # Compute passive equilibrium at initial actuated config
    q_act0_map = {name: float(q_act[0, i]) for i, name in enumerate(act_names)}
    q_pas = _compute_passive_equilibrium(dyn, q_act0_map, pas_names)
    dq_pas = {jn: 0.0 for jn in pas_names}

    q_pas_traj[0] = [q_pas[jn] for jn in pas_names]
    dq_pas_traj[0] = [dq_pas[jn] for jn in pas_names]

    for k in range(N):
        q_act_map = {name: float(q_act[k, i]) for i, name in enumerate(act_names)}
        dq_act_map = {name: float(dq_act[k, i]) for i, name in enumerate(act_names)}
        qdd_act_map = {name: float(qdd_act[k, i]) for i, name in enumerate(act_names)}

        try:
            res = dyn.compute_passive_acceleration(
                q_act=q_act_map,
                dq_act=dq_act_map,
                q_pas=q_pas,
                dq_pas=dq_pas,
                qdd_act=qdd_act_map,
            )
            qdd_pas = res.qdd_passive
        except Exception:
            qdd_pas = {jn: 0.0 for jn in pas_names}

        for jn in pas_names:
            qdd = float(qdd_pas.get(jn, 0.0))
            # Clamp to avoid instability
            qdd = float(np.clip(qdd, -10.0, 10.0))
            q_pas[jn] += Ts * dq_pas[jn] + 0.5 * Ts ** 2 * qdd
            dq_pas[jn] += Ts * qdd

        q_pas_traj[k + 1] = [q_pas[jn] for jn in pas_names]
        dq_pas_traj[k + 1] = [dq_pas[jn] for jn in pas_names]

    return q_pas_traj, dq_pas_traj


def _fail(msg: str) -> StandalonePlanResult:
    return StandalonePlanResult(
        stack_name="vpsto_ilqr",
        success=False,
        message=msg,
        q_waypoints=np.zeros((0, 7), dtype=float),
        tcp_xyz=np.zeros((0, 3), dtype=float),
        tcp_yaw_rad=np.zeros(0, dtype=float),
        reference_xyz=np.zeros((0, 3), dtype=float),
        reference_yaw_rad=np.zeros(0, dtype=float),
        diagnostics={},
    )
