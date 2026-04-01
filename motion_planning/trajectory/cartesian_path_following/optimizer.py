from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import inspect

import numpy as np

from motion_planning.core.types import TrajectoryRequest, TrajectoryResult
from motion_planning.trajectory.base import TrajectoryOptimizer
from motion_planning.trajectory.dynamics import build_underactuated_qdd_symbolic
from motion_planning.trajectory.limits import prepare_control_bounds_from_limits
from motion_planning.trajectory.planning_limits import load_planning_limits_yaml
from motion_planning.trajectory.cartesian_path_following.config import CartesianPathFollowingConfig
from motion_planning.trajectory.path_following.spline import (
    bspline_basis_all_symbolic,
    clamped_uniform_knots,
)

_FREE_TIME_WEIGHT_DEFAULT = 1e-3
_FREE_TIME_T_MIN_DEFAULT = 0.5
_FREE_TIME_T_MAX_DEFAULT = 60.0


def _required_deps():
    import casadi as ca
    import pinocchio as pin
    from pinocchio import casadi as cpin
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

    return ca, pin, cpin, AcadosModel, AcadosOcp, AcadosOcpSolver


def _as_float_array(values: Sequence[float], size: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.shape[0] != size:
        raise ValueError(f"{name} must have length {size}, got {arr.shape[0]}.")
    return arr


def _polyline_length(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _derive_fixed_time_candidates(
    *,
    ctrl_pts_xyz: np.ndarray,
    T_min: float,
    T_max: float,
    requested_duration: Optional[float],
    requested_candidates: Sequence[float],
    nominal_tcp_speed: float,
    min_duration_s: float,
    max_duration_s: float,
    sway_slack_s: float,
) -> tuple[np.ndarray, float]:
    if requested_candidates:
        values = np.asarray(requested_candidates, dtype=float).reshape(-1)
    elif requested_duration is not None:
        values = np.asarray([float(requested_duration)], dtype=float)
    else:
        path_length = _polyline_length(ctrl_pts_xyz)
        base = path_length / max(float(nominal_tcp_speed), 1e-3) + float(sway_slack_s)
        nominal = float(np.clip(base, float(min_duration_s), float(max_duration_s)))
        values = np.asarray([0.85 * nominal, nominal, 1.2 * nominal], dtype=float)

    clipped = np.clip(values, float(T_min), float(T_max))
    rounded_unique = []
    for value in clipped.tolist():
        value = round(float(value), 6)
        if value not in rounded_unique:
            rounded_unique.append(value)
    if not rounded_unique:
        midpoint = round(float(0.5 * (T_min + T_max)), 6)
        rounded_unique.append(midpoint)
    arr = np.asarray(rounded_unique, dtype=float)
    return arr, float(arr[min(1, arr.shape[0] - 1)])


def _progress_warm_start(
    *,
    N: int,
    s0: float,
    sdot0: float,
    T_guess: float,
    warm_start_progress: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if N <= 0:
        return np.asarray([s0], dtype=float), np.asarray([sdot0], dtype=float)

    tau = np.linspace(0.0, 1.0, N + 1, dtype=float)
    if not warm_start_progress:
        s = np.linspace(float(s0), 1.0, N + 1, dtype=float)
        sdot = np.gradient(s, tau, edge_order=1) / max(float(T_guess), 1e-6)
        sdot[0] = float(sdot0)
        return s, sdot

    start = float(np.clip(s0, 0.0, 1.0))
    remaining = max(1e-6, 1.0 - start)
    smooth = 3.0 * tau**2 - 2.0 * tau**3
    ds_dtau = 6.0 * tau * (1.0 - tau)
    s = start + remaining * smooth
    sdot = remaining * ds_dtau / max(float(T_guess), 1e-6)
    sdot[0] = float(sdot0)
    s[-1] = 1.0
    sdot[-1] = 0.0
    return s, sdot


def _fk_num(pin, model, data, q_dec: np.ndarray, tool_fid: int) -> np.ndarray:
    """Numeric FK: decision q → TCP xyz (3,)."""
    nq_pin = int(model.nq)
    q_pin = np.asarray(pin.neutral(model), dtype=float)
    for jid in range(1, model.njoints):
        jm = model.joints[jid]
        iv = int(jm.idx_v)
        iq = int(jm.idx_q)
        if int(jm.nq) == 1:
            q_pin[iq] = q_dec[iv]
        elif int(jm.nq) == 2 and int(jm.nv) == 1:
            th = q_dec[iv]
            q_pin[iq] = np.cos(th)
            q_pin[iq + 1] = np.sin(th)
    pin.forwardKinematics(model, data, q_pin)
    pin.updateFramePlacements(model, data)
    return np.asarray(data.oMf[tool_fid].translation, dtype=float).copy()


def _fk_yaw_num(pin, model, data, q_dec: np.ndarray, tool_fid: int) -> float:
    """Numeric FK: decision q → timber ``phiTool`` angle (scalar)."""
    nq_pin = int(model.nq)
    q_pin = np.asarray(pin.neutral(model), dtype=float)
    for jid in range(1, model.njoints):
        jm = model.joints[jid]
        iv = int(jm.idx_v)
        iq = int(jm.idx_q)
        if int(jm.nq) == 1:
            q_pin[iq] = q_dec[iv]
        elif int(jm.nq) == 2 and int(jm.nv) == 1:
            th = q_dec[iv]
            q_pin[iq] = np.cos(th)
            q_pin[iq + 1] = np.sin(th)
    pin.forwardKinematics(model, data, q_pin)
    pin.updateFramePlacements(model, data)
    R = np.asarray(data.oMf[tool_fid].rotation, dtype=float)
    yaw = np.arctan2(R[1, 1], R[0, 1])
    return float(yaw)



def _bspline_eval_symbolic_param(s, ctrl_flat, n_ctrl: int, degree: int, dim: int = 3):
    """Evaluate spline with symbolic control points ``ctrl_flat`` (n_ctrl*dim,)."""
    if n_ctrl == 4 and degree == 3:
        t = s
        omt = 1.0 - t
        b0 = omt * omt * omt
        b1 = 3.0 * omt * omt * t
        b2 = 3.0 * omt * t * t
        b3 = t * t * t
        c0 = ctrl_flat[0:dim]
        c1 = ctrl_flat[dim : 2 * dim]
        c2 = ctrl_flat[2 * dim : 3 * dim]
        c3 = ctrl_flat[3 * dim : 4 * dim]
        return b0 * c0 + b1 * c1 + b2 * c2 + b3 * c3

    knots = clamped_uniform_knots(n_ctrl=n_ctrl, degree=degree)
    basis = bspline_basis_all_symbolic(s=s, knots=knots, degree=degree, n_ctrl=n_ctrl)
    y = 0.0 * ctrl_flat[0:dim]
    for i in range(n_ctrl):
        y = y + basis[i] * ctrl_flat[i * dim : (i + 1) * dim]
    return y


def _yaw_from_rotation_matrix_symbolic(R):
    """Extract timber ``phiTool`` from a symbolic rotation matrix."""
    import casadi as ca
    return ca.atan2(R[1, 1], R[0, 1])



class CartesianPathFollowingOptimizer(TrajectoryOptimizer):
    """Task-space path-following OCP with Cartesian FK cost.

    State:  x = [q_dec(nv), dq(nv), s, sdot]
    Input:  u = [qdd_actuated(n_act), v]   where v = s_ddot

    The Cartesian reference path ``xyz_ref(s)`` is a B-spline in task space
    built from control points sampled from ``req.path`` (BSplinePath) or
    supplied explicitly via ``req.config["ctrl_pts_xyz"]``.

    The running cost tracks ``||FK(q) - xyz_ref(s)||^2`` using symbolic FK via
    pinocchio.casadi, following the pycrane tspfc formulation.
    """

    def __init__(self, config: CartesianPathFollowingConfig):
        self.config = config
        self._solver = None
        self._solver_key: Optional[Tuple[Any, ...]] = None
        if bool(self.config.precompile_on_init):
            self._precompile_solver()

    def _precompile_solver(self) -> None:
        """Compile solver once at object creation using a dummy request."""
        n_ctrl = int(self.config.spline_ctrl_points)
        dummy_ctrl = np.zeros((n_ctrl, 3), dtype=float)
        dummy_req = TrajectoryRequest(
            scenario=None,
            path=None,
            config={
                "ctrl_pts_xyz": dummy_ctrl,
                "__compile_only": True,
            },
        )
        self.optimize(dummy_req)

    @staticmethod
    def _joint_meta(model) -> List[Dict[str, int]]:
        meta: List[Dict[str, int]] = []
        for jid in range(1, model.njoints):
            jmodel = model.joints[jid]
            if int(jmodel.nv) != 1:
                raise ValueError(
                    f"Joint '{model.names[jid]}' has nv={jmodel.nv}; only nv=1 joints supported."
                )
            meta.append(
                {
                    "jid": int(jid),
                    "name": str(model.names[jid]),
                    "idx_q": int(jmodel.idx_q),
                    "nq": int(jmodel.nq),
                    "idx_v": int(jmodel.idx_v),
                }
            )
        return meta

    def optimize(self, req: TrajectoryRequest) -> TrajectoryResult:
        ca, pin, cpin, AcadosModel, AcadosOcp, AcadosOcpSolver = _required_deps()
        cfg = self.config
        urdf_path = Path(cfg.urdf_path).expanduser().resolve()
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        # ── Build reduced model ───────────────────────────────────────────────
        full_model = pin.buildModelFromUrdf(str(urdf_path))
        full_meta = self._joint_meta(full_model)
        full_name_to_jid = {m["name"]: m["jid"] for m in full_meta}
        passive_set = set(cfg.passive_joints)
        actuated_set = set(cfg.actuated_joints)
        if passive_set & actuated_set:
            raise ValueError(
                f"Joints cannot be both passive and actuated: {sorted(passive_set & actuated_set)}"
            )

        lock_joint_ids: List[int] = []
        for name in cfg.lock_joint_names:
            if name in full_name_to_jid:
                lock_joint_ids.append(int(full_name_to_jid[name]))
            elif cfg.print_model_prep:
                print(f"[cartesian-pfc] warning: lock joint '{name}' not found.")

        reduced_model = pin.buildReducedModel(
            full_model, lock_joint_ids, pin.neutral(full_model)
        )
        model = reduced_model
        reduced_meta = self._joint_meta(model)
        keep_names = [m["name"] for m in reduced_meta]
        keep_set = set(keep_names)
        if not passive_set.issubset(keep_set):
            missing = sorted(passive_set - keep_set)
            raise ValueError(f"Passive joints removed by reduction: {missing}")
        if cfg.print_model_prep:
            print("[cartesian-pfc] reduced joints:", keep_names)

        # ── Tool frame ID ─────────────────────────────────────────────────────
        tool_fid = model.getFrameId(cfg.tool_frame_name)
        if tool_fid >= model.nframes:
            raise ValueError(
                f"Frame '{cfg.tool_frame_name}' not found in reduced model. "
                f"Available frames: {[f.name for f in model.frames]}"
            )

        # ── CasADi model ──────────────────────────────────────────────────────
        cmodel = cpin.Model(model)
        cdata = cmodel.createData()
        nq_pin = int(model.nq)
        nv = int(model.nv)
        nqd = nv
        joint_meta = reduced_meta

        def q_dec_to_q_pin_sym(q_dec):
            q_pin = ca.SX.zeros(nq_pin, 1)
            for jm in joint_meta:
                iq = jm["idx_q"]
                iv = jm["idx_v"]
                if jm["nq"] == 1:
                    q_pin[iq] = q_dec[iv]
                elif jm["nq"] == 2:
                    th = q_dec[iv]
                    q_pin[iq] = ca.cos(th)
                    q_pin[iq + 1] = ca.sin(th)
            return q_pin

        # ── Parse q0, q_goal, dq0 ────────────────────────────────────────────
        q0_in = np.asarray(req.config.get("q0", np.zeros(nqd)), dtype=float).reshape(-1)
        if q0_in.shape[0] != nqd:
            raise ValueError(f"q0 length must be reduced nv={nqd}, got {q0_in.shape[0]}.")
        q0 = q0_in
        q_goal_in = np.asarray(req.config.get("q_goal", q0.copy()), dtype=float).reshape(-1)
        if q_goal_in.shape[0] != nqd:
            raise ValueError(f"q_goal length must be reduced nv={nqd}, got {q_goal_in.shape[0]}.")
        q_goal = q_goal_in
        dq0_in = np.asarray(req.config.get("dq0", np.zeros(nv)), dtype=float).reshape(-1)
        if dq0_in.shape[0] != nv:
            raise ValueError(f"dq0 length must be reduced nv={nv}, got {dq0_in.shape[0]}.")
        dq0 = dq0_in
        q0 = _as_float_array(q0, nqd, "q0")
        q_goal = _as_float_array(q_goal, nqd, "q_goal")
        dq0 = _as_float_array(dq0, nv, "dq0")

        N = int(req.config.get("horizon_steps", cfg.horizon_steps))
        dt_tau = 1.0 / max(N, 1)

        # ── Cartesian control points ──────────────────────────────────────────
        spline_degree = int(req.config.get("spline_degree", cfg.spline_degree))
        n_ctrl_cfg = int(req.config.get("spline_ctrl_points", cfg.spline_ctrl_points))
        ctrl_pts_xyz = req.config.get("ctrl_pts_xyz", None)
        if ctrl_pts_xyz is not None:
            ctrl_pts_xyz = np.asarray(ctrl_pts_xyz, dtype=float)
            if ctrl_pts_xyz.ndim != 2 or ctrl_pts_xyz.shape[1] != 3:
                raise ValueError(
                    f"ctrl_pts_xyz must have shape (n_ctrl, 3), got {ctrl_pts_xyz.shape}."
                )
            if "spline_ctrl_points" in req.config and ctrl_pts_xyz.shape[0] != n_ctrl_cfg:
                raise ValueError(
                    f"ctrl_pts_xyz rows ({ctrl_pts_xyz.shape[0]}) must match "
                    f"spline_ctrl_points ({n_ctrl_cfg}) when both are set."
                )
            n_ctrl = int(ctrl_pts_xyz.shape[0])
        elif req.path is not None:
            # Use a denser Cartesian reference from the geometric path to avoid
            # flattening curved paths into near-straight references.
            path_ref_points = int(req.config.get("path_ref_points", n_ctrl_cfg))
            path_ref_points = max(path_ref_points, spline_degree + 1)
            ctrl_pts_xyz = req.path.sample(path_ref_points)  # (n_ctrl, 3)
            n_ctrl = int(ctrl_pts_xyz.shape[0])
        else:
            # Fallback: straight line between FK(q0) and FK(q_goal).
            n_ctrl = n_ctrl_cfg
            red_data = model.createData()
            xyz0 = _fk_num(pin, model, red_data, q0, tool_fid)
            xyzN = _fk_num(pin, model, red_data, q_goal, tool_fid)
            alphas = np.linspace(0.0, 1.0, n_ctrl)
            ctrl_pts_xyz = np.vstack([(1.0 - a) * xyz0 + a * xyzN for a in alphas])
        if n_ctrl < spline_degree + 1:
            raise ValueError(
                f"Need at least degree+1 control points, got n_ctrl={n_ctrl}, degree={spline_degree}."
            )

        # ── Yaw control points ─────────────────────────────────────────────────
        ctrl_pts_yaw = req.config.get("ctrl_pts_yaw", None)
        if ctrl_pts_yaw is not None:
            ctrl_pts_yaw = np.asarray(ctrl_pts_yaw, dtype=float).reshape(-1)
            if ctrl_pts_yaw.shape[0] != n_ctrl:
                raise ValueError(
                    f"ctrl_pts_yaw length ({ctrl_pts_yaw.shape[0]}) must match "
                    f"n_ctrl ({n_ctrl})."
                )
        else:
            # Fallback: interpolate yaw between FK(q0) and FK(q_goal)
            red_data = model.createData()
            yaw0 = _fk_yaw_num(pin, model, red_data, q0, tool_fid)
            yawN = _fk_yaw_num(pin, model, red_data, q_goal, tool_fid)
            alphas = np.linspace(0.0, 1.0, n_ctrl)
            ctrl_pts_yaw = np.asarray([(1.0 - a) * yaw0 + a * yawN for a in alphas], dtype=float)


        # ── Joint indices ─────────────────────────────────────────────────────
        name_to_joint_id = {m["name"]: m["jid"] for m in joint_meta}
        reduced_name_to_vidx = {m["name"]: int(m["idx_v"]) for m in joint_meta}
        act_joint_ids = [int(name_to_joint_id[name]) for name in cfg.actuated_joints]
        act_v_idx = [int(model.joints[jid].idx_v) for jid in act_joint_ids]
        passive_v_idx = [reduced_name_to_vidx[name] for name in sorted(passive_set & keep_set)]
        n_act = len(act_v_idx)

        optimize_time = bool(req.config.get("optimize_time", cfg.optimize_time))
        T_min = float(req.config.get("T_min", _FREE_TIME_T_MIN_DEFAULT))
        T_max = float(req.config.get("T_max", _FREE_TIME_T_MAX_DEFAULT))
        if not (T_min > 0.0 and T_min <= T_max):
            raise ValueError(f"Invalid free-time bounds: require 0 < T_min <= T_max, got {T_min}, {T_max}.")
        fixed_time_candidates = ()
        if not optimize_time:
            fixed_time_candidates, T_guess = _derive_fixed_time_candidates(
                ctrl_pts_xyz=np.asarray(ctrl_pts_xyz, dtype=float),
                T_min=T_min,
                T_max=T_max,
                requested_duration=(
                    None
                    if req.config.get("fixed_time_duration_s", cfg.fixed_time_duration_s) is None
                    else float(req.config.get("fixed_time_duration_s", cfg.fixed_time_duration_s))
                ),
                requested_candidates=req.config.get(
                    "fixed_time_duration_candidates",
                    cfg.fixed_time_duration_candidates,
                ),
                nominal_tcp_speed=float(
                    req.config.get(
                        "fixed_time_nominal_tcp_speed",
                        cfg.fixed_time_nominal_tcp_speed,
                    )
                ),
                min_duration_s=float(
                    req.config.get(
                        "fixed_time_min_duration_s",
                        cfg.fixed_time_min_duration_s,
                    )
                ),
                max_duration_s=float(
                    req.config.get(
                        "fixed_time_max_duration_s",
                        cfg.fixed_time_max_duration_s,
                    )
                ),
                sway_slack_s=float(
                    req.config.get(
                        "fixed_time_sway_slack_s",
                        cfg.fixed_time_sway_slack_s,
                    )
                ),
            )
        else:
            T_guess = 0.5 * (T_min + T_max)
        payload_mass_default = float(req.config.get("payload_mass", cfg.payload_mass))
        payload_com_default = np.asarray(req.config.get("payload_com_tcp", cfg.payload_com_tcp), dtype=float).reshape(-1)
        if payload_com_default.shape[0] != 3:
            raise ValueError(f"payload_com_tcp must have length 3, got {payload_com_default.shape[0]}.")
        g_world = np.asarray(req.config.get("gravity_world", [0.0, 0.0, -9.81]), dtype=float).reshape(-1)
        if g_world.shape[0] != 3:
            raise ValueError(f"gravity_world must have length 3, got {g_world.shape[0]}.")

        # ── CasADi state / input symbols ─────────────────────────────────────
        q = ca.SX.sym("q", nqd)
        dq = ca.SX.sym("dq", nv)
        s = ca.SX.sym("s")
        sdot = ca.SX.sym("sdot")
        T = ca.SX.sym("T")
        x = ca.vertcat(q, dq, s, sdot, T)

        u_qdd = ca.SX.sym("u_qdd", n_act)
        v = ca.SX.sym("v")
        u = ca.vertcat(u_qdd, v)
        xdot = ca.SX.sym("xdot", nqd + nv + 3)

        # ── Symbolic FK → TCP position and yaw ───────────────────────────────
        q_pin = q_dec_to_q_pin_sym(q)
        cpin.forwardKinematics(cmodel, cdata, q_pin)
        cpin.updateFramePlacements(cmodel, cdata)
        xyz_tcp = cdata.oMf[tool_fid].translation  # (3,) CasADi symbolic
        R_tcp = cdata.oMf[tool_fid].rotation  # (3, 3) rotation matrix
        yaw_tcp = _yaw_from_rotation_matrix_symbolic(R_tcp)  # scalar yaw angle

        # ── Cartesian reference spline ────────────────────────────────────────
        n_pas = len(passive_v_idx)
        n_ctrl_params = n_ctrl * 3
        n_yaw_params = n_ctrl
        p = ca.SX.sym("p", n_ctrl_params + n_yaw_params + n_pas + 4)
        p_ctrl = p[:n_ctrl_params]
        p_yaw_ctrl = p[n_ctrl_params : n_ctrl_params + n_yaw_params]
        p_passive_eq = p[n_ctrl_params + n_yaw_params : n_ctrl_params + n_yaw_params + n_pas]
        p_payload = p[n_ctrl_params + n_yaw_params + n_pas :]
        xyz_ref_s = _bspline_eval_symbolic_param(
            s=s, ctrl_flat=p_ctrl, n_ctrl=n_ctrl, degree=spline_degree, dim=3
        )
        yaw_ref_s_expr = _bspline_eval_symbolic_param(
            s=s, ctrl_flat=p_yaw_ctrl, n_ctrl=n_ctrl, degree=spline_degree, dim=1
        )
        yaw_ref_s = yaw_ref_s_expr[0]  # Extract scalar from (1,) vector
        xyz_err = xyz_tcp - xyz_ref_s  # (3,)
        yaw_err = yaw_tcp - yaw_ref_s  # scalar
        payload_mass = p_payload[0]
        payload_com_tcp = p_payload[1:4]
        tool_body_id = int(model.frames[tool_fid].parentJoint)
        tcp_in_body = model.frames[tool_fid].placement
        p_b_tcp = ca.DM(np.asarray(tcp_in_body.translation, dtype=float).reshape(3))
        R_b_tcp = ca.DM(np.asarray(tcp_in_body.rotation, dtype=float))
        payload_com_body = p_b_tcp + R_b_tcp @ payload_com_tcp
        cmodel_dyn = cmodel.copy()
        original_inertia = cmodel_dyn.inertias[tool_body_id]
        total_mass = original_inertia.mass + payload_mass
        total_com = (original_inertia.mass * original_inertia.lever + payload_mass * payload_com_body) / (total_mass + 1e-9)
        cmodel_dyn.inertias[tool_body_id] = cpin.Inertia(total_mass, total_com, original_inertia.inertia)
        cdata_dyn = cmodel_dyn.createData()

        # ── Dynamics (projected or split) ─────────────────────────────────────
        dynamics_mode = str(req.config.get("dynamics_mode", cfg.dynamics_mode)).lower()
        if "split_passive_dynamics" in req.config:
            dynamics_mode = "split" if bool(req.config["split_passive_dynamics"]) else "projected"
        if dynamics_mode not in {"split", "projected"}:
            raise ValueError(f"Unsupported dynamics_mode '{dynamics_mode}'.")

        qdd = build_underactuated_qdd_symbolic(
            ca=ca,
            cpin=cpin,
            cmodel=cmodel_dyn,
            cdata=cdata_dyn,
            model=model,
            q_pin=q_pin,
            dq=dq,
            u_qdd=u_qdd,
            act_v_idx=act_v_idx,
            passive_v_idx=passive_v_idx,
            dynamics_mode=dynamics_mode,
            passive_solve_damping=float(req.config.get("passive_solve_damping", cfg.passive_solve_damping)),
        )

        f_expl = ca.vertcat(T * ca.vertcat(dq, qdd, sdot, v), 0.0)
        f_impl = xdot - f_expl

        # ── Cost weights ──────────────────────────────────────────────────────
        sdot_ref = float(req.config.get("sdot_ref", cfg.sdot_ref))
        xyz_w = float(req.config.get("xyz_weight", cfg.xyz_weight))
        xyz_wN = float(req.config.get("terminal_xyz_weight", cfg.terminal_xyz_weight))
        yaw_w = float(req.config.get("yaw_weight", cfg.yaw_weight))
        yaw_wN = float(req.config.get("terminal_yaw_weight", cfg.terminal_yaw_weight))
        s_w = float(req.config.get("s_weight", cfg.s_weight))
        sdot_w = float(req.config.get("sdot_weight", cfg.sdot_weight))
        u_w = float(req.config.get("qdd_u_weight", cfg.qdd_u_weight))
        v_w = float(req.config.get("v_weight", cfg.v_weight))
        sN_w = float(req.config.get("terminal_s_weight", cfg.terminal_s_weight))
        sdotN_w = float(req.config.get("terminal_sdot_weight", cfg.terminal_sdot_weight))
        dqN_w = float(req.config.get("terminal_dq_weight", cfg.terminal_dq_weight))
        passive_q_w = float(req.config.get("passive_q_sway_weight", cfg.passive_q_sway_weight))
        passive_dq_w = float(req.config.get("passive_dq_sway_weight", cfg.passive_dq_sway_weight))
        passive_qN_w = float(req.config.get("terminal_passive_q_sway_weight", cfg.terminal_passive_q_sway_weight))
        passive_dqN_w = float(req.config.get("terminal_passive_dq_sway_weight", cfg.terminal_passive_dq_sway_weight))
        passive_dq_soft_max = float(req.config.get("passive_dq_soft_max", cfg.passive_dq_soft_max))
        passive_dq_use_slack = bool(req.config.get("passive_dq_use_slack", cfg.passive_dq_use_slack))
        passive_dq_slack_w = float(req.config.get("passive_dq_slack_weight", cfg.passive_dq_slack_weight))
        passive_dq_slackN_w = float(
            req.config.get("terminal_passive_dq_slack_weight", cfg.terminal_passive_dq_slack_weight)
        )
        passive_dq_soft_eps = float(req.config.get("passive_dq_soft_abs_eps", cfg.passive_dq_soft_abs_eps))
        if passive_dq_use_slack:
            if passive_dq_soft_max <= 0.0:
                raise ValueError(f"passive_dq_soft_max must be > 0, got {passive_dq_soft_max}.")
            if passive_dq_soft_eps <= 0.0:
                raise ValueError(f"passive_dq_soft_abs_eps must be > 0, got {passive_dq_soft_eps}.")

        # Passive sway: penalise deviation from initial passive joint position (anti-sway)
        # and passive joint velocity (anti-swing damping). No joint-space path reference
        # is available for the Cartesian optimizer, so we use q0 as the sway equilibrium.
        if n_pas > 0:
            q_passive = ca.vertcat(*[q[i] for i in passive_v_idx])
            dq_passive = ca.vertcat(*[dq[i] for i in passive_v_idx])
            q_passive_err = q_passive - p_passive_eq
            if passive_dq_use_slack:
                dq_abs = ca.sqrt(dq_passive * dq_passive + passive_dq_soft_eps)
                dq_excess = dq_abs - passive_dq_soft_max
                dq_soft_slack = 0.5 * (dq_excess + ca.sqrt(dq_excess * dq_excess + passive_dq_soft_eps))
            else:
                dq_soft_slack = ca.SX.zeros(0, 1)
        else:
            q_passive_err = ca.SX.zeros(0, 1)
            dq_passive = ca.SX.zeros(0, 1)
            dq_soft_slack = ca.SX.zeros(0, 1)

        # ── Cost (NONLINEAR_LS) ───────────────────────────────────────────────
        # With NONLINEAR_LS + GAUSS_NEWTON, acados uses J^T W J which is always
        # positive semi-definite. The EXTERNAL cost with EXACT Hessian becomes
        # indefinite for large Cartesian residuals (due to second-order FK terms),
        # causing MINSTEP failures for large joint motions.
        #
        # Running residuals: [xyz_err(3) | yaw_err | s | (sdot-sdot_ref) | u_qdd(n_act) | v | [passive]]
        # A stage-varying yref on s enforces smooth progress over tau and avoids
        # reaching s~=1 too early and then waiting for joint velocities to settle.
        t_scale = ca.sqrt(T)
        y_parts: list = [t_scale * xyz_err, t_scale * yaw_err, t_scale * s, t_scale * (sdot - sdot_ref), t_scale * u_qdd, t_scale * v]
        w_diag: list = [xyz_w, xyz_w, xyz_w, yaw_w, s_w, sdot_w] + [u_w] * n_act + [v_w]
        if n_pas > 0:
            y_parts += [t_scale * q_passive_err, t_scale * dq_passive]
            w_diag += [passive_q_w] * n_pas + [passive_dq_w] * n_pas
            if passive_dq_use_slack:
                y_parts += [t_scale * dq_soft_slack]
                w_diag += [passive_dq_slack_w] * n_pas
        time_weight = float(req.config.get("time_weight", _FREE_TIME_WEIGHT_DEFAULT))
        if not optimize_time:
            time_weight = 0.0
        y_parts += [T]
        w_diag += [time_weight]
        y_expr = ca.vertcat(*y_parts)
        ny = len(w_diag)
        W_run = np.diag(np.asarray(w_diag, dtype=float))

        # Terminal residuals: [xyz_err(3) | yaw_err | dq(nv) | (1-s) | sdot | [passive]]
        y_e_parts: list = [xyz_err, yaw_err, dq, 1.0 - s, sdot]
        w_e_diag: list = [xyz_wN, xyz_wN, xyz_wN, yaw_wN] + [dqN_w] * nv + [sN_w, sdotN_w]
        if n_pas > 0:
            y_e_parts += [q_passive_err, dq_passive]
            w_e_diag += [passive_qN_w] * n_pas + [passive_dqN_w] * n_pas
            if passive_dq_use_slack:
                y_e_parts += [dq_soft_slack]
                w_e_diag += [passive_dq_slackN_w] * n_pas
        y_e_expr = ca.vertcat(*y_e_parts)
        ny_e = len(w_e_diag)
        W_term = np.diag(np.asarray(w_e_diag, dtype=float))

        # ── Acados model ──────────────────────────────────────────────────────
        ac_model = AcadosModel()
        ac_model.name = "crane_cartesian_pfc_ocp"
        ac_model.x = x
        ac_model.xdot = xdot
        ac_model.u = u
        ac_model.f_expl_expr = f_expl
        ac_model.f_impl_expr = f_impl
        ac_model.cost_y_expr = y_expr
        ac_model.cost_y_expr_e = y_e_expr
        ac_model.p = p

        ocp = AcadosOcp()
        ocp.model = ac_model
        ocp.dims.N = N
        ocp.solver_options.tf = 1.0
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = W_run
        ocp.cost.yref = np.zeros(ny, dtype=float)
        ocp.cost.W_e = W_term
        ocp.cost.yref_e = np.zeros(ny_e, dtype=float)
        ocp.parameter_values = np.concatenate(
            [np.zeros(n_ctrl_params + n_yaw_params + n_pas, dtype=float), np.asarray([payload_mass_default, *payload_com_default.tolist()], dtype=float)]
        )

        s0 = float(req.config.get("s0", 0.0))
        sdot0 = float(req.config.get("sdot0", 0.0))
        x0_partial = np.concatenate([q0, dq0, np.asarray([s0, sdot0], dtype=float)])
        ocp.constraints.idxbx_0 = np.arange(nqd + nv + 2, dtype=int)
        ocp.constraints.lbx_0 = x0_partial
        ocp.constraints.ubx_0 = x0_partial

        # ── Input bounds ──────────────────────────────────────────────────────
        lbu, ubu, _, limit_warnings, limits_yaml = prepare_control_bounds_from_limits(
            req_config=req.config,
            actuated_joints=cfg.actuated_joints,
            act_v_idx=act_v_idx,
            reduced_name_to_vidx=reduced_name_to_vidx,
            velocity_limits=np.asarray(model.velocityLimit, dtype=float),
            dt=dt_tau,
            joint_accel_limits_yaml=cfg.joint_accel_limits_yaml,
            validate_joint_limits_with_urdf=cfg.validate_joint_limits_with_urdf,
            qdd_u_min=cfg.qdd_u_min,
            qdd_u_max=cfg.qdd_u_max,
            v_min=cfg.v_min,
            v_max=cfg.v_max,
        )
        ocp.constraints.lbu = lbu
        ocp.constraints.ubu = ubu
        ocp.constraints.idxbu = np.arange(n_act + 1, dtype=int)

        # ── State bounds (joint positions + progress) ────────────────────────
        s_min = float(req.config.get("s_min", 0.0))
        s_max = float(req.config.get("s_max", 1.0))
        sdot_min = float(req.config.get("sdot_min", cfg.sdot_min))
        sdot_max = float(req.config.get("sdot_max", cfg.sdot_max))
        idx_s = nqd + nv
        idx_sdot = idx_s + 1
        idx_T = idx_sdot + 1
        idxbx: List[int] = []
        lbx: List[float] = []
        ubx: List[float] = []
        pos_limits_yaml = Path(req.config.get("joint_position_limits_yaml", cfg.joint_position_limits_yaml))

        if bool(req.config.get("enforce_joint_position_limits", True)):
            passive_vidx_set = set(passive_v_idx)
            q_lb_dec = np.full(nqd, -np.inf, dtype=float)
            q_ub_dec = np.full(nqd, np.inf, dtype=float)
            q_lb_pin = np.asarray(model.lowerPositionLimit, dtype=float)
            q_ub_pin = np.asarray(model.upperPositionLimit, dtype=float)
            for jm in joint_meta:
                if jm["nq"] != 1:
                    continue
                iv = int(jm["idx_v"])
                if iv in passive_vidx_set:
                    continue
                iq = int(jm["idx_q"])
                q_lb_dec[iv] = q_lb_pin[iq]
                q_ub_dec[iv] = q_ub_pin[iq]

            yaml_overrides, _ = load_planning_limits_yaml(pos_limits_yaml)
            merged_overrides: Dict[str, Tuple[Optional[float], Optional[float]]] = {
                str(jn): (lo, hi) for jn, (lo, hi) in dict(yaml_overrides).items()
            }
            for jn, bounds in dict(cfg.joint_position_overrides).items():
                merged_overrides[str(jn)] = (bounds[0], bounds[1])
            raw_req_overrides = req.config.get("joint_position_overrides", {})
            if isinstance(raw_req_overrides, Mapping):
                for jn, bounds in raw_req_overrides.items():
                    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                        continue
                    lo = None if bounds[0] is None else float(bounds[0])
                    hi = None if bounds[1] is None else float(bounds[1])
                    merged_overrides[str(jn)] = (lo, hi)

            for jn, (lo_ovr, hi_ovr) in merged_overrides.items():
                iv = reduced_name_to_vidx.get(jn)
                if iv is None:
                    continue
                if lo_ovr is not None:
                    q_lb_dec[iv] = max(q_lb_dec[iv], lo_ovr) if np.isfinite(q_lb_dec[iv]) else float(lo_ovr)
                if hi_ovr is not None:
                    q_ub_dec[iv] = min(q_ub_dec[iv], hi_ovr) if np.isfinite(q_ub_dec[iv]) else float(hi_ovr)

            finite_mask = np.isfinite(q_lb_dec) & np.isfinite(q_ub_dec)
            finite_idx = np.nonzero(finite_mask)[0]
            for iv in finite_idx.tolist():
                if q_lb_dec[iv] > q_ub_dec[iv]:
                    raise ValueError(f"Infeasible position bounds at idx_v={iv}: lb={q_lb_dec[iv]} > ub={q_ub_dec[iv]}")
            idxbx.extend(finite_idx.astype(int).tolist())
            lbx.extend(q_lb_dec[finite_idx].astype(float).tolist())
            ubx.extend(q_ub_dec[finite_idx].astype(float).tolist())

        T_stage_lb = T_min if optimize_time else T_guess
        T_stage_ub = T_max if optimize_time else T_guess
        idxbx.extend([idx_s, idx_sdot, idx_T])
        lbx.extend([s_min, sdot_min, T_stage_lb])
        ubx.extend([s_max, sdot_max, T_stage_ub])
        ocp.constraints.idxbx = np.asarray(idxbx, dtype=int)
        ocp.constraints.lbx = np.asarray(lbx, dtype=float)
        ocp.constraints.ubx = np.asarray(ubx, dtype=float)

        if bool(req.config.get("terminal_hard_zero_velocity", cfg.terminal_hard_zero_velocity)):
            idx_list = [*np.arange(nqd, nqd + nv, dtype=int).tolist(), idx_sdot]
            lb_list = [0.0] * nv + [0.0]
            ub_list = [0.0] * nv + [0.0]
            if bool(req.config.get("terminal_hard_end_progress", cfg.terminal_hard_end_progress)):
                idx_list.append(idx_s)
                lb_list.append(1.0)
                ub_list.append(1.0)
            if not optimize_time:
                idx_list.append(idx_T)
                lb_list.append(float(T_guess))
                ub_list.append(float(T_guess))
            ocp.constraints.idxbx_e = np.asarray(idx_list, dtype=int)
            ocp.constraints.lbx_e = np.asarray(lb_list, dtype=float)
            ocp.constraints.ubx_e = np.asarray(ub_list, dtype=float)
        elif not optimize_time:
            ocp.constraints.idxbx_e = np.asarray([idx_T], dtype=int)
            ocp.constraints.lbx_e = np.asarray([float(T_guess)], dtype=float)
            ocp.constraints.ubx_e = np.asarray([float(T_guess)], dtype=float)

        # ── Solver options ────────────────────────────────────────────────────
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.qp_solver_warm_start = int(req.config.get("qp_solver_warm_start", 1))
        ocp.solver_options.qp_solver_iter_max = int(
            req.config.get("qp_solver_iter_max", cfg.qp_solver_iter_max)
        )
        qp_tol = float(req.config.get("qp_tol", cfg.qp_tol))
        ocp.solver_options.qp_solver_tol_stat = qp_tol
        ocp.solver_options.qp_solver_tol_eq = qp_tol
        ocp.solver_options.qp_solver_tol_ineq = qp_tol
        ocp.solver_options.qp_solver_tol_comp = qp_tol
        hess = str(req.config.get("hessian_approx", cfg.hessian_approx)).upper()
        ocp.solver_options.hessian_approx = hess
        ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        ocp.solver_options.integrator_type = "IRK"
        ocp.solver_options.nlp_solver_type = str(
            req.config.get("nlp_solver_type", cfg.nlp_solver_type)
        )
        ocp.solver_options.nlp_solver_max_iter = int(
            req.config.get("nlp_solver_max_iter", cfg.nlp_solver_max_iter)
        )
        nlp_tol = float(req.config.get("nlp_tol", cfg.nlp_tol))
        ocp.solver_options.nlp_solver_tol_stat = nlp_tol
        ocp.solver_options.nlp_solver_tol_eq = nlp_tol
        ocp.solver_options.nlp_solver_tol_ineq = nlp_tol
        ocp.solver_options.nlp_solver_tol_comp = nlp_tol

        nlp_solver_type = str(req.config.get("nlp_solver_type", cfg.nlp_solver_type))
        passive_solve_damping = float(req.config.get("passive_solve_damping", cfg.passive_solve_damping))
        solver_key: Tuple[Any, ...] = (
            int(N),
            float(dt_tau),
            int(n_ctrl),
            int(spline_degree),
            dynamics_mode,
            tuple(act_v_idx),
            tuple(passive_v_idx),
            bool(req.config.get("enforce_joint_position_limits", True)),
            bool(req.config.get("terminal_hard_zero_velocity", cfg.terminal_hard_zero_velocity)),
            bool(req.config.get("terminal_hard_end_progress", cfg.terminal_hard_end_progress)),
            hess,
            nlp_solver_type,
            bool(passive_dq_use_slack),
        )

        if self._solver is None or self._solver_key != solver_key:
            code_export_dir = Path(req.config.get("code_export_dir", cfg.code_export_dir))
            code_export_dir.mkdir(parents=True, exist_ok=True)
            ocp.code_export_directory = str(code_export_dir)
            solver_json = str(
                code_export_dir / str(req.config.get("solver_json_name", cfg.solver_json_name))
            )
            solver_kwargs = {
                "json_file": solver_json,
                "build": bool(req.config.get("acados_build", True)),
                "generate": bool(req.config.get("acados_generate", True)),
                "verbose": bool(req.config.get("acados_verbose", True)),
            }
            if "check_reuse_possible" in inspect.signature(AcadosOcpSolver.__init__).parameters:
                solver_kwargs["check_reuse_possible"] = bool(
                    req.config.get("acados_check_reuse_possible", True)
                )
            self._solver = AcadosOcpSolver(ocp, **solver_kwargs)
            self._solver_key = solver_key
        solver = self._solver

        if bool(req.config.get("__compile_only", False)):
            return TrajectoryResult(
                success=True,
                message="acados solver compiled",
                time_s=np.zeros(0, dtype=float),
                state=np.zeros((0, nqd + nv + 3), dtype=float),
                control=np.zeros((0, n_act + 1), dtype=float),
                cost=None,
                diagnostics={"compiled_only": True},
            )

        # Runtime-tunable settings: keep codegen stable and avoid recompilation.
        for k in range(N):
            solver.cost_set(k, "W", W_run)
        solver.cost_set(N, "W", W_term)

        # Runtime option mutation is brittle across acados releases. Keep the
        # solver options defined on the OCP object above and avoid calling
        # options_set() here, since some builds abort in the C layer for fields
        # that the Python wrapper still advertises.

        # Keep runtime bounds mutable (e.g., T_min/T_max tuning) without codegen.
        for k in range(N):
            solver.set(k, "lbu", lbu)
            solver.set(k, "ubu", ubu)
        idxbx_arr = np.asarray(idxbx, dtype=int)
        pos_s = np.where(idxbx_arr == idx_s)[0]
        pos_sdot = np.where(idxbx_arr == idx_sdot)[0]
        terminal_hold_steps = int(req.config.get("terminal_hold_steps", cfg.terminal_hold_steps))
        ctrl_flat = np.asarray(ctrl_pts_xyz, dtype=float).reshape(-1)
        yaw_flat = np.asarray(ctrl_pts_yaw, dtype=float).reshape(-1)
        payload_params = np.asarray([payload_mass_default, *payload_com_default.tolist()], dtype=float)
        if n_pas > 0:
            q_pas_start = q0[passive_v_idx]
            q_pas_goal = q_goal[passive_v_idx]
            p_stage_values = []
            for k in range(N + 1):
                alpha = float(k) / float(N) if N > 0 else 1.0
                q_pas_k = (1.0 - alpha) * q_pas_start + alpha * q_pas_goal
                p_stage_values.append(
                    np.concatenate([ctrl_flat, yaw_flat, q_pas_k, payload_params])
                )
        else:
            shared = np.concatenate([ctrl_flat, yaw_flat, payload_params])
            p_stage_values = [shared] * (N + 1)

        p_val = (
            np.concatenate([ctrl_flat, yaw_flat, np.zeros(n_pas), payload_params])
            if n_pas > 0
            else np.concatenate([ctrl_flat, yaw_flat, payload_params])
        )

        candidate_times = np.asarray(
            [float(T_guess)] if not optimize_time else [float(T_guess)],
            dtype=float,
        )
        warm_start_progress = bool(
            req.config.get("warm_start_progress", cfg.warm_start_progress)
        )
        best_attempt: Optional[Dict[str, Any]] = None
        selected_candidate = float(candidate_times[0])

        for candidate_idx, candidate_T in enumerate(candidate_times.tolist()):
            lbx_stage = np.asarray(lbx, dtype=float).copy()
            ubx_stage = np.asarray(ubx, dtype=float).copy()
            if not optimize_time:
                t_pos = int(np.where(idxbx_arr == idx_T)[0][0])
                lbx_stage[t_pos] = float(candidate_T)
                ubx_stage[t_pos] = float(candidate_T)

            for k in range(1, N):
                solver.set(k, "lbx", lbx_stage)
                solver.set(k, "ubx", ubx_stage)
            if terminal_hold_steps > 0 and pos_s.size > 0 and pos_sdot.size > 0:
                hold_start = max(1, N - terminal_hold_steps)
                lbx_hold = lbx_stage.copy()
                ubx_hold = ubx_stage.copy()
                lbx_hold[int(pos_s[0])] = 1.0
                ubx_hold[int(pos_s[0])] = 1.0
                lbx_hold[int(pos_sdot[0])] = 0.0
                ubx_hold[int(pos_sdot[0])] = 0.0
                for k in range(hold_start, N):
                    solver.set(k, "lbx", lbx_hold)
                    solver.set(k, "ubx", ubx_hold)

            s_guess, sdot_guess = _progress_warm_start(
                N=N,
                s0=s0,
                sdot0=sdot0,
                T_guess=float(candidate_T),
                warm_start_progress=warm_start_progress,
            )
            x_ref = np.concatenate(
                [
                    q_goal,
                    np.zeros(nv, dtype=float),
                    np.asarray([1.0, 0.0, float(candidate_T)], dtype=float),
                ]
            )
            x0_full = np.concatenate(
                [q0, dq0, np.asarray([s0, sdot0, float(candidate_T)], dtype=float)]
            )
            x_traj = np.linspace(x0_full, x_ref, N + 1)
            x_traj[:, nqd + nv] = s_guess
            x_traj[:, nqd + nv + 1] = sdot_guess
            x_traj[:, nqd + nv + 2] = float(candidate_T)

            u_guess = np.zeros(n_act + 1, dtype=float)
            if candidate_T > 1e-9:
                sdot_target = float(np.clip(np.max(sdot_guess), sdot_min, sdot_max))
                u_guess[-1] = np.clip((sdot_target - sdot0) / candidate_T, lbu[-1], ubu[-1])

            for k in range(N + 1):
                solver.set(k, "x", x_traj[k])
                solver.set(k, "p", p_stage_values[k])
            solver.constraints_set(0, "lbx", x0_partial)
            solver.constraints_set(0, "ubx", x0_partial)
            for k in range(N):
                solver.set(k, "u", u_guess)
                yref_k = np.zeros(ny, dtype=float)
                yref_k[4] = s_guess[k + 1]
                solver.set(k, "yref", yref_k)
            solver.set(N, "yref", np.zeros(ny_e, dtype=float))

            status = int(solver.solve())
            try:
                cost_value = float(solver.get_cost())
            except Exception:
                cost_value = float("inf")
            try:
                residuals = np.asarray(solver.get_stats("residuals"), dtype=float).reshape(-1)
            except Exception:
                residuals = None
            try:
                sqp_iter = int(solver.get_stats("sqp_iter"))
            except Exception:
                sqp_iter = None
            try:
                qp_iter_stats = np.asarray(solver.get_stats("qp_iter"), dtype=float).reshape(-1)
            except Exception:
                qp_iter_stats = None

            attempt = {
                "status": status,
                "candidate_T": float(candidate_T),
                "cost": cost_value,
                "residuals": residuals,
                "sqp_iter": sqp_iter,
                "qp_iter_stats": qp_iter_stats,
                "candidate_index": candidate_idx,
            }
            if (
                best_attempt is None
                or int(status) == 0
                or (
                    int(best_attempt["status"]) != 0
                    and (
                        float(cost_value) < float(best_attempt["cost"])
                        or (
                            residuals is not None
                            and best_attempt["residuals"] is not None
                            and float(np.linalg.norm(residuals))
                            < float(np.linalg.norm(best_attempt["residuals"]))
                        )
                    )
                )
            ):
                best_attempt = attempt
                selected_candidate = float(candidate_T)
            if status == 0:
                break

        assert best_attempt is not None
        status = int(best_attempt["status"])
        success = status == 0
        sqp_iter = best_attempt["sqp_iter"]
        residuals = best_attempt["residuals"]
        qp_iter_stats = best_attempt["qp_iter_stats"]

        state = np.zeros((N + 1, nqd + nv + 3), dtype=float)
        control = np.zeros((N, n_act + 1), dtype=float)
        for k in range(N + 1):
            state[k, :] = np.asarray(solver.get(k, "x"), dtype=float).reshape(-1)
        for k in range(N):
            control[k, :] = np.asarray(solver.get(k, "u"), dtype=float).reshape(-1)

        tau = np.linspace(0.0, 1.0, N + 1, dtype=float)
        q_traj = state[:, :nqd]
        dq_traj = state[:, nqd : nqd + nv]
        s_traj = state[:, nqd + nv]
        T_traj = state[:, nqd + nv + 2]
        T_opt = float(T_traj[-1])
        t = T_opt * tau

        # Compute Cartesian FK trajectory for diagnostics.
        red_data_num = model.createData()
        xyz_traj = np.zeros((N + 1, 3), dtype=float)
        for k in range(N + 1):
            xyz_traj[k] = _fk_num(pin, model, red_data_num, q_traj[k], tool_fid)

        # Evaluate xyz_ref along s_traj.
        xyz_ref_fn = ca.Function("xyz_ref_fn", [s, p], [xyz_ref_s])
        xyz_ref_traj = np.zeros((N + 1, 3), dtype=float)
        for k in range(N + 1):
            sk = float(np.clip(s_traj[k], 0.0, 1.0))
            xyz_ref_traj[k] = np.asarray(xyz_ref_fn(sk, p_val), dtype=float).reshape(-1)

        msg = "acados solve success" if success else f"acados solve failed (status={status})"
        diagnostics: Dict[str, Any] = {
            "status": status,
            "solver_time_tot": float(solver.get_stats("time_tot")),
            "q_trajectory": q_traj,
            "dq_trajectory": dq_traj,
            "s_trajectory": s_traj,
            "sdot_trajectory": state[:, nqd + nv + 1],
            "T_trajectory": T_traj,
            "optimized_T": T_opt,
            "time_s_real": t,
            "tau": tau,
            "dt_tau": float(dt_tau),
            "horizon_N": int(N),
            "terminal_hold_steps": max(0, terminal_hold_steps),
            "xyz_trajectory": xyz_traj,
            "xyz_ref_trajectory": xyz_ref_traj,
            "ctrl_pts_xyz": ctrl_pts_xyz,
            "reduced_joint_names": keep_names,
            "actuated_joint_names": list(cfg.actuated_joints),
            "passive_joint_names": sorted(passive_set & keep_set),
            "actuated_v_indices": act_v_idx,
            "passive_v_indices": passive_v_idx,
            "passive_dq_soft_max": passive_dq_soft_max,
            "passive_dq_use_slack": bool(passive_dq_use_slack),
            "passive_dq_slack_weight": passive_dq_slack_w,
            "terminal_passive_dq_slack_weight": passive_dq_slackN_w,
            "control_names": list(cfg.actuated_joints) + ["path_progress_accel"],
            "dynamics_mode": dynamics_mode,
            "joint_accel_limits_yaml": str(limits_yaml),
            "joint_position_limits_yaml": str(pos_limits_yaml),
            "joint_accel_limit_warnings": limit_warnings,
            "payload_mass": payload_mass_default,
            "payload_com_tcp": payload_com_default.copy(),
            "gravity_world": g_world.copy(),
            "optimize_time": bool(optimize_time),
            "fixed_time_candidates_s": np.asarray(candidate_times, dtype=float),
            "selected_time_candidate_s": float(selected_candidate),
        }
        if sqp_iter is not None:
            diagnostics["sqp_iter"] = sqp_iter
        if residuals is not None:
            diagnostics["nlp_residuals"] = residuals
        if qp_iter_stats is not None:
            diagnostics["qp_iter_stats"] = qp_iter_stats
            diagnostics["qp_iter_sum"] = int(np.sum(qp_iter_stats))
            diagnostics["qp_iter_max"] = int(np.max(qp_iter_stats)) if qp_iter_stats.size > 0 else 0
            diagnostics["qp_iter_mean"] = float(np.mean(qp_iter_stats)) if qp_iter_stats.size > 0 else 0.0
        try:
            diagnostics["cost"] = float(solver.get_cost())
        except Exception:
            pass

        return TrajectoryResult(
            success=success,
            message=msg,
            time_s=t,
            state=state,
            control=control,
            cost=diagnostics.get("cost"),
            diagnostics=diagnostics,
        )



def main():
    q_start = np.array([1.265, 0.291, 1.069, 0.165, 0.165, 0.211, 1.571, 1.36])  # q_init

    from motion_planning.mechanics.analytic import create_crane_config

    p_start = np.array([-8.891, -5.842, 2.460])  # p_start
    yaw_start = 61.2 / 180.0 * np.pi
    p_end = np.array([-13.023, 2.725, 3.362])  # p_end
    yaw_end = 56.7 / 180.0 * np.pi

    # Full Cartesian task-space tracking (position + yaw orientation)
    n_ctrl = 4
    alphas = np.linspace(0.0, 1.0, n_ctrl)
    ctrl_pts_xyz = np.vstack([(1.0 - a) * p_start + a * p_end for a in alphas])
    ctrl_pts_yaw = np.asarray([(1.0 - a) * yaw_start + a * yaw_end for a in alphas], dtype=float)

    # q_goal is used as an initial guess/reference for state warm-start and
    # passive-equilibrium interpolation; Cartesian path is enforced via ctrl_pts_xyz and ctrl_pts_yaw.
    q_goal_init = np.array([-0.372, 0.47, 0.99, 0.253, 0.253, 0.11, 1.571, 1.461])

    cfg = CartesianPathFollowingConfig(
        urdf_path=Path(create_crane_config().urdf_path),
    )
    optimizer = CartesianPathFollowingOptimizer(cfg)
    req = TrajectoryRequest(
        scenario=None,
        path=None,
        config={
            "q0": q_start,
            "q_goal": q_goal_init,
            "dq0": np.zeros_like(q_start),
            "ctrl_pts_xyz": ctrl_pts_xyz,
            "ctrl_pts_yaw": ctrl_pts_yaw,
            "T_min": 0.5,
            "T_max": 30.0,
            "nlp_solver_max_iter": 1000,
            "qp_solver_iter_max": 300,
        },
    )
    result = optimizer.optimize(req)

    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    if result.cost is not None:
        print(f"Cost: {result.cost:.6f}")
    print(f"Optimized T: {float(result.diagnostics.get('optimized_T', np.nan)):.4f}")
    print(f"Final path progress s(T): {float(result.diagnostics.get('s_trajectory', [np.nan])[-1]):.4f}")
    print(f"Solver time (s): {result.diagnostics.get('solver_time_tot', np.nan):.4f}")
    if "sqp_iter" in result.diagnostics:
        print(f"SQP iterations: {int(result.diagnostics['sqp_iter'])}")
    if "nlp_residuals" in result.diagnostics:
        res = np.asarray(result.diagnostics["nlp_residuals"], dtype=float).reshape(-1)
        print(f"NLP residuals [stat, eq, ineq, comp]: {res}")

    import matplotlib.pyplot as plt

    t = result.time_s
    s = np.asarray(result.diagnostics["s_trajectory"], dtype=float)
    sdot = np.asarray(result.diagnostics["sdot_trajectory"], dtype=float)
    xyz = np.asarray(result.diagnostics["xyz_trajectory"], dtype=float)
    xyz_ref = np.asarray(result.diagnostics["xyz_ref_trajectory"], dtype=float)
    dq = np.asarray(result.diagnostics["dq_trajectory"], dtype=float)
    u = np.asarray(result.control, dtype=float)
    u_t = t[:-1] if u.shape[0] > 0 else t
    control_names = list(result.diagnostics.get("control_names", []))

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=False)

    axes[0].plot(t, s, label="s")
    axes[0].plot(t, sdot, label="sdot")
    axes[0].set_title("Path Progress")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("value")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    labels = ["x", "y", "z"]
    for i, lab in enumerate(labels):
        axes[1].plot(t, xyz[:, i], label=f"{lab} actual")
        axes[1].plot(t, xyz_ref[:, i], "--", label=f"{lab} ref")
    axes[1].set_title("Cartesian Tracking")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("position [m]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", ncol=2, fontsize=8)

    for j in range(dq.shape[1]):
        axes[2].plot(t, dq[:, j], lw=1.1, label=f"dq{j}")
    axes[2].set_title("Joint Velocities")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("joint velocity")
    axes[2].grid(True, alpha=0.3)
    if dq.shape[1] <= 8:
        axes[2].legend(loc="best", ncol=2, fontsize=8)

    for i in range(u.shape[1]):
        label = control_names[i] if i < len(control_names) else f"u{i}"
        axes[3].step(u_t, u[:, i], where="post", label=label)
    axes[3].set_title("Control Inputs")
    axes[3].set_xlabel("time [s]")
    axes[3].set_ylabel("command")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
