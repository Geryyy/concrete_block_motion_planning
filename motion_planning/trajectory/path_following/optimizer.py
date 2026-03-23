from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from motion_planning.core.types import TrajectoryRequest, TrajectoryResult
from motion_planning.trajectory.base import TrajectoryOptimizer
from motion_planning.trajectory.dynamics import build_underactuated_qdd_symbolic
from motion_planning.trajectory.limits import prepare_control_bounds_from_limits
from motion_planning.trajectory.planning_limits import load_planning_limits_yaml
from motion_planning.trajectory.path_following.config import CranePathFollowingAcadosConfig
from motion_planning.trajectory.path_following.spline import bspline_eval_symbolic

# Keep solver behavior in-code to keep config minimal.
_VALIDATE_LIMITS_WITH_URDF = True
_TERMINAL_HARD_ZERO_VELOCITY = True
_TERMINAL_HARD_END_PROGRESS = True
_TERMINAL_HARD_ZERO_CONTROL = True
_QP_SOLVER_ITER_MAX = 300
_QP_TOL = 1e-5
_HESSIAN_APPROX = "GAUSS_NEWTON"
_NLP_SOLVER_TYPE = "SQP"
_NLP_SOLVER_MAX_ITER = 300
_NLP_TOL = 1e-5
_REGULARIZE_METHOD = "PROJECT"
_LEVENBERG_MARQUARDT = 1e-3
_TIME_WEIGHT_DEFAULT = 1e-3
_T_MIN_DEFAULT = 0.5
_T_MAX_DEFAULT = 60.0


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


def _default_spline_ctrl_points(q0: np.ndarray, q_goal: np.ndarray, n_ctrl: int) -> np.ndarray:
    if n_ctrl < 2:
        raise ValueError(f"spline_ctrl_points must be >= 2, got {n_ctrl}.")
    alphas = np.linspace(0.0, 1.0, n_ctrl, dtype=float)
    return np.vstack([(1.0 - a) * q0 + a * q_goal for a in alphas])


class CranePathFollowingAcadosOptimizer(TrajectoryOptimizer):
    """Configuration-space path-following OCP with spline progress dynamics.

    State: x = [q, dq, s, sdot]
    Input: u = [qdd_actuated, v] where v = s_ddot.
    """

    def __init__(self, config: CranePathFollowingAcadosConfig):
        self.config = config
        self._solver = None
        self._solver_key: Optional[Tuple[Any, ...]] = None
        if bool(self.config.precompile_on_init):
            self._precompile_solver()

    def _precompile_solver(self) -> None:
        dummy_req = TrajectoryRequest(
            scenario=None,
            path=None,
            config={"__compile_only": True},
        )
        self.optimize(dummy_req)

    @staticmethod
    def _joint_meta(model) -> List[Dict[str, int]]:
        meta: List[Dict[str, int]] = []
        for jid in range(1, model.njoints):
            jmodel = model.joints[jid]
            if int(jmodel.nv) != 1:
                raise ValueError(f"Joint '{model.names[jid]}' has nv={jmodel.nv}; nv=1 joints only are supported.")
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

        full_model = pin.buildModelFromUrdf(str(urdf_path))
        full_meta = self._joint_meta(full_model)
        full_name_to_jid = {m["name"]: m["jid"] for m in full_meta}
        passive_set = set(cfg.passive_joints)
        actuated_set = set(cfg.actuated_joints)
        if passive_set & actuated_set:
            raise ValueError(f"Joints cannot be both passive and actuated: {sorted(passive_set & actuated_set)}")

        lock_joint_ids: List[int] = []
        for name in cfg.lock_joint_names:
            if name in full_name_to_jid:
                lock_joint_ids.append(int(full_name_to_jid[name]))
            elif cfg.print_model_prep:
                print(f"[path-following] warning: lock joint '{name}' not found.")

        reduced_model = pin.buildReducedModel(full_model, lock_joint_ids, pin.neutral(full_model))
        model = reduced_model
        reduced_meta = self._joint_meta(model)
        keep_names = [m["name"] for m in reduced_meta]
        keep_set = set(keep_names)
        if not passive_set.issubset(keep_set):
            missing = sorted(passive_set - keep_set)
            raise ValueError(f"Passive joints removed by reduction: {missing}")
        if cfg.print_model_prep:
            print("[path-following] reduced joints:", keep_names)
        tool_fid = model.getFrameId(cfg.tool_frame_name)
        if tool_fid >= model.nframes:
            raise ValueError(
                f"Frame '{cfg.tool_frame_name}' not found in reduced model. "
                f"Available frames: {[f.name for f in model.frames]}"
            )

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
        # --- FREE TIME MODIFICATION ---
        # OCP is posed on normalized time tau in [0, 1].
        dt_tau = 1.0 / max(N, 1)

        n_ctrl = int(req.config.get("spline_ctrl_points", cfg.spline_ctrl_points))
        spline_ctrl = req.config.get("spline_ctrl_points_q", None)
        if spline_ctrl is None:
            spline_ctrl_points = _default_spline_ctrl_points(q0=q0, q_goal=q_goal, n_ctrl=n_ctrl)
        else:
            spline_ctrl_points = np.asarray(spline_ctrl, dtype=float)
            if spline_ctrl_points.shape != (n_ctrl, nqd):
                raise ValueError(
                    f"spline_ctrl_points_q must have shape ({n_ctrl}, {nqd}), got {spline_ctrl_points.shape}."
                )

        name_to_joint_id = {m["name"]: m["jid"] for m in joint_meta}
        reduced_name_to_vidx = {m["name"]: int(m["idx_v"]) for m in joint_meta}
        act_joint_ids = [int(name_to_joint_id[name]) for name in cfg.actuated_joints]
        act_v_idx = [int(model.joints[jid].idx_v) for jid in act_joint_ids]
        passive_v_idx = [reduced_name_to_vidx[name] for name in sorted(passive_set & keep_set)]
        n_act = len(act_v_idx)
        lbu, ubu, _, limit_warnings, limits_yaml = prepare_control_bounds_from_limits(
            req_config=req.config,
            actuated_joints=cfg.actuated_joints,
            act_v_idx=act_v_idx,
            reduced_name_to_vidx=reduced_name_to_vidx,
            velocity_limits=np.asarray(model.velocityLimit, dtype=float),
            dt=dt_tau,
            joint_accel_limits_yaml=cfg.joint_accel_limits_yaml,
            validate_joint_limits_with_urdf=_VALIDATE_LIMITS_WITH_URDF,
            qdd_u_min=cfg.qdd_u_min,
            qdd_u_max=cfg.qdd_u_max,
            v_min=cfg.v_min,
            v_max=cfg.v_max,
        )

        T_min = float(req.config.get("T_min", _T_MIN_DEFAULT))
        T_max = float(req.config.get("T_max", _T_MAX_DEFAULT))
        if not (T_min > 0.0 and T_min <= T_max):
            raise ValueError(f"Invalid free-time bounds: require 0 < T_min <= T_max, got {T_min}, {T_max}.")
        T_guess = 0.5 * (T_min + T_max)

        # --- FREE TIME MODIFICATION ---
        # Extend state with constant final-time variable T.
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
        q_pin = q_dec_to_q_pin_sym(q)

        payload_mass_default = float(req.config.get("payload_mass", cfg.payload_mass))
        payload_com_default = np.asarray(req.config.get("payload_com_tcp", cfg.payload_com_tcp), dtype=float).reshape(-1)
        if payload_com_default.shape[0] != 3:
            raise ValueError(f"payload_com_tcp must have length 3, got {payload_com_default.shape[0]}.")
        g_world = np.asarray(req.config.get("gravity_world", [0.0, 0.0, -9.81]), dtype=float).reshape(-1)
        if g_world.shape[0] != 3:
            raise ValueError(f"gravity_world must have length 3, got {g_world.shape[0]}.")
        p_dyn = ca.SX.sym("p_dyn", 4)  # [payload_mass, com_x, com_y, com_z]
        payload_mass = p_dyn[0]
        payload_com_tcp = p_dyn[1:4]
        # Parameterize payload as additional symbolic inertia attached to the tool body.
        # payload_com_tcp is specified in TCP frame and mapped to tool-body (joint) frame.
        tool_body_id = int(model.frames[tool_fid].parentJoint)
        tcp_in_body = model.frames[tool_fid].placement
        p_b_tcp = ca.DM(np.asarray(tcp_in_body.translation, dtype=float).reshape(3))
        R_b_tcp = ca.DM(np.asarray(tcp_in_body.rotation, dtype=float))
        payload_com_body = p_b_tcp + R_b_tcp @ payload_com_tcp
        cmodel_dyn = cmodel.copy()
        original_inertia = cmodel_dyn.inertias[tool_body_id]
        # Keep existing rigid-body inertia tensor and update mass+CoM to include payload.
        total_mass = original_inertia.mass + payload_mass
        total_com = (original_inertia.mass * original_inertia.lever + payload_mass * payload_com_body) / (total_mass + 1e-9)
        cmodel_dyn.inertias[tool_body_id] = cpin.Inertia(total_mass, total_com, original_inertia.inertia)
        cdata_dyn = cmodel_dyn.createData()

        q_ref_s = bspline_eval_symbolic(s=s, control_points=spline_ctrl_points, degree=int(cfg.spline_degree))
        dq_ref_ds = ca.jacobian(q_ref_s, s)
        dq_path_ref = dq_ref_ds * sdot
        dynamics_mode = str(req.config.get("dynamics_mode", cfg.dynamics_mode)).lower()
        # Backward compatibility for previous boolean flag.
        if "split_passive_dynamics" in req.config:
            dynamics_mode = "split" if bool(req.config["split_passive_dynamics"]) else "projected"
        if dynamics_mode not in {"split", "projected"}:
            raise ValueError(f"Unsupported dynamics_mode '{dynamics_mode}'. Expected 'split' or 'projected'.")

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

        # --- FREE TIME MODIFICATION ---
        # Time scaling: dx/dtau = T * f(x, u), with T_dot = 0.
        f_expl_base = ca.vertcat(dq, qdd, sdot, v)
        f_expl = ca.vertcat(T * f_expl_base, 0.0)
        f_impl = xdot - f_expl

        q_err = q - q_ref_s
        dq_err = dq - dq_path_ref
        sdot_ref = float(req.config.get("sdot_ref", cfg.sdot_ref))
        q_w = float(req.config.get("q_path_weight", cfg.q_path_weight))
        dq_w_legacy = req.config.get("dq_path_weight", None)
        if dq_w_legacy is not None:
            dq_w_art = float(dq_w_legacy)
            dq_w_pas = float(dq_w_legacy)
        else:
            dq_w_art = float(req.config.get("dq_path_weight_articulated", cfg.dq_path_weight_articulated))
            dq_w_pas = float(req.config.get("dq_path_weight_passive", cfg.dq_path_weight_passive))
        s_w = float(req.config.get("s_weight", cfg.s_weight))
        sdot_w = float(req.config.get("sdot_weight", cfg.sdot_weight))
        u_w = float(req.config.get("qdd_u_weight", cfg.qdd_u_weight))
        v_w = float(req.config.get("v_weight", cfg.v_weight))
        qN_w = float(req.config.get("terminal_q_path_weight", cfg.terminal_q_path_weight))
        sN_w = float(req.config.get("terminal_s_weight", cfg.terminal_s_weight))
        sdotN_w = float(req.config.get("terminal_sdot_weight", cfg.terminal_sdot_weight))
        dqN_w_legacy = req.config.get("terminal_dq_weight", None)
        if dqN_w_legacy is not None:
            dqN_w_art = float(dqN_w_legacy)
            dqN_w_pas = float(dqN_w_legacy)
        else:
            dqN_w_art = float(req.config.get("terminal_dq_weight_articulated", cfg.terminal_dq_weight_articulated))
            dqN_w_pas = float(req.config.get("terminal_dq_weight_passive", cfg.terminal_dq_weight_passive))
        passive_q_w = float(req.config.get("passive_q_sway_weight", cfg.passive_q_sway_weight))
        passive_dq_w = float(req.config.get("passive_dq_sway_weight", cfg.passive_dq_sway_weight))
        passive_qN_w = float(req.config.get("terminal_passive_q_sway_weight", cfg.terminal_passive_q_sway_weight))
        passive_dqN_w = float(req.config.get("terminal_passive_dq_sway_weight", cfg.terminal_passive_dq_sway_weight))
        passive_dq_soft_max = float(req.config.get("passive_dq_soft_max", cfg.passive_dq_soft_max))
        passive_dq_slack_w = float(req.config.get("passive_dq_slack_weight", cfg.passive_dq_slack_weight))
        passive_dq_slackN_w = float(
            req.config.get("terminal_passive_dq_slack_weight", cfg.terminal_passive_dq_slack_weight)
        )
        passive_dq_soft_eps = float(req.config.get("passive_dq_soft_abs_eps", cfg.passive_dq_soft_abs_eps))
        if passive_dq_soft_max <= 0.0:
            raise ValueError(f"passive_dq_soft_max must be > 0, got {passive_dq_soft_max}.")
        if passive_dq_soft_eps <= 0.0:
            raise ValueError(f"passive_dq_soft_abs_eps must be > 0, got {passive_dq_soft_eps}.")

        q_err_passive = ca.vertcat(*[q_err[i] for i in passive_v_idx]) if len(passive_v_idx) > 0 else ca.SX.zeros(0, 1)
        dq_passive = ca.vertcat(*[dq[i] for i in passive_v_idx]) if len(passive_v_idx) > 0 else ca.SX.zeros(0, 1)
        dq_err_passive = ca.vertcat(*[dq_err[i] for i in passive_v_idx]) if len(passive_v_idx) > 0 else ca.SX.zeros(0, 1)
        articulated_v_idx = [i for i in range(nv) if i not in set(passive_v_idx)]
        dq_articulated = ca.vertcat(*[dq[i] for i in articulated_v_idx]) if len(articulated_v_idx) > 0 else ca.SX.zeros(0, 1)
        dq_err_articulated = ca.vertcat(*[dq_err[i] for i in articulated_v_idx]) if len(articulated_v_idx) > 0 else ca.SX.zeros(0, 1)

        l_cost_base = (
            q_w * ca.dot(q_err, q_err)
            + dq_w_art * ca.dot(dq_err_articulated, dq_err_articulated)
            + dq_w_pas * ca.dot(dq_err_passive, dq_err_passive)
            + s_w * ca.power(1.0 - s, 2)
            + sdot_w * ca.power(sdot - sdot_ref, 2)
            + u_w * ca.dot(u_qdd, u_qdd)
            + v_w * ca.power(v, 2)
        )
        # --- FREE TIME MODIFICATION ---
        # Correct time-scaled objective: integral over tau of T * L(x,u).
        # Add a small extra linear time penalty to bias shorter motions.
        time_weight = float(req.config.get("time_weight", _TIME_WEIGHT_DEFAULT))
        l_cost = T * l_cost_base + time_weight * T
        if len(passive_v_idx) > 0:
            dq_abs = ca.sqrt(dq_passive * dq_passive + passive_dq_soft_eps)
            dq_excess = dq_abs - passive_dq_soft_max
            dq_soft_slack = 0.5 * (dq_excess + ca.sqrt(dq_excess * dq_excess + passive_dq_soft_eps))
            l_cost = l_cost + T * (
                passive_q_w * ca.dot(q_err_passive, q_err_passive)
                + passive_dq_w * ca.dot(dq_passive, dq_passive)
                + passive_dq_slack_w * ca.dot(dq_soft_slack, dq_soft_slack)
            )
        m_cost = (
            qN_w * ca.dot(q_err, q_err)
            + dqN_w_art * ca.dot(dq_articulated, dq_articulated)
            + dqN_w_pas * ca.dot(dq_passive, dq_passive)
            + sN_w * ca.power(1.0 - s, 2)
            + sdotN_w * ca.power(sdot - sdot_ref, 2)
        )
        if len(passive_v_idx) > 0:
            m_cost = m_cost + (
                passive_qN_w * ca.dot(q_err_passive, q_err_passive)
                + passive_dqN_w * ca.dot(dq_passive, dq_passive)
                + passive_dq_slackN_w * ca.dot(dq_soft_slack, dq_soft_slack)
            )

        ac_model = AcadosModel()
        ac_model.name = "crane_acados_path_following_ocp"
        ac_model.x = x
        ac_model.xdot = xdot
        ac_model.u = u
        ac_model.f_expl_expr = f_expl
        ac_model.f_impl_expr = f_impl
        ac_model.p = p_dyn
        ac_model.cost_expr_ext_cost = l_cost
        ac_model.cost_expr_ext_cost_e = m_cost

        ocp = AcadosOcp()
        ocp.model = ac_model
        ocp.dims.N = N
        # --- FREE TIME MODIFICATION ---
        # Normalized horizon in tau-space.
        ocp.solver_options.tf = 1.0
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.parameter_values = np.asarray(
            [payload_mass_default, *payload_com_default.tolist()], dtype=float
        )

        s0 = float(req.config.get("s0", 0.0))
        sdot0 = float(req.config.get("sdot0", 0.0))
        # --- FREE TIME MODIFICATION ---
        # Do not fix T at stage 0; constrain only [q, dq, s, sdot] initial state.
        x0_partial = np.concatenate([q0, dq0, np.asarray([s0, sdot0], dtype=float)])
        ocp.constraints.idxbx_0 = np.arange(nqd + nv + 2, dtype=int)
        ocp.constraints.lbx_0 = x0_partial
        ocp.constraints.ubx_0 = x0_partial

        ocp.constraints.lbu = lbu
        ocp.constraints.ubu = ubu
        ocp.constraints.idxbu = np.arange(n_act + 1, dtype=int)

        # State bounds for joint positions + path progress variables.
        s_min = float(req.config.get("s_min", 0.0))
        s_max = float(req.config.get("s_max", 1.0))
        sdot_min = float(req.config.get("sdot_min", cfg.sdot_min))
        sdot_max = float(req.config.get("sdot_max", cfg.sdot_max))
        idx_s = nqd + nv
        idx_sdot = idx_s + 1
        # --- FREE TIME MODIFICATION ---
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

        # --- FREE TIME MODIFICATION ---
        # Add free-time bounds to path constraints.
        idxbx.extend([idx_s, idx_sdot, idx_T])
        lbx.extend([s_min, sdot_min, T_min])
        ubx.extend([s_max, sdot_max, T_max])
        ocp.constraints.idxbx = np.asarray(idxbx, dtype=int)
        ocp.constraints.lbx = np.asarray(lbx, dtype=float)
        ocp.constraints.ubx = np.asarray(ubx, dtype=float)
        if _TERMINAL_HARD_ZERO_VELOCITY:
            idx_list = [*np.arange(nqd, nqd + nv, dtype=int).tolist(), idx_sdot]
            lb_list = [0.0] * nv + [0.0]
            ub_list = [0.0] * nv + [0.0]
            if _TERMINAL_HARD_END_PROGRESS:
                idx_list.append(idx_s)
                lb_list.append(1.0)
                ub_list.append(1.0)
            idxbx_e = np.asarray(idx_list, dtype=int)
            lbx_e = np.asarray(lb_list, dtype=float)
            ubx_e = np.asarray(ub_list, dtype=float)
            ocp.constraints.idxbx_e = idxbx_e
            ocp.constraints.lbx_e = lbx_e
            ocp.constraints.ubx_e = ubx_e

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.qp_solver_warm_start = int(req.config.get("qp_solver_warm_start", 0))
        qp_iter_max = int(req.config.get("qp_solver_iter_max", _QP_SOLVER_ITER_MAX))
        ocp.solver_options.qp_solver_iter_max = qp_iter_max
        qp_tol = float(req.config.get("qp_tol", _QP_TOL))
        ocp.solver_options.qp_solver_tol_stat = qp_tol
        ocp.solver_options.qp_solver_tol_eq = qp_tol
        ocp.solver_options.qp_solver_tol_ineq = qp_tol
        ocp.solver_options.qp_solver_tol_comp = qp_tol
        hess = _HESSIAN_APPROX
        ocp.solver_options.hessian_approx = hess
        ocp.solver_options.regularize_method = _REGULARIZE_METHOD
        ocp.solver_options.levenberg_marquardt = _LEVENBERG_MARQUARDT
        if hess == "EXACT":
            ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        ocp.solver_options.integrator_type = "IRK"
        ocp.solver_options.nlp_solver_type = _NLP_SOLVER_TYPE
        nlp_max_iter = int(req.config.get("nlp_solver_max_iter", _NLP_SOLVER_MAX_ITER))
        ocp.solver_options.nlp_solver_max_iter = nlp_max_iter
        nlp_tol = float(req.config.get("nlp_tol", _NLP_TOL))
        ocp.solver_options.nlp_solver_tol_stat = nlp_tol
        ocp.solver_options.nlp_solver_tol_eq = nlp_tol
        ocp.solver_options.nlp_solver_tol_ineq = nlp_tol
        ocp.solver_options.nlp_solver_tol_comp = nlp_tol

        solver_key: Tuple[Any, ...] = (
            int(N),
            float(dt_tau),
            int(n_ctrl),
            int(cfg.spline_degree),
            dynamics_mode,
            tuple(act_v_idx),
            tuple(passive_v_idx),
            tuple(np.round(np.asarray(spline_ctrl_points, dtype=float).reshape(-1), 8).tolist()),
            tuple(np.asarray(idxbx, dtype=int).tolist()),
            bool(req.config.get("enforce_joint_position_limits", True)),
            float(sdot_ref),
            float(q_w),
            float(dq_w_art),
            float(dq_w_pas),
            float(s_w),
            float(sdot_w),
            float(u_w),
            float(v_w),
            float(qN_w),
            float(sN_w),
            float(sdotN_w),
            float(dqN_w_art),
            float(dqN_w_pas),
            float(passive_q_w),
            float(passive_dq_w),
            float(passive_qN_w),
            float(passive_dqN_w),
            float(time_weight),
        )

        if self._solver is None or self._solver_key != solver_key:
            code_export_dir = Path(req.config.get("code_export_dir", cfg.code_export_dir))
            code_export_dir.mkdir(parents=True, exist_ok=True)
            ocp.code_export_directory = str(code_export_dir)
            solver_json = str(code_export_dir / str(req.config.get("solver_json_name", cfg.solver_json_name)))
            self._solver = AcadosOcpSolver(ocp, json_file=solver_json)
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

        # Runtime option mutation is brittle across acados releases. Keep the
        # solver options defined on the OCP object above and avoid calling
        # options_set() here, since some builds abort in the C layer for fields
        # that the Python wrapper still advertises.
        for k in range(N):
            solver.set(k, "lbu", lbu)
            solver.set(k, "ubu", ubu)
            solver.set(k, "p", np.asarray([payload_mass_default, *payload_com_default.tolist()], dtype=float))
        solver.set(N, "p", np.asarray([payload_mass_default, *payload_com_default.tolist()], dtype=float))
        idxbx_arr = np.asarray(idxbx, dtype=int)
        pos_s = np.where(idxbx_arr == idx_s)[0]
        pos_sdot = np.where(idxbx_arr == idx_sdot)[0]
        for k in range(1, N):
            solver.set(k, "lbx", np.asarray(lbx, dtype=float))
            solver.set(k, "ubx", np.asarray(ubx, dtype=float))
        terminal_hold_steps = int(req.config.get("terminal_hold_steps", cfg.terminal_hold_steps))
        if terminal_hold_steps > 0 and pos_s.size > 0 and pos_sdot.size > 0:
            hold_start = max(1, N - terminal_hold_steps)
            lbx_hold = np.asarray(lbx, dtype=float).copy()
            ubx_hold = np.asarray(ubx, dtype=float).copy()
            lbx_hold[int(pos_s[0])] = 1.0
            ubx_hold[int(pos_s[0])] = 1.0
            lbx_hold[int(pos_sdot[0])] = 0.0
            ubx_hold[int(pos_sdot[0])] = 0.0
            for k in range(hold_start, N):
                solver.set(k, "lbx", lbx_hold)
                solver.set(k, "ubx", ubx_hold)

        # --- FREE TIME MODIFICATION ---
        x_ref = np.concatenate([q_goal, np.zeros(nv, dtype=float), np.asarray([1.0, sdot_ref, T_guess], dtype=float)])
        x0_full = np.concatenate([q0, dq0, np.asarray([s0, sdot0, T_guess], dtype=float)])
        x_traj = np.linspace(x0_full, x_ref, N + 1)
        u_guess = np.zeros(n_act + 1, dtype=float)
        if T_guess > 1e-9:
            u_guess[-1] = np.clip((sdot_ref - sdot0) / T_guess, lbu[-1], ubu[-1])

        for k in range(N + 1):
            solver.set(k, "x", x_traj[k])
        solver.constraints_set(0, "lbx", x0_partial)
        solver.constraints_set(0, "ubx", x0_partial)
        for k in range(N):
            solver.set(k, "u", u_guess)
        if _TERMINAL_HARD_ZERO_CONTROL:
            u_zero = np.zeros(n_act + 1, dtype=float)
            solver.set(N - 1, "lbu", u_zero)
            solver.set(N - 1, "ubu", u_zero)

        status = int(solver.solve())
        success = status == 0
        sqp_iter: Optional[int] = None
        residuals: Optional[np.ndarray] = None
        try:
            sqp_iter = int(solver.get_stats("sqp_iter"))
        except Exception:
            sqp_iter = None
        try:
            residuals = np.asarray(solver.get_stats("residuals"), dtype=float).reshape(-1)
        except Exception:
            residuals = None

        state = np.zeros((N + 1, nqd + nv + 3), dtype=float)
        control = np.zeros((N, n_act + 1), dtype=float)
        for k in range(N + 1):
            state[k, :] = np.asarray(solver.get(k, "x"), dtype=float).reshape(-1)
        for k in range(N):
            control[k, :] = np.asarray(solver.get(k, "u"), dtype=float).reshape(-1)

        # --- FREE TIME MODIFICATION ---
        tau = np.linspace(0.0, 1.0, N + 1, dtype=float)
        q_traj = state[:, :nqd]
        dq_traj = state[:, nqd : nqd + nv]
        s_traj = state[:, nqd + nv]
        sdot_traj = state[:, nqd + nv + 1]
        T_traj = state[:, nqd + nv + 2]
        T_opt = float(T_traj[-1])
        t = T_opt * tau
        q_ref_traj = np.zeros_like(q_traj)
        dq_ref_traj = np.zeros_like(dq_traj)
        q_ref_fn = ca.Function("q_ref_fn", [s], [q_ref_s])
        for k in range(N + 1):
            sk = float(np.clip(s_traj[k], 0.0, 1.0))
            q_ref_traj[k] = np.asarray(q_ref_fn(sk), dtype=float).reshape(-1)
        dq_ref_traj = np.gradient(q_ref_traj, t, axis=0)

        msg = "acados solve success" if success else f"acados solve failed (status={status})"
        diagnostics = {
            "status": status,
            "solver_time_tot": float(solver.get_stats("time_tot")),
            "q_trajectory": q_traj,
            "dq_trajectory": dq_traj,
            "s_trajectory": s_traj,
            "sdot_trajectory": sdot_traj,
            "q_path_reference_trajectory": q_ref_traj,
            "dq_path_reference_trajectory": dq_ref_traj,
            "path_control_points_q": spline_ctrl_points,
            "path_spline_degree": int(cfg.spline_degree),
            "actuated_joint_names": list(cfg.actuated_joints),
            "reduced_joint_names": keep_names,
            "passive_joint_names": sorted(passive_set & keep_set),
            "actuated_v_indices": act_v_idx,
            "passive_v_indices": passive_v_idx,
            "passive_dq_soft_max": passive_dq_soft_max,
            "passive_dq_slack_weight": passive_dq_slack_w,
            "terminal_passive_dq_slack_weight": passive_dq_slackN_w,
            "control_names": list(cfg.actuated_joints) + ["path_progress_accel"],
            "control_interface": "qdd_plus_path_progress",
            "dynamics_mode": dynamics_mode,
            "horizon_steps_effective": int(N),
            # --- FREE TIME MODIFICATION ---
            "optimized_T": T_opt,
            "T_trajectory": T_traj,
            "tau": tau,
            "time_s_real": t,
            "dt_tau": float(dt_tau),
            "horizon_N": int(N),
            "terminal_hold_steps": max(0, terminal_hold_steps),
            "joint_accel_limits_yaml": str(limits_yaml),
            "joint_position_limits_yaml": str(pos_limits_yaml),
            "joint_accel_limit_warnings": limit_warnings,
            "terminal_hard_zero_control": _TERMINAL_HARD_ZERO_CONTROL,
            "payload_mass": payload_mass_default,
            "payload_com_tcp": payload_com_default.copy(),
            "gravity_world": g_world.copy(),
        }
        try:
            diagnostics["cost"] = float(solver.get_cost())
        except Exception:
            pass
        if sqp_iter is not None:
            diagnostics["sqp_iter"] = sqp_iter
        if residuals is not None:
            diagnostics["nlp_residuals"] = residuals

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
    q_start = np.array([1.265, 0.291, 1.069, 0.165, 0.165, 0.211, 1.571, 1.36])
    q_end = np.array([-0.372, 0.47, 0.99, 0.253, 0.253, 0.11, 1.571, 1.461])

    from motion_planning.mechanics.analytic import create_crane_config

    cfg = CranePathFollowingAcadosConfig(
        urdf_path=Path(create_crane_config().urdf_path),
    )
    optimizer = CranePathFollowingAcadosOptimizer(cfg)
    req = TrajectoryRequest(
        scenario=None,
        path=None,
        config={
            "q0": q_start,
            "q_goal": q_end,
            "dq0": np.zeros(8),
        },
    )
    result = optimizer.optimize(req)

    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Solver time (s): {result.diagnostics.get('solver_time_tot', np.nan):.4f}")
    print(f"Final path progress s(T): {result.diagnostics.get('s_trajectory', [np.nan])[-1]:.4f}")
    if result.cost is not None:
        print(f"Cost: {result.cost:.6f}")

    import matplotlib.pyplot as plt

    t = result.time_s
    q = np.asarray(result.diagnostics["q_trajectory"], dtype=float)
    dq = np.asarray(result.diagnostics["dq_trajectory"], dtype=float)
    q_ref = np.asarray(result.diagnostics["q_path_reference_trajectory"], dtype=float)
    s = np.asarray(result.diagnostics["s_trajectory"], dtype=float)
    sdot = np.asarray(result.diagnostics["sdot_trajectory"], dtype=float)
    u = np.asarray(result.control, dtype=float)
    u_t = t[:-1] if u.shape[0] > 0 else t
    control_names = list(result.diagnostics.get("control_names", []))
    joint_names = list(result.diagnostics.get("reduced_joint_names", [f"q{j}" for j in range(q.shape[1])]))
    passive_idx = set(int(i) for i in result.diagnostics.get("passive_v_indices", []))
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=False)

    axes[0].plot(t, s, label="s")
    axes[0].plot(t, sdot, label="sdot")
    axes[0].set_title("Path Progress")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("value")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", title="Path states")

    for j in range(q.shape[1]):
        color = cmap(j % 10)
        jname = joint_names[j] if j < len(joint_names) else f"q{j}"
        ls = "-" if j in passive_idx else "--"
        axes[1].plot(t, q[:, j], ls=ls, lw=1.3, color=color, label=f"{jname} actual")
        axes[1].plot(t, q_ref[:, j], ":", lw=1.0, color=color, alpha=0.9, label=f"{jname} ref")
    axes[1].set_title("Joint Trajectory vs Path Reference")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("joint position")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8, ncol=2, title="Joint position")

    for j in range(dq.shape[1]):
        color = cmap(j % 10)
        jname = joint_names[j] if j < len(joint_names) else f"q{j}"
        ls = "-" if j in passive_idx else "--"
        axes[2].plot(t, dq[:, j], ls=ls, lw=1.2, color=color, label=f"{jname} dq")
    axes[2].set_title("Joint Velocities")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("joint velocity")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best", fontsize=8, ncol=2, title="Joint velocity")

    for i in range(u.shape[1]):
        label = control_names[i] if i < len(control_names) else f"u{i}"
        axes[3].step(u_t, u[:, i], where="post", label=label)
    axes[3].set_title("Control Inputs")
    axes[3].set_xlabel("time [s]")
    axes[3].set_ylabel("command")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best", title="Control")

    fig.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
