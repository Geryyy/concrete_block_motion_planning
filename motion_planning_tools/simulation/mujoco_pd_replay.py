"""MuJoCo computed-torque trajectory replay with optional interactive viewer."""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np
from motion_planning.control import ComputedTorqueController


STARTUP_VIEW = {
    "lookat": (-6.0, 0.0, 0.0),
    "azimuth": 90.0,
    "elevation": -45.0,
    "distance": 20.0,
}
START_KEYFRAME = "start"


# ── SimulationResult ──────────────────────────────────────────────────────────


@dataclass
class SimulationResult:
    """Tracking metrics from a MuJoCo computed-torque replay."""

    time_s: np.ndarray
    q_ref: np.ndarray            # (T, n_act) reference joint positions
    q_actual: np.ndarray         # (T, n_act) simulated joint positions
    dq_ref: np.ndarray           # (T, n_act) reference joint velocities
    dq_actual: np.ndarray        # (T, n_act) simulated joint velocities
    rmse_per_joint: np.ndarray   # (n_act,) position RMSE per actuator
    max_error_per_joint: np.ndarray
    joint_names: List[str]
    tcp_world: Optional[np.ndarray] = None  # (T, 3) optional monitored TCP/site trace

    def plot(self) -> None:
        """Plot reference vs. actual joint positions."""
        import matplotlib.pyplot as plt

        n = len(self.joint_names)
        fig, axes = plt.subplots(n, 1, figsize=(12, 2 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for i, name in enumerate(self.joint_names):
            ax = axes[i]
            ax.plot(self.time_s, np.degrees(self.q_ref[:, i]), label="ref", linestyle="--")
            ax.plot(self.time_s, np.degrees(self.q_actual[:, i]), label="actual")
            ax.set_ylabel("[deg]")
            ax.set_title(f"{name}  RMSE={np.degrees(self.rmse_per_joint[i]):.4f}°")
            ax.legend(fontsize=8)
            ax.grid(True)
        axes[-1].set_xlabel("Time [s]")
        fig.tight_layout()
        plt.show()

    def save(self, path: Union[str, Path]) -> Path:
        """Save metrics to .npz."""
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            sim_time_s=self.time_s,
            controlled_joint_names=np.asarray(self.joint_names, dtype=str),
            sim_q_trajectory=self.q_actual,
            ref_q_trajectory=self.q_ref,
            sim_dq_trajectory=self.dq_actual,
            ref_dq_trajectory=self.dq_ref,
            q_error=self.q_actual - self.q_ref,
            dq_error=self.dq_actual - self.dq_ref,
            q_rmse=self.rmse_per_joint,
            q_max_abs=self.max_error_per_joint,
            tcp_world=self.tcp_world if self.tcp_world is not None else np.empty((0, 3), dtype=float),
        )
        return path


# ── Internal helpers ──────────────────────────────────────────────────────────


def _interp_vector_series(t: float, time_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    tc = float(np.clip(t, float(time_s[0]), float(time_s[-1])))
    out = np.zeros(values.shape[1], dtype=float)
    for i in range(values.shape[1]):
        out[i] = float(np.interp(tc, time_s, values[:, i]))
    return out


def _quat_xyzw_to_rot(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=float).reshape(4)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def _reset_to_keyframe_if_available(model, data, key_name: str) -> None:
    import mujoco

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)


def _load_trajectory(traj) -> tuple:
    """Load from a Trajectory object or a path to a .npz file.

    Returns (time_s, q_ref, dq_ref, reduced_joint_names, actuated_joint_names).
    """
    if isinstance(traj, (str, Path)):
        with np.load(traj, allow_pickle=False) as d:
            time_s = np.asarray(d["time_s"], dtype=float).reshape(-1)
            q_ref = np.asarray(d["q_trajectory"], dtype=float)
            dq_ref = np.asarray(d["dq_trajectory"], dtype=float)
            reduced_joint_names = [
                str(x) for x in np.asarray(d["reduced_joint_names"]).reshape(-1)
            ]
            actuated_joint_names = [
                str(x) for x in np.asarray(d["actuated_joint_names"]).reshape(-1)
            ]
    else:
        # Duck-typed Trajectory object (motion_planning.planner.Trajectory)
        time_s = np.asarray(traj.time_s, dtype=float).reshape(-1)
        q_ref = np.asarray(traj.q, dtype=float)
        dq_ref = np.asarray(traj.dq, dtype=float)
        reduced_joint_names = list(traj.joint_names)
        actuated_joint_names = list(
            traj.actuated_joint_names
            if traj.actuated_joint_names is not None
            else traj.joint_names
        )
    return time_s, q_ref, dq_ref, reduced_joint_names, actuated_joint_names


# ── Public API ────────────────────────────────────────────────────────────────


def replay_trajectory_with_pd(
    trajectory,
    *,
    mujoco_model: Union[str, Path],
    kp: float = 20.0,
    kd: float = 5.0,
    view: bool = True,
    speed: float = 1.0,
    tail_s: float = 2.0,
    report_out: Optional[Path] = None,
    overlay_path_xyz: Optional[np.ndarray] = None,
    overlay_start_xyz: Optional[np.ndarray] = None,
    overlay_goal_xyz: Optional[np.ndarray] = None,
    overlay_blocks: Optional[Sequence[dict[str, Any]]] = None,
    overlay_proxy_provider: Optional[Callable[[Any, Any, float], Sequence[dict[str, Any]]]] = None,
    overlay_tcp_marker: Optional[dict[str, Any]] = None,
    monitor_tcp_site: Optional[str] = None,
    keep_open_after_end: bool = False,
    continue_physics_after_end: bool = False,
    existing_model: Optional[Any] = None,
    existing_data: Optional[Any] = None,
    existing_viewer: Optional[Any] = None,
    reset_to_start_keyframe: bool = True,
) -> Optional[SimulationResult]:
    """Replay a planned trajectory in MuJoCo with computed-torque (PD) control.

    Args:
        trajectory: A ``Trajectory`` object or path to a ``.npz`` file.
        mujoco_model: Path to the MuJoCo scene XML.
        kp: Proportional gain (scalar applied to all joints).
        kd: Derivative gain (scalar applied to all joints).
        view: Open the interactive viewer (``True``) or run headless (``False``).
        speed: Playback speed multiplier (viewer only).
        tail_s: Extra simulation time after the reference horizon.
        report_out: Optional path to save the ``SimulationResult`` .npz.

    Returns:
        ``SimulationResult`` with tracking metrics.
    """
    import mujoco
    import mujoco.viewer

    time_s, q_ref, dq_ref, reduced_joint_names, actuated_joint_names = _load_trajectory(
        trajectory
    )
    if q_ref.shape != dq_ref.shape:
        raise ValueError(f"q and dq shape mismatch: {q_ref.shape} vs {dq_ref.shape}.")
    if q_ref.shape[0] != time_s.shape[0]:
        raise ValueError("time_s length must match q rows.")
    if q_ref.shape[1] != len(reduced_joint_names):
        raise ValueError("reduced_joint_names length must match q columns.")

    qdd_ref = np.gradient(dq_ref, time_s, axis=0, edge_order=2)

    if existing_model is not None and existing_data is not None:
        model = existing_model
        data = existing_data
    else:
        model_path = Path(mujoco_model).expanduser().resolve()
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
    if bool(reset_to_start_keyframe):
        _reset_to_keyframe_if_available(model, data, START_KEYFRAME)

    name_to_jid: dict[str, int] = {}
    name_to_qadr: dict[str, int] = {}
    name_to_vadr: dict[str, int] = {}
    for jid in range(model.njnt):
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        name_to_jid[jn] = int(jid)
        name_to_qadr[jn] = int(model.jnt_qposadr[jid])
        name_to_vadr[jn] = int(model.jnt_dofadr[jid])

    ref_name_to_idx = {jn: i for i, jn in enumerate(reduced_joint_names)}
    actuator_ids: list[int] = []
    actuator_joint_names: list[str] = []
    actuator_dof_idx: list[int] = []
    actuator_gear: list[float] = []
    for aid in range(model.nu):
        if int(model.actuator_trntype[aid]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        jid = int(model.actuator_trnid[aid, 0])
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        actuator_ids.append(aid)
        actuator_joint_names.append(jn)
        actuator_dof_idx.append(name_to_vadr[jn])
        g = float(model.actuator_gear[aid, 0])
        actuator_gear.append(g if abs(g) > 1e-12 else 1.0)

    if not actuator_ids:
        raise RuntimeError("No MuJoCo effort actuators matched the trajectory joints.")

    telescope_ref_name = "q4_big_telescope"
    if telescope_ref_name not in ref_name_to_idx:
        raise KeyError(f"Trajectory missing required joint '{telescope_ref_name}'.")
    q9_name = "q9_left_rail_joint"
    q11_name = "q11_right_rail_joint"
    if q9_name not in name_to_jid or q11_name not in name_to_jid:
        raise KeyError("MuJoCo model missing q9/q11 rail joints.")
    q9_lower = float(model.jnt_range[name_to_jid[q9_name], 0])
    q11_lower = float(model.jnt_range[name_to_jid[q11_name], 0])

    ctrl_min = model.actuator_ctrlrange[actuator_ids, 0].astype(float)
    ctrl_max = model.actuator_ctrlrange[actuator_ids, 1].astype(float)
    kp_vec = np.full(len(actuator_ids), float(kp), dtype=float)
    kd_vec = np.full(len(actuator_ids), float(kd), dtype=float)
    for i, jn in enumerate(actuator_joint_names):
        if jn in (q9_name, q11_name):
            kp_vec[i] = max(kp_vec[i], 400.0)
            kd_vec[i] = max(kd_vec[i], 80.0)
    controller = ComputedTorqueController(kp=kp_vec, kd=kd_vec, u_min=ctrl_min, u_max=ctrl_max)

    def _desired_for_joint(jn, q_ref_t, dq_ref_t, qdd_ref_t):
        tel_idx = ref_name_to_idx[telescope_ref_name]
        if jn in ("q4_big_telescope", "q5_small_telescope"):
            return (
                0.5 * float(q_ref_t[tel_idx]),
                0.5 * float(dq_ref_t[tel_idx]),
                0.5 * float(qdd_ref_t[tel_idx]),
            )
        if jn == q9_name:
            return q9_lower, 0.0, 0.0
        if jn == q11_name:
            return q11_lower, 0.0, 0.0
        if jn in ref_name_to_idx:
            idx = ref_name_to_idx[jn]
            return float(q_ref_t[idx]), float(dq_ref_t[idx]), float(qdd_ref_t[idx])
        return float(data.qpos[name_to_qadr[jn]]), 0.0, 0.0

    # Initialise simulation to start configuration.
    q0 = q_ref[0]
    for jn, idx in ref_name_to_idx.items():
        if jn in name_to_qadr:
            data.qpos[name_to_qadr[jn]] = float(q0[idx])
    tel0 = float(q0[ref_name_to_idx[telescope_ref_name]])
    if "q4_big_telescope" in name_to_qadr:
        data.qpos[name_to_qadr["q4_big_telescope"]] = 0.5 * tel0
    if "q5_small_telescope" in name_to_qadr:
        data.qpos[name_to_qadr["q5_small_telescope"]] = 0.5 * tel0
    data.qpos[name_to_qadr[q9_name]] = q9_lower
    data.qpos[name_to_qadr[q11_name]] = q11_lower
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    all_joint_names: list[str] = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        for jid in range(model.njnt)
    ]

    def _desired_full_state(q_ref_t, dq_ref_t, qdd_ref_t):
        q_des_all = np.asarray(data.qpos, dtype=float).copy()
        dq_des_all = np.asarray(data.qvel, dtype=float).copy()
        qdd_des_all = np.zeros(model.nv, dtype=float)
        for jn in all_joint_names:
            q_d, dq_d, qdd_d = _desired_for_joint(jn, q_ref_t, dq_ref_t, qdd_ref_t)
            q_des_all[name_to_qadr[jn]] = q_d
            dq_des_all[name_to_vadr[jn]] = dq_d
            qdd_des_all[name_to_vadr[jn]] = qdd_d
        return q_des_all, dq_des_all, qdd_des_all

    sim_t_hist: list[float] = []
    q_sim_hist: list[np.ndarray] = []
    q_des_hist: list[np.ndarray] = []
    dq_sim_hist: list[np.ndarray] = []
    dq_des_hist: list[np.ndarray] = []
    tcp_hist: list[np.ndarray] = []
    ctrl_clip_hist = np.zeros(len(actuator_ids), dtype=int)

    tcp_site_id = -1
    if monitor_tcp_site is not None:
        tcp_site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, str(monitor_tcp_site)))

    start_wall = time.perf_counter()
    sim_t0 = float(data.time)
    horizon = float(time_s[-1])
    end_time = horizon + max(0.0, float(tail_s))

    def step_once(sync_viewer) -> bool:
        t_rel = float(data.time) - sim_t0
        if t_rel > end_time and not (bool(view) and bool(continue_physics_after_end)):
            return False

        q_ref_t = _interp_vector_series(min(t_rel, horizon), time_s, q_ref)
        dq_ref_t = _interp_vector_series(min(t_rel, horizon), time_s, dq_ref)
        qdd_ref_t = _interp_vector_series(min(t_rel, horizon), time_s, qdd_ref)
        q_des_all, dq_des_all, qdd_des_all = _desired_full_state(q_ref_t, dq_ref_t, qdd_ref_t)

        q_des = np.array([q_des_all[name_to_qadr[jn]] for jn in actuator_joint_names], dtype=float)
        dq_des = np.array([dq_des_all[name_to_vadr[jn]] for jn in actuator_joint_names], dtype=float)
        q_now = np.array([float(data.qpos[name_to_qadr[jn]]) for jn in actuator_joint_names], dtype=float)
        dq_now = np.array([float(data.qvel[name_to_vadr[jn]]) for jn in actuator_joint_names], dtype=float)
        qdd_ff = np.array([qdd_des_all[dof_idx] for dof_idx in actuator_dof_idx], dtype=float)
        qacc_cmd = controller.compute_acceleration(
            q_des=q_des,
            dq_des=dq_des,
            q=q_now,
            dq=dq_now,
            qdd_ff=qdd_ff,
        )

        data.qacc[:] = qdd_des_all
        for i, dof_idx in enumerate(actuator_dof_idx):
            data.qacc[dof_idx] = float(qacc_cmd[i])
        mujoco.mj_inverse(model, data)

        tau_vec = np.array([float(data.qfrc_inverse[dof_idx]) for dof_idx in actuator_dof_idx], dtype=float)
        u_cmd, clipped = controller.torque_to_control(tau=tau_vec, gear=np.asarray(actuator_gear, dtype=float))
        ctrl_clip_hist[:] += clipped.astype(int)
        data.ctrl[:] = 0.0
        for i, aid in enumerate(actuator_ids):
            data.ctrl[aid] = float(u_cmd[i])

        mujoco.mj_step(model, data)

        t_new = float(data.time) - sim_t0
        q_ref_new = _interp_vector_series(min(t_new, horizon), time_s, q_ref)
        dq_ref_new = _interp_vector_series(min(t_new, horizon), time_s, dq_ref)
        q_des_n, dq_des_n, _ = _desired_full_state(q_ref_new, dq_ref_new, np.zeros(len(reduced_joint_names)))

        sim_t_hist.append(t_new)
        q_sim_hist.append(np.array([float(data.qpos[name_to_qadr[jn]]) for jn in actuator_joint_names]))
        q_des_hist.append(np.array([q_des_n[name_to_qadr[jn]] for jn in actuator_joint_names]))
        dq_sim_hist.append(np.array([float(data.qvel[name_to_vadr[jn]]) for jn in actuator_joint_names]))
        dq_des_hist.append(np.array([dq_des_n[name_to_vadr[jn]] for jn in actuator_joint_names]))
        if tcp_site_id >= 0:
            tcp_hist.append(np.asarray(data.site_xpos[tcp_site_id], dtype=float).reshape(3).copy())

        target_wall = start_wall + t_new / max(speed, 1e-6)
        sleep_dt = target_wall - time.perf_counter()
        if sleep_dt > 0:
            time.sleep(sleep_dt)
        if sync_viewer is not None:
            sync_viewer()
        return True

    def _draw_overlay(viewer) -> None:
        if viewer is None:
            return
        scn = viewer.user_scn
        scn.ngeom = 0
        if overlay_path_xyz is not None:
            pts = np.asarray(overlay_path_xyz, dtype=float).reshape(-1, 3)
            n_pts = min(pts.shape[0], 120)
            if n_pts > 0:
                sel = np.linspace(0, pts.shape[0] - 1, n_pts).astype(int)
                for idx in sel:
                    if scn.ngeom >= len(scn.geoms):
                        break
                    mujoco.mjv_initGeom(
                        scn.geoms[scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([0.045, 0.0, 0.0], dtype=float),
                        pos=np.asarray(pts[idx], dtype=float),
                        mat=np.eye(3, dtype=float).reshape(-1),
                        rgba=np.array([0.1, 0.5, 1.0, 0.8], dtype=float),
                    )
                    scn.ngeom += 1
        if overlay_start_xyz is not None and scn.ngeom < len(scn.geoms):
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.10, 0.0, 0.0], dtype=float),
                pos=np.asarray(overlay_start_xyz, dtype=float).reshape(3),
                mat=np.eye(3, dtype=float).reshape(-1),
                rgba=np.array([1.0, 0.1, 0.1, 0.95], dtype=float),
            )
            scn.ngeom += 1
        if overlay_goal_xyz is not None and scn.ngeom < len(scn.geoms):
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.10, 0.0, 0.0], dtype=float),
                pos=np.asarray(overlay_goal_xyz, dtype=float).reshape(3),
                mat=np.eye(3, dtype=float).reshape(-1),
                rgba=np.array([0.1, 0.95, 0.1, 0.95], dtype=float),
            )
            scn.ngeom += 1
        if overlay_blocks is not None:
            for block in overlay_blocks:
                if scn.ngeom >= len(scn.geoms):
                    break
                size = np.asarray(block["size"], dtype=float).reshape(3)
                pos = np.asarray(block["position"], dtype=float).reshape(3)
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=0.5 * size,
                    pos=pos,
                    mat=np.eye(3, dtype=float).reshape(-1),
                    rgba=np.array([0.85, 0.45, 0.05, 0.40], dtype=float),
                )
                scn.ngeom += 1
        if overlay_proxy_provider is not None and scn.ngeom < len(scn.geoms):
            t_rel = float(data.time) - sim_t0
            proxy_boxes_world = overlay_proxy_provider(model, data, t_rel)
            for proxy in proxy_boxes_world:
                if scn.ngeom >= len(scn.geoms):
                    break
                pos = np.asarray(proxy["position"], dtype=float).reshape(3)
                if "mat" in proxy:
                    mat = np.asarray(proxy["mat"], dtype=float).reshape(3, 3).reshape(-1)
                else:
                    mat = _quat_xyzw_to_rot(proxy.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0])).reshape(-1)
                size = 0.5 * np.asarray(proxy.get("size", [0.1, 0.1, 0.1]), dtype=float).reshape(3)
                rgba = np.asarray(proxy.get("rgba", [0.95, 0.1, 0.1, 0.35]), dtype=float).reshape(4)
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=size,
                    pos=pos,
                    mat=mat,
                    rgba=rgba,
                )
                scn.ngeom += 1
        if overlay_tcp_marker is not None and scn.ngeom < len(scn.geoms):
            p_marker = None
            site_name = overlay_tcp_marker.get("site", None)
            if site_name is not None:
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, str(site_name))
                if sid >= 0:
                    p_marker = np.asarray(data.site_xpos[sid], dtype=float).reshape(3)
            if p_marker is None:
                body_name = str(overlay_tcp_marker.get("body", "K8_tool_center_point"))
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if bid >= 0:
                    p_w = np.asarray(data.xpos[bid], dtype=float).reshape(3)
                    R_wb = np.asarray(data.xmat[bid], dtype=float).reshape(3, 3)
                    offset_local = np.asarray(overlay_tcp_marker.get("offset_local", [0.0, 0.0, 1.0]), dtype=float).reshape(3)
                    p_marker = p_w + R_wb @ offset_local
            if p_marker is not None:
                radius = float(overlay_tcp_marker.get("radius", 0.08))
                rgba = np.asarray(overlay_tcp_marker.get("rgba", [0.1, 0.35, 1.0, 0.95]), dtype=float).reshape(4)
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([radius, 0.0, 0.0], dtype=float),
                    pos=p_marker,
                    mat=np.eye(3, dtype=float).reshape(-1),
                    rgba=rgba,
                )
                scn.ngeom += 1

    if not view:
        while step_once(None):
            pass
    else:
        if existing_viewer is not None:
            viewer = existing_viewer
            def _sync_existing():
                _draw_overlay(viewer)
                viewer.sync()
            while viewer.is_running():
                if step_once(_sync_existing):
                    continue
                if not bool(keep_open_after_end):
                    break
                while viewer.is_running():
                    _sync_existing()
                    time.sleep(0.02)
                break
        else:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                viewer.cam.lookat[:] = STARTUP_VIEW["lookat"]
                viewer.cam.azimuth = STARTUP_VIEW["azimuth"]
                viewer.cam.elevation = STARTUP_VIEW["elevation"]
                viewer.cam.distance = STARTUP_VIEW["distance"]
                viewer.sync()
                def _sync():
                    _draw_overlay(viewer)
                    viewer.sync()
                while viewer.is_running():
                    if step_once(_sync):
                        continue
                    if not bool(keep_open_after_end):
                        break
                    while viewer.is_running():
                        _sync()
                        time.sleep(0.02)
                    break

    sim_t = np.asarray(sim_t_hist, dtype=float)
    q_sim = np.asarray(q_sim_hist, dtype=float)
    q_des = np.asarray(q_des_hist, dtype=float)
    dq_sim = np.asarray(dq_sim_hist, dtype=float)
    dq_des = np.asarray(dq_des_hist, dtype=float)
    tcp_world = np.asarray(tcp_hist, dtype=float) if tcp_hist else None

    if sim_t.size == 0:
        print("[pd-replay] No simulation samples collected.")
        return None

    q_err = q_sim - q_des
    q_rmse = np.sqrt(np.mean(np.square(q_err), axis=0))
    q_max_abs = np.max(np.abs(q_err), axis=0)

    result = SimulationResult(
        time_s=sim_t,
        q_ref=q_des,
        q_actual=q_sim,
        dq_ref=dq_des,
        dq_actual=dq_sim,
        rmse_per_joint=q_rmse,
        max_error_per_joint=q_max_abs,
        joint_names=actuator_joint_names,
        tcp_world=tcp_world,
    )

    print("=== Computed-Torque Replay Tracking Error ===")
    for i, jn in enumerate(actuator_joint_names):
        print(f"{jn:<24} q_rmse={q_rmse[i]:.6f} q_max={q_max_abs[i]:.6f}")
    if sim_t.size > 0:
        print("=== Actuator Control Clipping ===")
        for i, jn in enumerate(actuator_joint_names):
            print(f"{jn:<24} clip_frac={ctrl_clip_hist[i] / sim_t.size:.3f}")

    if report_out is not None:
        result.save(report_out)
        print("[pd-replay] saved_report:", str(report_out))

    return result


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_model = repo_root / "crane_urdf" / "crane.xml"

    parser = argparse.ArgumentParser(
        description="Replay optimized OCP trajectory in MuJoCo with computed-torque control."
    )
    parser.add_argument("--traj", type=Path, default=Path("/tmp/crane_acados_ocp_trajectory.npz"))
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument("--kp", type=float, default=80.0)
    parser.add_argument("--kd", type=float, default=20.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--tail-s", type=float, default=2.0)
    parser.add_argument("--no-view", action="store_true")
    parser.add_argument("--report-out", type=Path, default=Path("/tmp/crane_pd_replay_report.npz"))
    args = parser.parse_args()

    traj_path = args.traj.expanduser().resolve()
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")
    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model not found: {model_path}")

    replay_trajectory_with_pd(
        traj_path,
        mujoco_model=model_path,
        kp=float(args.kp),
        kd=float(args.kd),
        view=not bool(args.no_view),
        speed=float(args.speed),
        tail_s=float(args.tail_s),
        report_out=args.report_out.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
