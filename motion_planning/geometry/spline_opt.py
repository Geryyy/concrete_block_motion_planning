import numpy as np
from typing import Tuple, Callable, Dict, Optional, List, Any

try:
    from scipy.interpolate import make_interp_spline
    from scipy.optimize import minimize
except ImportError as e:
    raise ImportError("Please install SciPy: pip install scipy") from e


def build_cubic_bspline(points: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build a cubic (C2) B-spline interpolant through the given waypoints.
    points: (N,3) with N >= 4 for cubic interpolation.
    Returns S(u) with u in [0,1].
    """
    n = points.shape[0]
    if n < 4:
        raise ValueError("Cubic B-spline interpolation requires at least 4 waypoints.")
    u = np.linspace(0.0, 1.0, n)
    spline = make_interp_spline(u, points, k=3, axis=0)
    return lambda uq: spline(np.asarray(uq, dtype=float))


def build_scalar_bspline(values: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Build a smooth scalar spline with automatic degree selection."""
    y = np.asarray(values, dtype=float).reshape(-1)
    n = y.size
    if n < 2:
        raise ValueError("Scalar spline requires at least 2 control points.")
    u = np.linspace(0.0, 1.0, n)
    k = min(3, n - 1)
    spline = make_interp_spline(u, y, k=k, axis=0)
    return lambda uq: np.asarray(spline(np.asarray(uq, dtype=float)), dtype=float)


def yaw_deg_to_quat(yaw_deg: float) -> Tuple[float, float, float, float]:
    """Quaternion [x,y,z,w] for a pure yaw rotation about +z."""
    half = 0.5 * np.deg2rad(float(yaw_deg))
    return (0.0, 0.0, float(np.sin(half)), float(np.cos(half)))


def _quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.asarray(q1, dtype=float).reshape(4)
    x2, y2, z2, w2 = np.asarray(q2, dtype=float).reshape(4)
    return np.asarray(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=float,
    )


def _quat_normalize_xyzw(q: np.ndarray) -> np.ndarray:
    qq = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(qq))
    if n <= 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)
    return qq / n


def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, y, z, w = _quat_normalize_xyzw(q)
    R = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
    return R @ np.asarray(v, dtype=float).reshape(3)


def sample_curve(S: Callable, n: int = 101) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample curve and first differences for length/curvature approximations.
    Returns (P, dP) with shapes (n,3) and (n-1,3)
    """
    us = np.linspace(0.0, 1.0, n)
    P = S(us)
    dP = np.diff(P, axis=0)
    return P, dP


def path_length(P: np.ndarray) -> float:
    """
    Discrete path length of P (n,3)
    """
    dP = np.diff(P, axis=0)
    seg = np.linalg.norm(dP, axis=1)
    return float(np.sum(seg))


def curvature_cost(P: np.ndarray) -> float:
    """
    Discrete bending energy approximation: integral(kappa^2 ds).
    Low value -> smoother, less sharply curved paths.
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    if n < 3:
        return 0.0
    du = 1.0 / float(n - 1)
    d1 = np.gradient(P, du, axis=0)
    d2 = np.gradient(d1, du, axis=0)
    speed = np.linalg.norm(d1, axis=1)
    cross = np.linalg.norm(np.cross(d1, d2), axis=1)
    eps = 1e-9
    kappa = cross / np.maximum(speed, eps) ** 3
    return float(np.sum((kappa * kappa) * speed) * du)


def mean_turn_angle_deg(P: np.ndarray, eps: float = 1e-12) -> float:
    """Mean turning angle between consecutive segments (degrees)."""
    dP = np.diff(P, axis=0)
    if dP.shape[0] < 2:
        return 0.0
    a = dP[:-1]
    b = dP[1:]
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    valid = (an > eps) & (bn > eps)
    if not np.any(valid):
        return 0.0
    cosang = np.sum(a[valid] * b[valid], axis=1) / (an[valid] * bn[valid])
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.arccos(cosang)
    return float(np.degrees(np.mean(ang)))


def yaw_smoothness_cost(yaw_deg_samples: np.ndarray) -> float:
    """Smoothness penalty for yaw profile."""
    y = np.asarray(yaw_deg_samples, dtype=float).reshape(-1)
    if y.size < 3:
        return 0.0
    D2 = y[:-2] - 2.0 * y[1:-1] + y[2:]
    return float(np.sum(D2 * D2))


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(arr))
    if n < eps:
        return np.zeros_like(arr)
    return arr / n


def _goal_approach_alignment_cost(
    P: np.ndarray,
    goal_normals: np.ndarray,
    terminal_fraction: float = 0.1,
) -> float:
    """Penalize final approach direction that is not aligned with -summed goal normals."""
    if P.shape[0] < 3 or goal_normals.size == 0:
        return 0.0
    tail_n = max(3, int(np.ceil(float(terminal_fraction) * P.shape[0])))
    P_tail = P[-tail_n:]
    seg = np.diff(P_tail, axis=0)
    if seg.shape[0] == 0:
        return 0.0
    v = _normalize(np.sum(seg, axis=0))
    if not np.any(v):
        return 0.0

    N = np.asarray(goal_normals, dtype=float).reshape(-1, 3)
    Nn = np.array([_normalize(n) for n in N], dtype=float)
    if Nn.size == 0:
        return 0.0
    s = _normalize(np.sum(Nn, axis=0))
    if not np.any(s):
        s = _normalize(Nn[0])
    if not np.any(s):
        return 0.0
    # Approach should move opposite to resultant outward surface normal.
    c = float(np.dot(v, -s))
    return float((1.0 - np.clip(c, -1.0, 1.0)) ** 2)


def _path_distances(
    scene,
    P: np.ndarray,
    moving_block_size: Optional[Tuple[float, float, float]] = None,
    moving_block_quat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    moving_block_quats: Optional[np.ndarray] = None,
    moving_proxies: Optional[List[Dict[str, Any]]] = None,
    ignore_ids: Optional[List[str]] = None,
) -> np.ndarray:
    """Distance profile for points, one moving block, or multiple moving proxy blocks."""
    use_single_block = moving_block_size is not None
    use_multi_proxy = bool(moving_proxies)
    if not use_single_block and not use_multi_proxy:
        return np.array([scene.signed_distance(p) for p in P], dtype=float)

    if moving_block_quats is not None:
        Q = np.asarray(moving_block_quats, dtype=float)
        if Q.shape != (P.shape[0], 4):
            raise ValueError("moving_block_quats must have shape (len(P), 4)")
    else:
        q0 = _quat_normalize_xyzw(np.asarray(moving_block_quat, dtype=float).reshape(4))
        Q = np.tile(q0.reshape(1, 4), (P.shape[0], 1))

    d_all = np.full(P.shape[0], np.inf, dtype=float)
    for i, p in enumerate(P):
        q_body = _quat_normalize_xyzw(Q[i])
        d_i = np.inf
        if use_single_block:
            d_i = min(
                d_i,
                float(
                    scene.signed_distance_block(
                        size=moving_block_size,
                        position=p,
                        quat=tuple(q_body.tolist()),
                        ignore_ids=ignore_ids,
                    )
                ),
            )
        if use_multi_proxy:
            for proxy in moving_proxies or []:
                size = tuple(float(v) for v in proxy["size"])
                offset = np.asarray(proxy.get("offset_xyz", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
                q_local = _quat_normalize_xyzw(np.asarray(proxy.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0]), dtype=float))
                p_world = np.asarray(p, dtype=float).reshape(3) + _quat_rotate_xyzw(q_body, offset)
                q_world = _quat_normalize_xyzw(_quat_mul_xyzw(q_body, q_local))
                d_i = min(
                    d_i,
                    float(
                        scene.signed_distance_block(
                            size=size,
                            position=p_world,
                            quat=tuple(q_world.tolist()),
                            ignore_ids=ignore_ids,
                        )
                    ),
                )
        d_all[i] = d_i
    return d_all


def safety_cost(
    scene,
    P: np.ndarray,
    required_clearance: float,
    moving_block_size: Optional[Tuple[float, float, float]] = None,
    moving_block_quat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    moving_block_quats: Optional[np.ndarray] = None,
    moving_proxies: Optional[List[Dict[str, Any]]] = None,
    ignore_ids: Optional[List[str]] = None,
) -> float:
    """
    Safety penalty from sampled signed distances along the path.
    cost = sum(max(0, required_clearance - d_i)^2)
    """
    dists = _path_distances(
        scene,
        P,
        moving_block_size=moving_block_size,
        moving_block_quat=moving_block_quat,
        moving_block_quats=moving_block_quats,
        moving_proxies=moving_proxies,
        ignore_ids=ignore_ids,
    )
    deficit = np.maximum(0.0, float(required_clearance) - dists)
    return float(np.sum(deficit * deficit))


def _default_via_initialization(start: np.ndarray, goal: np.ndarray, n_vias: int) -> np.ndarray:
    """Place vias uniformly on the line from start to goal."""
    if n_vias <= 0:
        return np.empty((0, 3), dtype=float)
    t = np.linspace(1.0 / (n_vias + 1), n_vias / (n_vias + 1), n_vias)
    return start[None, :] + t[:, None] * (goal - start)[None, :]


def _simple_cem_optimize(
    objective: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    sigma0: np.ndarray,
    population_size: int = 72,
    elite_frac: float = 0.2,
    max_iter: int = 80,
    min_sigma: float = 1e-3,
    init_scale: Optional[float] = None,
    sample_method: str = "Gaussian",
    seed: Optional[int] = None,
):
    """Simple CEM wrapper using the external `cross_entropy_method` implementation."""
    try:
        from cross_entropy_method.cem import CEM  # type: ignore
    except Exception:
        try:
            from cem import CEM  # type: ignore
        except Exception as exc:
            from pathlib import Path
            import sys

            # Support submodule layout: <repo_root>/cross_entropy_method/cem.py
            cem_dir = Path(__file__).resolve().parents[1] / "cross_entropy_method"
            if cem_dir.is_dir() and str(cem_dir) not in sys.path:
                sys.path.insert(0, str(cem_dir))
            try:
                from cem import CEM  # type: ignore
            except Exception as exc2:
                raise ImportError(
                    "Simple CEM backend requires `cem.py` from cross_entropy_method. "
                    "Expected either import `cross_entropy_method.cem` or module `cem`."
                ) from exc2

    if seed is not None:
        import random

        np.random.seed(int(seed))
        random.seed(int(seed))

    x0 = np.asarray(x0, dtype=float).reshape(-1)
    sigma = np.maximum(np.asarray(sigma0, dtype=float).reshape(-1), float(min_sigma))
    d = int(x0.size)

    if init_scale is None:
        init_scale = float(np.mean(sigma))

    sample_method = str(sample_method).strip().capitalize()
    if sample_method not in {"Gaussian", "Uniform"}:
        raise ValueError("sample_method must be either 'Gaussian' or 'Uniform'.")

    # Keep sampling bounded around x0 for numerical stability.
    v_min = (x0 - 6.0 * sigma).tolist()
    v_max = (x0 + 6.0 * sigma).tolist()
    Ne = max(2, int(np.ceil(float(population_size) * float(elite_frac))))

    def _target(X: np.ndarray):
        X = np.asarray(X, dtype=float)
        vals = objective(X)
        return np.asarray(vals, dtype=float).reshape(-1).tolist()

    cem = CEM(
        func=_target,
        d=d,
        maxits=int(max_iter),
        N=int(population_size),
        Ne=int(Ne),
        argmin=True,
        v_min=v_min,
        v_max=v_max,
        init_scale=float(init_scale),
        sampleMethod=sample_method,
        init_mu=x0,
        init_sigma=sigma,
    )
    x_best = np.asarray(cem.eval(), dtype=float).reshape(-1)
    fun = float(objective(x_best.reshape(1, -1))[0])

    return {
        "x": x_best,
        "fun": fun,
        "nit": int(max_iter),
        "success": True,
        "message": "Simple CEM finished",
    }


def _solve_optimizer(
    objective_single: Callable[[np.ndarray], Tuple],
    x0: np.ndarray,
    sigma0: np.ndarray,
    method: str,
    options: Optional[Dict] = None,
) -> Dict:
    method_upper = method.upper()
    if method_upper == "CEM":
        cem_options = {
            "population_size": 72,
            "elite_frac": 0.2,
            "max_iter": 80,
            "min_sigma": 1e-3,
            "init_scale": None,
            "sample_method": "Gaussian",
            "seed": None,
        }
        if options:
            for key, value in options.items():
                if key in cem_options:
                    cem_options[key] = value

        def objective_batch(X):
            return np.array([objective_single(x)[0] for x in X], dtype=float)

        res = _simple_cem_optimize(objective_batch, x0=x0, sigma0=sigma0, **cem_options)
        return {
            "x": res["x"],
            "success": bool(res["success"]),
            "message": str(res["message"]),
            "nit": int(res["nit"]),
            "fun": float(res["fun"]),
        }

    if method_upper in {"NELDER", "NEAD-MELDER", "NEAD_MELDER"}:
        method = "Nelder-Mead"
        method_upper = "NELDER-MEAD"
    if method_upper == "POWELL":
        scipy_options = {"maxiter": 220, "xtol": 1e-3, "ftol": 1e-3}
    elif method_upper == "NELDER-MEAD":
        scipy_options = {"maxiter": 300, "xatol": 1e-3, "fatol": 1e-3}
    else:
        scipy_options = {"maxiter": 250, "xatol": 1e-3, "fatol": 1e-3}
    if options:
        method_key = method.lower().replace("-", "_")
        if isinstance(options.get(method_key), dict):
            scipy_options.update(options[method_key])
        else:
            scipy_options.update(options)
    scipy_res = minimize(
        lambda x: objective_single(x)[0],
        x0,
        method=method,
        options=scipy_options,
    )
    return {
        "x": scipy_res.x,
        "success": bool(scipy_res.success),
        "message": str(scipy_res.message),
        "nit": int(getattr(scipy_res, "nit", 0)),
        "fun": float(scipy_res.fun),
    }


def optimize_bspline_path(
    scene,
    start: np.ndarray,
    goal: np.ndarray,
    n_vias: int = 3,
    initial_vias: Optional[np.ndarray] = None,
    tool_half_extents: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    moving_block_size: Optional[Tuple[float, float, float]] = None,
    moving_block_quat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    moving_proxies: Optional[List[Dict[str, Any]]] = None,
    collision_ignore_ids: Optional[List[str]] = None,
    safety_margin: float = 0.01,
    n_samples_curve: int = 121,
    collision_check_subsample: int = 1,
    start_yaw_deg: float = 0.0,
    goal_yaw_deg: float = 0.0,
    n_yaw_vias: int = 0,
    combined_4d: bool = True,
    w_len: float = 1.0,
    w_curv: float = 0.1,
    w_yaw_smooth: float = 0.0,
    w_safe: float = 50.0,
    preferred_safety_margin: Optional[float] = None,
    relax_preferred_final_fraction: float = 0.0,
    w_safe_preferred: float = 0.0,
    w_approach_rebound: float = 0.0,
    w_goal_clearance: float = 0.0,
    goal_clearance_target: Optional[float] = None,
    w_goal_clearance_target: float = 0.0,
    approach_only_clearance: Optional[float] = None,
    contact_window_fraction: float = 0.1,
    w_approach_clearance: float = 0.0,
    w_approach_collision: float = 0.0,
    approach_fraction: float = 0.2,
    w_via_dev: float = 0.0,
    w_yaw_dev: float = 0.0,
    w_yaw_monotonic: float = 0.0,
    yaw_goal_reach_u: float = 1.0,
    w_yaw_schedule: float = 0.0,
    goal_approach_normals: Optional[np.ndarray] = None,
    goal_approach_window_fraction: float = 0.1,
    w_goal_approach_normal: float = 0.0,
    cost_mode: str = "full",
    init_offset_scale: float = 1.0,
    method: str = "Powell",
    options: Optional[Dict] = None,
) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, Dict]:
    """
    Optimize all vias for a cubic B-spline through [start, vias..., goal].
    """
    start = np.asarray(start, float).reshape(3)
    goal = np.asarray(goal, float).reshape(3)
    if n_vias < 2:
        raise ValueError("n_vias must be >= 2 (cubic spline needs >=4 points total).")
    if collision_check_subsample < 1:
        raise ValueError("collision_check_subsample must be >= 1")
    if n_yaw_vias < 0:
        raise ValueError("n_yaw_vias must be >= 0")
    if combined_4d and n_yaw_vias not in (0, n_vias):
        raise ValueError("For combined_4d=True, n_yaw_vias must be 0 or equal to n_vias.")
    if not (0.0 < float(approach_fraction) <= 1.0):
        raise ValueError("approach_fraction must be in (0, 1].")
    if not (0.0 < float(contact_window_fraction) < 1.0):
        raise ValueError("contact_window_fraction must be in (0, 1).")
    if not (0.0 < float(yaw_goal_reach_u) <= 1.0):
        raise ValueError("yaw_goal_reach_u must be in (0, 1].")
    if not (0.0 <= float(relax_preferred_final_fraction) < 1.0):
        raise ValueError("relax_preferred_final_fraction must be in [0, 1).")
    if not (0.0 < float(goal_approach_window_fraction) <= 1.0):
        raise ValueError("goal_approach_window_fraction must be in (0, 1].")
    cost_mode = str(cost_mode).strip().lower()
    if cost_mode not in {"full", "simple"}:
        raise ValueError("cost_mode must be one of: 'full', 'simple'.")

    if moving_block_size is None and any(float(v) > 0.0 for v in tool_half_extents):
        hx, hy, hz = map(float, tool_half_extents)
        moving_block_size = (2.0 * hx, 2.0 * hy, 2.0 * hz)

    required_clearance = float(safety_margin)
    preferred_clearance = float(preferred_safety_margin) if preferred_safety_margin is not None else required_clearance
    preferred_clearance = max(preferred_clearance, required_clearance)
    if initial_vias is None:
        via_init = _default_via_initialization(start, goal, n_vias)
    else:
        via_init = np.asarray(initial_vias, dtype=float).reshape(-1, 3)
        if via_init.shape != (n_vias, 3):
            raise ValueError(
                f"initial_vias must have shape ({n_vias}, 3), got {via_init.shape}."
            )
    x0_pos = via_init.reshape(-1)
    yaw_via_count = n_vias if combined_4d else n_yaw_vias
    has_yaw_opt = bool(yaw_via_count > 0)

    if has_yaw_opt:
        yaw_ctrl_ref = np.linspace(float(start_yaw_deg), float(goal_yaw_deg), yaw_via_count + 2, dtype=float)
        yaw_via_init = np.linspace(
            float(start_yaw_deg),
            float(goal_yaw_deg),
            yaw_via_count + 2,
            dtype=float,
        )[1:-1]
        x0 = np.hstack([x0_pos, yaw_via_init])
    else:
        yaw_ctrl_ref = np.array([float(start_yaw_deg), float(goal_yaw_deg)], dtype=float)
        x0 = x0_pos

    sigma_base = np.linalg.norm(goal - start) * init_offset_scale / max(n_vias, 1)
    sigma0_pos = np.full_like(x0_pos, max(0.05, sigma_base), dtype=float)
    if has_yaw_opt:
        sigma0_yaw = np.full((yaw_via_count,), 20.0, dtype=float)
        sigma0 = np.hstack([sigma0_pos, sigma0_yaw])
    else:
        sigma0 = sigma0_pos

    us = np.linspace(0.0, 1.0, n_samples_curve)

    def _decode_yaw_controls(x: np.ndarray) -> np.ndarray:
        if has_yaw_opt:
            yaw_vias = x[x0_pos.size:].reshape(yaw_via_count)
            return np.hstack([[float(start_yaw_deg)], yaw_vias, [float(goal_yaw_deg)]])
        return np.array([float(start_yaw_deg), float(goal_yaw_deg)], dtype=float)

    def objective_single(x: np.ndarray):
        vias = x[:x0_pos.size].reshape(n_vias, 3)

        yaw_ctrl = _decode_yaw_controls(x)
        if combined_4d:
            W4 = np.hstack(
                [
                    np.vstack([start, vias, goal]),
                    yaw_ctrl.reshape(-1, 1),
                ]
            )
            S4 = build_cubic_bspline(W4)
            Q4 = S4(us)
            P = Q4[:, :3]
            yaw_samples_deg = Q4[:, 3]
        else:
            W = np.vstack([start, vias, goal])
            S = build_cubic_bspline(W)
            P = S(us)
            S_yaw = build_scalar_bspline(yaw_ctrl)
            yaw_samples_deg = S_yaw(us)
        yaw_quats = np.array([yaw_deg_to_quat(y) for y in yaw_samples_deg], dtype=float)

        if collision_check_subsample > 1:
            idx_safe = np.arange(0, P.shape[0], collision_check_subsample, dtype=int)
            if idx_safe[-1] != P.shape[0] - 1:
                idx_safe = np.append(idx_safe, P.shape[0] - 1)
            P_safe = P[idx_safe]
            yaw_quats_safe = yaw_quats[idx_safe]
            us_safe = us[idx_safe]
        else:
            P_safe = P
            yaw_quats_safe = yaw_quats
            us_safe = us

        j_len = path_length(P)
        j_curv = curvature_cost(P)
        j_yaw = yaw_smoothness_cost(yaw_samples_deg)
        d_safe_raw = _path_distances(
            scene,
            P_safe,
            moving_block_size=moving_block_size,
            moving_block_quat=moving_block_quat,
            moving_block_quats=yaw_quats_safe,
            moving_proxies=moving_proxies,
            ignore_ids=collision_ignore_ids,
        )
        # Keep objective numerically stable when distance queries return non-finite values
        # (e.g. +inf in empty-space scenes).
        clearance_hi = max(float(preferred_clearance), float(required_clearance), 1.0)
        d_safe = np.asarray(d_safe_raw, dtype=float).copy()
        d_safe = np.where(np.isnan(d_safe), float(required_clearance), d_safe)
        d_safe = np.where(np.isposinf(d_safe), clearance_hi, d_safe)
        d_safe = np.where(np.isneginf(d_safe), -clearance_hi, d_safe)
        def_req = np.maximum(0.0, required_clearance - d_safe)
        j_safe = float(np.sum(def_req * def_req))
        j_safe_pref = 0.0
        if preferred_clearance > required_clearance and w_safe_preferred > 0.0:
            if relax_preferred_final_fraction > 0.0:
                keep_n = max(1, int(np.floor((1.0 - relax_preferred_final_fraction) * d_safe.shape[0])))
                d_pref = d_safe[:keep_n]
            else:
                d_pref = d_safe
            def_pref = np.maximum(0.0, preferred_clearance - d_pref)
            j_safe_pref = float(np.sum(def_pref * def_pref))

        # Penalize clearance rebound in the final approach segment.
        n_tail = max(3, int(np.ceil(float(approach_fraction) * d_safe.shape[0])))
        tail = d_safe[-n_tail:]
        tail_inc = np.maximum(0.0, np.diff(tail))
        j_approach_rebound = float(np.sum(tail_inc * tail_inc))

        # Penalize ending farther than preferred clearance (encourages proper approach).
        end_clear = float(d_safe[-1])
        j_goal_clear = float(max(0.0, end_clear - preferred_clearance) ** 2)
        j_goal_target = 0.0
        if goal_clearance_target is not None and w_goal_clearance_target > 0.0:
            j_goal_target = float((end_clear - float(goal_clearance_target)) ** 2)

        # Keep clearance in approach, allow near-contact only in the terminal contact window.
        approach_mask = us_safe < (1.0 - float(contact_window_fraction))
        if np.any(approach_mask):
            d_approach = d_safe[approach_mask]
        else:
            d_approach = d_safe[:-1] if d_safe.shape[0] > 1 else d_safe
        approach_target = preferred_clearance if approach_only_clearance is None else float(approach_only_clearance)
        def_approach = np.maximum(0.0, approach_target - d_approach)
        j_approach_clear = float(np.sum(def_approach * def_approach))
        col_approach = np.maximum(0.0, -d_approach)
        j_approach_col = float(np.sum(col_approach * col_approach))

        # Penalize unnecessary control-point movement from straight-line/yaw interpolation.
        j_via_dev = float(np.sum((vias - via_init) ** 2))
        j_yaw_dev = float(np.sum((yaw_ctrl - yaw_ctrl_ref) ** 2))

        # Enforce one-way yaw motion (no back-and-forth).
        dyaw = np.diff(yaw_samples_deg)
        if float(goal_yaw_deg) >= float(start_yaw_deg):
            backtrack = np.maximum(0.0, -dyaw)
        else:
            backtrack = np.maximum(0.0, dyaw)
        j_yaw_mono = float(np.sum(backtrack * backtrack))

        # Encourage yaw to reach target orientation early (e.g. around u=0.5).
        t_sched = np.clip(us / float(yaw_goal_reach_u), 0.0, 1.0)
        yaw_sched = float(start_yaw_deg) + (float(goal_yaw_deg) - float(start_yaw_deg)) * t_sched
        j_yaw_sched = float(np.sum((yaw_samples_deg - yaw_sched) ** 2))
        j_goal_normal = _goal_approach_alignment_cost(
            P,
            goal_normals=np.asarray(goal_approach_normals, dtype=float) if goal_approach_normals is not None else np.empty((0, 3), dtype=float),
            terminal_fraction=float(goal_approach_window_fraction),
        )

        if cost_mode == "simple":
            # Compact objective that keeps the main planning intent:
            # short + smooth + safe, with optional yaw and goal-approach shaping.
            j = (
                w_len * j_len
                + w_curv * j_curv
                + w_safe * j_safe
                + w_approach_collision * j_approach_col
                + w_yaw_smooth * j_yaw
                + w_goal_approach_normal * j_goal_normal
            )
        else:
            j = (
                w_len * j_len
                + w_curv * j_curv
                + w_yaw_smooth * j_yaw
                + w_safe * j_safe
                + w_safe_preferred * j_safe_pref
                + w_approach_rebound * j_approach_rebound
                + w_goal_clearance * j_goal_clear
                + w_goal_clearance_target * j_goal_target
                + w_approach_clearance * j_approach_clear
                + w_approach_collision * j_approach_col
                + w_via_dev * j_via_dev
                + w_yaw_dev * j_yaw_dev
                + w_yaw_monotonic * j_yaw_mono
                + w_yaw_schedule * j_yaw_sched
                + w_goal_approach_normal * j_goal_normal
            )
        return (
            j,
            j_len,
            j_curv,
            j_safe,
            j_yaw,
            j_safe_pref,
            j_approach_rebound,
            j_goal_clear,
            j_goal_target,
            j_approach_clear,
            j_approach_col,
            j_via_dev,
            j_yaw_dev,
            j_yaw_mono,
            j_yaw_sched,
            j_goal_normal,
            yaw_samples_deg,
            yaw_quats,
        )

    accept_straight_line = True
    if options is not None and "accept_straight_line_if_feasible" in options:
        accept_straight_line = bool(options["accept_straight_line_if_feasible"])

    if accept_straight_line:
        (
            j0,
            j_len0,
            j_curv0,
            j_safe0,
            j_yaw0,
            j_safe_pref0,
            j_approach_rebound0,
            j_goal_clear0,
            j_goal_target0,
            j_approach_clear0,
            j_approach_col0,
            j_via_dev0,
            j_yaw_dev0,
            j_yaw_mono0,
            j_yaw_sched0,
            j_goal_normal0,
            yaw_samples0,
            yaw_quats0,
        ) = objective_single(x0)
        if j_safe0 <= 1e-12 and j_approach_col0 <= 1e-12:
            vias_opt = x0[:x0_pos.size].reshape(n_vias, 3)
            us_opt = us
            yaw_ctrl_opt = _decode_yaw_controls(x0)
            if combined_4d:
                W4_opt = np.hstack(
                    [
                        np.vstack([start, vias_opt, goal]),
                        yaw_ctrl_opt.reshape(-1, 1),
                    ]
                )
                S4_opt = build_cubic_bspline(W4_opt)
                Q4_opt = S4_opt(us_opt)
                P_opt = Q4_opt[:, :3]

                def S_opt(uq):
                    q = np.asarray(S4_opt(uq), dtype=float)
                    if q.ndim == 1:
                        return q[:3].reshape(1, 3)
                    return q[:, :3]

                def S_yaw_opt(uq):
                    q = np.asarray(S4_opt(uq), dtype=float)
                    if q.ndim == 1:
                        return np.array([q[3]], dtype=float)
                    return q[:, 3]
            else:
                W_opt = np.vstack([start, vias_opt, goal])
                S_opt = build_cubic_bspline(W_opt)
                P_opt = S_opt(us_opt)
                S_yaw_opt = build_scalar_bspline(yaw_ctrl_opt)

            info = {
                "success": True,
                "message": "Straight-line initialization feasible; skipped optimizer.",
                "fun": float(j0),
                "length": float(j_len0),
                "curvature_cost": float(j_curv0),
                "yaw_smoothness_cost": float(j_yaw0),
                "safety_cost": float(j_safe0),
                "preferred_safety_cost": float(j_safe_pref0),
                "approach_rebound_cost": float(j_approach_rebound0),
                "goal_clearance_cost": float(j_goal_clear0),
                "goal_clearance_target_cost": float(j_goal_target0),
                "approach_clearance_cost": float(j_approach_clear0),
                "approach_collision_cost": float(j_approach_col0),
                "via_deviation_cost": float(j_via_dev0),
                "yaw_deviation_cost": float(j_yaw_dev0),
                "yaw_monotonic_cost": float(j_yaw_mono0),
                "yaw_schedule_cost": float(j_yaw_sched0),
                "goal_approach_normal_cost": float(j_goal_normal0),
                "min_clearance": float(np.min(_path_distances(
                    scene,
                    P_opt,
                    moving_block_size=moving_block_size,
                    moving_block_quat=moving_block_quat,
                    moving_block_quats=yaw_quats0,
                    moving_proxies=moving_proxies,
                    ignore_ids=collision_ignore_ids,
                ))),
                "mean_clearance": float(np.mean(_path_distances(
                    scene,
                    P_opt,
                    moving_block_size=moving_block_size,
                    moving_block_quat=moving_block_quat,
                    moving_block_quats=yaw_quats0,
                    moving_proxies=moving_proxies,
                    ignore_ids=collision_ignore_ids,
                ))),
                "turn_angle_mean_deg": mean_turn_angle_deg(P_opt),
                "yaw_start_deg": float(start_yaw_deg),
                "yaw_goal_deg": float(goal_yaw_deg),
                "yaw_ctrl_deg": yaw_ctrl_opt.copy(),
                "yaw_samples_deg": yaw_samples0.copy(),
                "yaw_fn": S_yaw_opt,
                "combined_4d": bool(combined_4d),
                "solver_method": method,
                "required_clearance": required_clearance,
                "preferred_clearance": preferred_clearance,
                "goal_clearance_target": goal_clearance_target,
                "approach_only_clearance": approach_only_clearance,
                "contact_window_fraction": float(contact_window_fraction),
                "goal_approach_window_fraction": float(goal_approach_window_fraction),
                "goal_approach_normals": None if goal_approach_normals is None else np.asarray(goal_approach_normals, dtype=float).copy(),
                "yaw_goal_reach_u": float(yaw_goal_reach_u),
                "cost_mode": cost_mode,
                "collision_model": "multi_box" if moving_proxies else ("box" if moving_block_size is not None else "point"),
                "nit": 0,
                "xyz_ctrl_pts": np.vstack([start, vias_opt, goal]).copy(),
                "xyz_spline_degree": 3,
            }
            return S_opt, vias_opt, info

    opt = _solve_optimizer(objective_single, x0=x0, sigma0=sigma0, method=method, options=options)
    x_opt = opt["x"]
    vias_opt = x_opt[:x0_pos.size].reshape(n_vias, 3)
    us_opt = us
    yaw_ctrl_opt = _decode_yaw_controls(x_opt)
    if combined_4d:
        W4_opt = np.hstack(
            [
                np.vstack([start, vias_opt, goal]),
                yaw_ctrl_opt.reshape(-1, 1),
            ]
        )
        S4_opt = build_cubic_bspline(W4_opt)
        Q4_opt = S4_opt(us_opt)
        P_opt = Q4_opt[:, :3]
        yaw_samples_opt = Q4_opt[:, 3]
        def S_opt(uq):
            q = np.asarray(S4_opt(uq), dtype=float)
            if q.ndim == 1:
                return q[:3].reshape(1, 3)
            return q[:, :3]

        def S_yaw_opt(uq):
            q = np.asarray(S4_opt(uq), dtype=float)
            if q.ndim == 1:
                return np.array([q[3]], dtype=float)
            return q[:, 3]
    else:
        W_opt = np.vstack([start, vias_opt, goal])
        S_opt = build_cubic_bspline(W_opt)
        P_opt = S_opt(us_opt)
        S_yaw_opt = build_scalar_bspline(yaw_ctrl_opt)
        yaw_samples_opt = S_yaw_opt(us_opt)
    yaw_quats_opt = np.array([yaw_deg_to_quat(y) for y in yaw_samples_opt], dtype=float)
    (
        _,
        j_len_opt,
        j_curv_opt,
        j_safe_opt,
        j_yaw_opt,
        j_safe_pref_opt,
        j_approach_rebound_opt,
        j_goal_clear_opt,
        j_goal_target_opt,
        j_approach_clear_opt,
        j_approach_col_opt,
        j_via_dev_opt,
        j_yaw_dev_opt,
        j_yaw_mono_opt,
        j_yaw_sched_opt,
        j_goal_normal_opt,
        _,
        _,
    ) = objective_single(x_opt)
    d_opt = _path_distances(
        scene,
        P_opt,
        moving_block_size=moving_block_size,
        moving_block_quat=moving_block_quat,
        moving_block_quats=yaw_quats_opt,
        moving_proxies=moving_proxies,
        ignore_ids=collision_ignore_ids,
    )

    info = {
        "success": opt["success"],
        "message": opt["message"],
        "fun": opt["fun"],
        "length": j_len_opt,
        "curvature_cost": j_curv_opt,
        "yaw_smoothness_cost": j_yaw_opt,
        "safety_cost": j_safe_opt,
        "preferred_safety_cost": j_safe_pref_opt,
        "approach_rebound_cost": j_approach_rebound_opt,
        "goal_clearance_cost": j_goal_clear_opt,
        "goal_clearance_target_cost": j_goal_target_opt,
        "approach_clearance_cost": j_approach_clear_opt,
        "approach_collision_cost": j_approach_col_opt,
        "via_deviation_cost": j_via_dev_opt,
        "yaw_deviation_cost": j_yaw_dev_opt,
        "yaw_monotonic_cost": j_yaw_mono_opt,
        "yaw_schedule_cost": j_yaw_sched_opt,
        "goal_approach_normal_cost": j_goal_normal_opt,
        "min_clearance": float(np.min(d_opt)),
        "mean_clearance": float(np.mean(d_opt)),
        "turn_angle_mean_deg": mean_turn_angle_deg(P_opt),
        "yaw_start_deg": float(start_yaw_deg),
        "yaw_goal_deg": float(goal_yaw_deg),
        "yaw_ctrl_deg": yaw_ctrl_opt.copy(),
        "yaw_samples_deg": yaw_samples_opt.copy(),
        "yaw_fn": S_yaw_opt,
        "combined_4d": bool(combined_4d),
        "solver_method": method,
        "required_clearance": required_clearance,
        "preferred_clearance": preferred_clearance,
        "goal_clearance_target": goal_clearance_target,
        "approach_only_clearance": approach_only_clearance,
        "contact_window_fraction": float(contact_window_fraction),
        "goal_approach_window_fraction": float(goal_approach_window_fraction),
        "goal_approach_normals": None if goal_approach_normals is None else np.asarray(goal_approach_normals, dtype=float).copy(),
        "yaw_goal_reach_u": float(yaw_goal_reach_u),
        "cost_mode": cost_mode,
        "collision_model": "multi_box" if moving_proxies else ("box" if moving_block_size is not None else "point"),
        "nit": opt["nit"],
        "xyz_ctrl_pts": np.vstack([start, vias_opt, goal]).copy(),
        "xyz_spline_degree": 3,
    }
    return S_opt, vias_opt, info


def optimize_bspline_with_vias(
    scene,
    start: np.ndarray,
    via: np.ndarray,
    goal: np.ndarray,
    n_additional_vias: int = 2,
    tool_half_extents: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    safety_margin: float = 0.01,
    n_samples_curve: int = 121,
    collision_check_subsample: int = 1,
    w_len: float = 1.0,
    w_curv: float = 0.1,
    w_safe: float = 50.0,
    init_offset_scale: float = 1.0,
    method: str = "CEM",
    options: Optional[Dict] = None,
) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, Dict]:
    """
    Optimize N additional via points so the cubic B-spline through
    [start, via, additional_vias..., goal] has low length, low curvature,
    and satisfies a safety margin using the scene SDF.

    Returns:
        S: callable S(u in [0,1])->(m,3) samples
        vias_opt: optimized additional via points (N,3)
        info: dict with objective breakdown
    """
    start = np.asarray(start, float).reshape(3)
    via = np.asarray(via, float).reshape(3)
    goal = np.asarray(goal, float).reshape(3)

    if n_additional_vias < 1:
        raise ValueError("n_additional_vias must be >= 1")
    if collision_check_subsample < 1:
        raise ValueError("collision_check_subsample must be >= 1")

    moving_block_size = None
    if any(float(v) > 0.0 for v in tool_half_extents):
        hx, hy, hz = map(float, tool_half_extents)
        moving_block_size = (2.0 * hx, 2.0 * hy, 2.0 * hz)
    required_clearance = float(safety_margin)

    via_init = _default_via_initialization(via, goal, n_additional_vias)
    x0 = via_init.reshape(-1)

    sigma_base = np.linalg.norm(goal - via) * init_offset_scale / max(n_additional_vias, 1)
    sigma0 = np.full_like(x0, max(0.05, sigma_base), dtype=float)

    def objective_single(x: np.ndarray):
        vias_add = x.reshape(n_additional_vias, 3)
        W = np.vstack([start, via, vias_add, goal])
        S = build_cubic_bspline(W)
        P, _ = sample_curve(S, n=n_samples_curve)
        if collision_check_subsample > 1:
            P_safe = P[::collision_check_subsample]
            if not np.allclose(P_safe[-1], P[-1]):
                P_safe = np.vstack([P_safe, P[-1]])
        else:
            P_safe = P

        j_len = path_length(P)
        j_curv = curvature_cost(P)
        j_safe = safety_cost(
            scene,
            P_safe,
            required_clearance=required_clearance,
            moving_block_size=moving_block_size,
        )

        j = w_len * j_len + w_curv * j_curv + w_safe * j_safe
        return j, j_len, j_curv, j_safe

    opt = _solve_optimizer(objective_single, x0=x0, sigma0=sigma0, method=method, options=options)
    x_opt = opt["x"]

    vias_opt = x_opt.reshape(n_additional_vias, 3)
    W_opt = np.vstack([start, via, vias_opt, goal])
    S_opt = build_cubic_bspline(W_opt)
    P_opt, _ = sample_curve(S_opt, n=n_samples_curve)
    _, j_len_opt, j_curv_opt, j_safe_opt = objective_single(x_opt)
    d_opt = _path_distances(
        scene,
        P_opt,
        moving_block_size=moving_block_size,
    )

    info = {
        "success": opt["success"],
        "message": opt["message"],
        "fun": opt["fun"],
        "length": j_len_opt,
        "curvature_cost": j_curv_opt,
        "safety_cost": j_safe_opt,
        "min_clearance": float(np.min(d_opt)),
        "mean_clearance": float(np.mean(d_opt)),
        "turn_angle_mean_deg": mean_turn_angle_deg(P_opt),
        "required_clearance": required_clearance,
        "collision_model": "box" if moving_block_size is not None else "point",
        "nit": opt["nit"],
    }
    return S_opt, vias_opt, info


def optimize_bspline_two_vias(
    scene,
    start: np.ndarray,
    via: np.ndarray,
    goal: np.ndarray,
    tool_half_extents: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    safety_margin: float = 0.01,
    n_samples_curve: int = 121,
    w_len: float = 1.0,
    w_curv: float = 0.1,
    w_safe: float = 50.0,
    init_offset_scale: float = 0.3,
    method: str = "Nelder-Mead",
    options: Dict = None,
) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray, Dict]:
    """Backward-compatible wrapper for two additional via points."""
    S, vias, info = optimize_bspline_with_vias(
        scene=scene,
        start=start,
        via=via,
        goal=goal,
        n_additional_vias=2,
        tool_half_extents=tool_half_extents,
        safety_margin=safety_margin,
        n_samples_curve=n_samples_curve,
        w_len=w_len,
        w_curv=w_curv,
        w_safe=w_safe,
        init_offset_scale=init_offset_scale,
        method=method,
        options=options,
    )
    return S, vias[0], vias[1], info
