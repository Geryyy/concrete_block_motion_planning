from __future__ import annotations

from typing import Any, Dict, Tuple

BASE_CONFIG: Dict[str, Any] = {
    "cost_mode": "simple",
    "n_vias": 2,
    "safety_margin": 0.0,
    "preferred_safety_margin": 0.02,
    "relax_preferred_final_fraction": 0.25,
    "approach_only_clearance": 0.015,
    "contact_window_fraction": 0.08,
    "n_yaw_vias": 2,
    "combined_4d": True,
    "approach_fraction": 0.25,
    "w_via_dev": 0.06,
    "w_yaw_monotonic": 80.0,
    "yaw_goal_reach_u": 0.5,
    "goal_approach_window_fraction": 0.12,
    "init_offset_scale": 0.7,
    "goal_clearance_target": 0.0,
    "w_len": 5.0,
    "n_samples_curve": 101,
    "collision_check_subsample": 1,
    "w_curv": 0.12,
    "w_yaw_smooth": 0.008,
    "w_safe": 380.0,
    "w_safe_preferred": 24.0,
    "w_approach_rebound": 280.0,
    "w_goal_clearance": 35.0,
    "w_goal_clearance_target": 260.0,
    "w_approach_clearance": 420.0,
    "w_approach_collision": 1400.0,
    "w_yaw_dev": 0.05,
    "w_yaw_schedule": 55.0,
    "w_goal_approach_normal": 80.0,
}

VALID_METHODS = {
    "POWELL",
    "NELDER-MEAD",
    "NELDER_MEAD",
    "NELDERMEAD",
    "NELDER",
    "NM",
    "CEM",
    "VP-STO",
    "OMPL-RRT",
    "OMPL",
    "RRT",
}

SEED_OFFSETS = {
    "POWELL": 0,
    "NELDER-MEAD": 5_000,
    "NELDER_MEAD": 5_000,
    "NELDERMEAD": 5_000,
    "NELDER": 5_000,
    "NM": 5_000,
    "CEM": 10_000,
    "VP-STO": 20_000,
    "OMPL-RRT": 30_000,
    "OMPL": 30_000,
    "RRT": 30_000,
}


def suggest_strategy_config(method: str, trial: Any, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cfg = dict(BASE_CONFIG)
    cfg["init_offset_scale"] = float(trial.suggest_categorical("cfg_init_offset_scale", [0.5, 0.7, 1.0]))
    cfg["w_len"] = float(trial.suggest_categorical("cfg_w_len", [3.5, 5.0, 6.0]))
    cfg["w_curv"] = float(trial.suggest_categorical("cfg_w_curv", [0.08, 0.12, 0.18]))
    cfg["w_safe"] = float(trial.suggest_categorical("cfg_w_safe", [260.0, 380.0, 520.0]))
    cfg["w_goal_approach_normal"] = float(trial.suggest_categorical("cfg_w_goal_approach_normal", [40.0, 80.0, 120.0]))
    cfg["w_approach_collision"] = float(trial.suggest_categorical("cfg_w_approach_collision", [1000.0, 1400.0, 1800.0]))

    mu = method.upper()
    if mu == "POWELL":
        options = {
            "maxiter": int(trial.suggest_categorical("powell_maxiter", [80, 140, 220])),
            "xtol": float(trial.suggest_categorical("powell_xtol", [3e-3, 1e-3])),
            "ftol": float(trial.suggest_categorical("powell_ftol", [3e-3, 1e-3])),
        }
    elif mu in {"NELDER-MEAD", "NELDER_MEAD", "NELDERMEAD", "NELDER", "NM"}:
        options = {
            "maxiter": int(trial.suggest_categorical("nelder_maxiter", [120, 220, 320])),
            "xatol": float(trial.suggest_categorical("nelder_xatol", [3e-3, 1e-3])),
            "fatol": float(trial.suggest_categorical("nelder_fatol", [3e-3, 1e-3])),
        }
    elif mu == "CEM":
        options = {
            "population_size": int(trial.suggest_categorical("cem_population_size", [48, 72, 96])),
            "elite_frac": float(trial.suggest_categorical("cem_elite_frac", [0.15, 0.2, 0.25])),
            "max_iter": int(trial.suggest_categorical("cem_max_iter", [50, 80, 110])),
            "min_sigma": float(trial.suggest_categorical("cem_min_sigma", [5e-4, 1e-3])),
            "init_scale": float(trial.suggest_categorical("cem_init_scale", [0.2, 0.4, 0.7])),
            "sample_method": "Gaussian",
            "seed": int(seed + trial.number),
        }
    elif mu == "VP-STO":
        options = {
            "pop_size": int(trial.suggest_categorical("vpsto_pop_size", [24, 32, 40])),
            "max_iter": int(trial.suggest_categorical("vpsto_max_iter", [120, 180, 260])),
            "N_via": int(trial.suggest_categorical("vpsto_N_via", [4, 6, 8])),
            "N_eval": int(cfg["n_samples_curve"]),
            "sigma_init": float(trial.suggest_categorical("vpsto_sigma_init", [0.25, 0.4, 0.6])),
        }
    elif mu in {"OMPL-RRT", "OMPL", "RRT"}:
        options = {
            "solve_time": float(trial.suggest_categorical("rrt_solve_time", [0.4, 0.8, 1.2])),
            "range": float(trial.suggest_categorical("rrt_range", [0.15, 0.25, 0.35])),
            "interpolate_points": int(cfg["n_samples_curve"]),
            "bounds_margin": float(trial.suggest_categorical("rrt_bounds_margin", [0.35, 0.5, 0.7])),
        }
    else:
        raise ValueError(f"Unsupported method: {method}")

    return cfg, options


def trial_params_from_config(method: str, config: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a concrete (config, options) pair into Optuna trial params."""
    params: Dict[str, Any] = {
        "cfg_init_offset_scale": float(config.get("init_offset_scale", BASE_CONFIG["init_offset_scale"])),
        "cfg_w_len": float(config.get("w_len", BASE_CONFIG["w_len"])),
        "cfg_w_curv": float(config.get("w_curv", BASE_CONFIG["w_curv"])),
        "cfg_w_safe": float(config.get("w_safe", BASE_CONFIG["w_safe"])),
        "cfg_w_goal_approach_normal": float(config.get("w_goal_approach_normal", BASE_CONFIG["w_goal_approach_normal"])),
        "cfg_w_approach_collision": float(config.get("w_approach_collision", BASE_CONFIG["w_approach_collision"])),
    }

    mu = method.upper()
    if mu == "POWELL":
        params.update(
            {
                "powell_maxiter": int(options.get("maxiter", 140)),
                "powell_xtol": float(options.get("xtol", 1e-3)),
                "powell_ftol": float(options.get("ftol", 1e-3)),
            }
        )
    elif mu in {"NELDER-MEAD", "NELDER_MEAD", "NELDERMEAD", "NELDER", "NM"}:
        params.update(
            {
                "nelder_maxiter": int(options.get("maxiter", 220)),
                "nelder_xatol": float(options.get("xatol", 1e-3)),
                "nelder_fatol": float(options.get("fatol", 1e-3)),
            }
        )
    elif mu == "CEM":
        params.update(
            {
                "cem_population_size": int(options.get("population_size", 72)),
                "cem_elite_frac": float(options.get("elite_frac", 0.2)),
                "cem_max_iter": int(options.get("max_iter", 80)),
                "cem_min_sigma": float(options.get("min_sigma", 1e-3)),
                "cem_init_scale": float(options.get("init_scale", 0.4)),
            }
        )
    elif mu == "VP-STO":
        params.update(
            {
                "vpsto_pop_size": int(options.get("pop_size", 32)),
                "vpsto_max_iter": int(options.get("max_iter", 180)),
                "vpsto_N_via": int(options.get("N_via", 6)),
                "vpsto_sigma_init": float(options.get("sigma_init", 0.4)),
            }
        )
    elif mu in {"OMPL-RRT", "OMPL", "RRT"}:
        params.update(
            {
                "rrt_solve_time": float(options.get("solve_time", 0.8)),
                "rrt_range": float(options.get("range", 0.25)),
                "rrt_bounds_margin": float(options.get("bounds_margin", 0.5)),
            }
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    return params


def config_options_from_trial_params(
    method: str,
    params: Dict[str, Any],
    seed: int,
    trial_number: int = 0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Rebuild (config, options) from frozen trial params (for resumed studies)."""
    cfg = dict(BASE_CONFIG)
    cfg["init_offset_scale"] = float(params.get("cfg_init_offset_scale", BASE_CONFIG["init_offset_scale"]))
    cfg["w_len"] = float(params.get("cfg_w_len", BASE_CONFIG["w_len"]))
    cfg["w_curv"] = float(params.get("cfg_w_curv", BASE_CONFIG["w_curv"]))
    cfg["w_safe"] = float(params.get("cfg_w_safe", BASE_CONFIG["w_safe"]))
    cfg["w_goal_approach_normal"] = float(params.get("cfg_w_goal_approach_normal", BASE_CONFIG["w_goal_approach_normal"]))
    cfg["w_approach_collision"] = float(params.get("cfg_w_approach_collision", BASE_CONFIG["w_approach_collision"]))

    mu = method.upper()
    if mu == "POWELL":
        options = {
            "maxiter": int(params.get("powell_maxiter", 140)),
            "xtol": float(params.get("powell_xtol", 1e-3)),
            "ftol": float(params.get("powell_ftol", 1e-3)),
        }
    elif mu in {"NELDER-MEAD", "NELDER_MEAD", "NELDERMEAD", "NELDER", "NM"}:
        options = {
            "maxiter": int(params.get("nelder_maxiter", 220)),
            "xatol": float(params.get("nelder_xatol", 1e-3)),
            "fatol": float(params.get("nelder_fatol", 1e-3)),
        }
    elif mu == "CEM":
        options = {
            "population_size": int(params.get("cem_population_size", 72)),
            "elite_frac": float(params.get("cem_elite_frac", 0.2)),
            "max_iter": int(params.get("cem_max_iter", 80)),
            "min_sigma": float(params.get("cem_min_sigma", 1e-3)),
            "init_scale": float(params.get("cem_init_scale", 0.4)),
            "sample_method": "Gaussian",
            "seed": int(seed + trial_number),
        }
    elif mu == "VP-STO":
        options = {
            "pop_size": int(params.get("vpsto_pop_size", 32)),
            "max_iter": int(params.get("vpsto_max_iter", 180)),
            "N_via": int(params.get("vpsto_N_via", 6)),
            "N_eval": int(cfg["n_samples_curve"]),
            "sigma_init": float(params.get("vpsto_sigma_init", 0.4)),
        }
    elif mu in {"OMPL-RRT", "OMPL", "RRT"}:
        options = {
            "solve_time": float(params.get("rrt_solve_time", 0.8)),
            "range": float(params.get("rrt_range", 0.25)),
            "interpolate_points": int(cfg["n_samples_curve"]),
            "bounds_margin": float(params.get("rrt_bounds_margin", 0.5)),
        }
    else:
        raise ValueError(f"Unsupported method: {method}")

    return cfg, options
