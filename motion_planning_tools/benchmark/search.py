from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List
import os
import multiprocessing as mp
import tempfile

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage
from optuna.trial import TrialState

from .config import config_options_from_trial_params, suggest_strategy_config, trial_params_from_config
from .metrics import aggregate_numeric
from .planners import run_single


def _sqlite_url_to_path(storage_url: str) -> Path | None:
    prefix = "sqlite:///"
    if not storage_url.startswith(prefix):
        return None
    raw = storage_url[len(prefix):]
    if not raw:
        return None
    return Path(raw)


def _normalize_n_jobs(n_jobs: int) -> int:
    nj = int(n_jobs)
    if nj <= 0:
        return max(1, int(os.cpu_count() or 1))
    return max(1, nj)


def _make_journal_storage(journal_path: str):
    try:
        from optuna.storages import JournalStorage  # type: ignore
        from optuna.storages.journal import JournalFileBackend  # type: ignore

        return JournalStorage(JournalFileBackend(journal_path))
    except Exception:
        from optuna.storages import JournalFileStorage, JournalStorage  # type: ignore

        return JournalStorage(JournalFileStorage(journal_path))


def _build_row(method: str, trial: optuna.Trial, cfg: Dict[str, Any], opts: Dict[str, Any], res: Dict[str, Any]) -> Dict[str, Any]:
    trial.set_user_attr("method", str(method))
    trial.set_user_attr("config", dict(cfg))
    trial.set_user_attr("options", dict(opts))
    trial.set_user_attr("mean_score", float(res["mean_score"]))
    trial.set_user_attr("std_score", float(res["std_score"]))
    trial.set_user_attr("success_rate", float(res["success_rate"]))
    return {
        "trial": int(trial.number + 1),
        "method": method,
        "config": cfg,
        "options": opts,
        "mean_score": float(res["mean_score"]),
        "std_score": float(res["std_score"]),
        "success_rate": float(res["success_rate"]),
    }


def _run_optuna_worker(
    wm: Any,
    train_scenarios: List[str],
    method: str,
    seed: int,
    study_name: str,
    storage_url: str,
    journal_path: str | None,
    worker_trials: int,
) -> None:
    if journal_path:
        storage = _make_journal_storage(journal_path)
    elif storage_url.startswith("sqlite"):
        storage = RDBStorage(
            url=storage_url,
            engine_kwargs={"connect_args": {"timeout": 120}},
        )
    else:
        storage = storage_url

    def objective(trial: optuna.Trial) -> float:
        cfg, opts = suggest_strategy_config(method=method, trial=trial, seed=seed)
        res = evaluate_config(wm, train_scenarios, method, cfg, opts)
        _build_row(method, trial, cfg, opts, res)
        return float(res["mean_score"])

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed),
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=int(worker_trials), show_progress_bar=False, n_jobs=1)


def evaluate_config(
    wm: Any,
    scenario_names: List[str],
    method: str,
    config: Dict[str, Any],
    options: Dict[str, Any],
) -> Dict[str, Any]:
    per_scenario: List[Dict[str, Any]] = []
    for name in scenario_names:
        try:
            per_scenario.append(run_single(wm, name, method, config, options))
        except Exception as exc:
            per_scenario.append(
                {
                    "scenario": name,
                    "runtime_s": 0.0,
                    "score": 1e9,
                    "success": False,
                    "fun": 1e9,
                    "length": 0.0,
                    "min_clearance": -1.0,
                    "nit": 0,
                    "message": f"Exception: {exc}",
                }
            )
    success_rate = float(np.mean([1.0 if r["success"] else 0.0 for r in per_scenario]))
    agg = aggregate_numeric(per_scenario)
    return {
        "mean_score": float(agg["score"]["mean"]),
        "std_score": float(agg["score"]["std"]),
        "success_rate": success_rate,
        "metrics": agg,
        "per_scenario": per_scenario,
    }


def hyperopt(
    wm: Any,
    train_scenarios: List[str],
    method: str,
    n_trials: int,
    seed: int,
    warm_start: Dict[str, Any] | None = None,
    study_name: str | None = None,
    storage_url: str | None = None,
    n_jobs: int = 1,
) -> Dict[str, Any]:
    trials: List[Dict[str, Any]] = []
    completed: Dict[int, Dict[str, Any]] = {}
    effective_n_jobs = _normalize_n_jobs(n_jobs)
    journal_path_str: str | None = None

    storage: str | RDBStorage | None = storage_url
    if storage_url and str(storage_url).startswith("sqlite"):
        # SQLite allows only one writer at a time. Parallel Optuna workers can
        # hit "database is locked" under write contention.
        if effective_n_jobs > 1:
            sqlite_path = _sqlite_url_to_path(storage_url)
            if sqlite_path is not None:
                journal_path = sqlite_path.with_name(f"{sqlite_path.name}.journal")
                journal_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    storage = _make_journal_storage(str(journal_path))
                    journal_path_str = str(journal_path)
                    warnings.warn(
                        f"Using persistent JournalStorage at '{journal_path}' for "
                        "parallel Optuna with SQLite URL input.",
                        RuntimeWarning,
                    )
                except Exception:
                    warnings.warn(
                        "Optuna journal storage backend is unavailable in this environment. "
                        "Falling back to single-worker SQLite to avoid lock errors.",
                        RuntimeWarning,
                    )
                    storage = RDBStorage(
                        url=storage_url,
                        engine_kwargs={"connect_args": {"timeout": 120}},
                    )
                    effective_n_jobs = 1
            else:
                warnings.warn(
                    "Could not derive a filesystem path from SQLite URL. "
                    "Falling back to single-worker SQLite to avoid lock errors.",
                    RuntimeWarning,
                )
                storage = RDBStorage(
                    url=storage_url,
                    engine_kwargs={"connect_args": {"timeout": 120}},
                )
                effective_n_jobs = 1
        else:
            storage = RDBStorage(
                url=storage_url,
                engine_kwargs={"connect_args": {"timeout": 120}},
            )
    elif not storage_url and effective_n_jobs > 1 and hasattr(os, "fork"):
        tmp_journal = Path(tempfile.gettempdir()) / f"optuna_benchmark_{os.getpid()}.journal"
        storage = _make_journal_storage(str(tmp_journal))
        journal_path_str = str(tmp_journal)
        warnings.warn(
            f"No --optuna-storage provided; using temporary journal storage at '{tmp_journal}' "
            "to enable multi-process parallel trials.",
            RuntimeWarning,
        )

    def objective(trial: optuna.Trial) -> float:
        cfg, opts = suggest_strategy_config(method=method, trial=trial, seed=seed)
        res = evaluate_config(wm, train_scenarios, method, cfg, opts)
        row = _build_row(method, trial, cfg, opts, res)
        completed[trial.number] = row
        return float(res["mean_score"])

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed),
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(study_name and storage_url),
    )
    if warm_start is not None:
        ws_cfg = dict(warm_start.get("config", {}))
        ws_opts = dict(warm_start.get("options", {}))
        if ws_cfg and ws_opts:
            try:
                study.enqueue_trial(trial_params_from_config(method, ws_cfg, ws_opts))
            except Exception:
                pass

    use_process_workers = (
        effective_n_jobs > 1
        and (bool(storage_url) or bool(journal_path_str))
        and bool(study_name)
        and hasattr(os, "fork")
    )
    if use_process_workers:
        ctx = mp.get_context("fork")
        total_trials = int(n_trials)
        n_workers = min(effective_n_jobs, total_trials)
        base = total_trials // n_workers
        rem = total_trials % n_workers
        chunks = [base + (1 if i < rem else 0) for i in range(n_workers)]
        procs = []
        for chunk in chunks:
            if chunk <= 0:
                continue
            p = ctx.Process(
                target=_run_optuna_worker,
                args=(
                    wm,
                    train_scenarios,
                    method,
                    seed,
                    str(study_name),
                    str(storage_url),
                    journal_path_str,
                    int(chunk),
                ),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Optuna worker process exited with code {p.exitcode}")
        # Reload after worker updates.
        study = optuna.load_study(study_name=str(study_name), storage=storage)
    else:
        study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False, n_jobs=effective_n_jobs)

    for t in sorted(study.trials, key=lambda x: x.number):
        if t.state != TrialState.COMPLETE:
            continue
        if t.number in completed:
            trials.append(completed[t.number])
            continue
        u = dict(t.user_attrs)
        if "config" in u and "options" in u:
            trials.append(
                {
                    "trial": int(t.number + 1),
                    "method": str(u.get("method", method)),
                    "config": dict(u["config"]),
                    "options": dict(u["options"]),
                    "mean_score": float(u.get("mean_score", t.value if t.value is not None else 1e9)),
                    "std_score": float(u.get("std_score", 0.0)),
                    "success_rate": float(u.get("success_rate", 0.0)),
                }
            )

    if not trials:
        raise RuntimeError("Optuna completed with no valid trials.")

    best_trial = study.best_trial
    if best_trial.number in completed:
        best = completed[best_trial.number]
    else:
        u = dict(best_trial.user_attrs)
        if "config" in u and "options" in u:
            best = {
                "trial": int(best_trial.number + 1),
                "method": str(u.get("method", method)),
                "config": dict(u["config"]),
                "options": dict(u["options"]),
                "mean_score": float(u.get("mean_score", best_trial.value if best_trial.value is not None else 1e9)),
                "std_score": float(u.get("std_score", 0.0)),
                "success_rate": float(u.get("success_rate", 0.0)),
            }
        else:
            cfg, opts = config_options_from_trial_params(
                method=method,
                params=dict(best_trial.params),
                seed=seed,
                trial_number=int(best_trial.number),
            )
            reevaluated = evaluate_config(wm, train_scenarios, method, cfg, opts)
            best = {
                "trial": int(best_trial.number + 1),
                "method": str(method),
                "config": cfg,
                "options": opts,
                "mean_score": float(reevaluated["mean_score"]),
                "std_score": float(reevaluated["std_score"]),
                "success_rate": float(reevaluated["success_rate"]),
            }
    return {"trials": trials, "best": best}


def benchmark_best(
    wm: Any,
    scenario_names: List[str],
    best_entry: Dict[str, Any],
) -> Dict[str, Any]:
    method = str(best_entry["method"])
    cfg = dict(best_entry["config"])
    opts = dict(best_entry["options"])
    eval_res = evaluate_config(wm, scenario_names, method, cfg, opts)
    return {
        "method": method,
        "config": cfg,
        "options": opts,
        "aggregate": {
            "mean_score": eval_res["mean_score"],
            "std_score": eval_res["std_score"],
            "success_rate": eval_res["success_rate"],
            "metrics": eval_res["metrics"],
        },
        "per_scenario": eval_res["per_scenario"],
    }
