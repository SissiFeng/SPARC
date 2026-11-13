from __future__ import annotations
import importlib
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from .ensemble import SparseEvidenceEnsemble
from .models import build_library, tune_library
from .preprocessing import Preprocessor
from .figure_io import check_sampling, compact_diagnostics, metrics_frame, write_json


class TunedBaseline:
    def __init__(self, name, template, random_state):
        self.name = name
        self.template = template
        self.random_state = random_state

    def fit(self, X, y):
        tuned, chosen = tune_library(
            [(self.name, self.template)],
            X,
            y,
            transformer_factory=Preprocessor,
            random_state=self.random_state,
        )
        self.hyperparameters_ = chosen[self.name]
        self.model_ = make_pipeline(Preprocessor(), tuned[0][1]).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def _benchmark_once(X, y, ids, out, seed, methods, family_priors, factories, adapters, n_jobs, tag, weight_strategy, global_scale):
    library = dict(build_library(seed))
    rows, records = ([], [])
    for fold, (tr, te) in enumerate(LeaveOneOut().split(X)):
        for name in methods:
            started = time.perf_counter()
            if name == "SPARC":
                model = SparseEvidenceEnsemble(
                    random_state=seed, family_priors=family_priors, n_jobs=n_jobs,
                    weight_strategy=weight_strategy, global_scale=global_scale
                )
            elif name in library:
                model = TunedBaseline(name, clone(library[name]), seed)
            else:
                model = factories[name](
                    random_state=seed, configuration=adapters[name]["configuration"]
                )
            model.fit(X[tr], y[tr])
            diagnostic = getattr(model, "diagnostics_", {})
            check_sampling(diagnostic)
            if name == "SPARC":
                pred, sd = model.predict_with_uncertainty(X[te])
            else:
                pred, sd = (np.asarray(model.predict(X[te])), np.full(len(te), np.nan))
            if np.asarray(pred).shape != (len(te),) or not np.isfinite(pred).all():
                raise ValueError(f"invalid predictions from {name}")
            elapsed = time.perf_counter() - started
            rows.extend(
                (
                    {
                        "seed": seed,
                        "method": name,
                        "fold": fold,
                        "row_index": int(i),
                        "row_id": ids[i],
                        "y_true": float(y[i]),
                        "y_pred": float(p),
                        "epistemic_sd": float(u),
                        "residual_scale": float(getattr(model, "residual_scale_", np.nan)),
                    }
                    for i, p, u in zip(te, pred, sd)
                )
            )
            records.append(
                {
                    "seed": seed,
                    "method": name,
                    "fold": fold,
                    "train_indices": tr,
                    "test_indices": te,
                    "seconds": elapsed,
                    "weights": getattr(model, "weights_", None),
                    "selection_frequencies": getattr(
                        model, "selection_frequencies_", None
                    ),
                    "hyperparameters": getattr(model, "hyperparameters_", None),
                    "diagnostics": compact_diagnostics(diagnostic),
                    "comparison_records": getattr(model, "records_", None),
                    "selection": getattr(model, "selection_", None),
                }
            )
        pd.DataFrame(rows).to_csv(out / f"predictions_{tag}.partial.csv", index=False)
        write_json(out / f"folds_{tag}.json", records)
        print(f"seed {seed} outer LOO {fold + 1}/{len(y)}", flush=True)
    (out / f"predictions_{tag}.partial.csv").unlink()
    return rows


def run_benchmark(
    X,
    y,
    ids,
    out,
    *,
    seed,
    family_priors=None,
    methods=None,
    adapters=None,
    repeat_seeds=None,
    n_jobs=1,
    weight_strategy="simplex",
    global_scale=1.0,
):
    library = dict(build_library(seed))
    methods = methods or ["SPARC"]
    adapters = adapters or {}
    seeds = [int(seed)] if repeat_seeds is None else [int(s) for s in repeat_seeds]
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("repeat_seeds must be distinct nonnegative integers")
    if len(set(methods)) != len(methods) or set(methods) - {
        "SPARC",
        *library,
        *adapters,
    }:
        raise ValueError(
            "methods must be unique SPARC, library names or configured adapters"
        )
    factories = {}
    for name, config in adapters.items():
        if name in {"SPARC", *library}:
            raise ValueError("adapter name collides with a built-in model")
        if not config.get("configuration") or not config.get("implementation"):
            raise ValueError(
                "comparison adapters require implementation and archived configuration"
            )
        module, attribute = config["implementation"].split(":")
        factories[name] = getattr(importlib.import_module(module), attribute)
    started = time.perf_counter()
    rows = []
    for current in seeds:
        rows.extend(
            _benchmark_once(
                X, y, ids, out, current, methods, family_priors, factories, adapters, n_jobs, current, weight_strategy, global_scale
            )
        )
    result = pd.DataFrame(rows)
    result.to_csv(out / "predictions.csv", index=False)
    per_seed = pd.concat(
        [
            metrics_frame(group.drop(columns="seed")).assign(seed=current)
            for current, group in result.groupby("seed", sort=False)
        ],
        ignore_index=True,
    )
    per_seed.to_csv(out / "metrics_by_seed.csv", index=False)
    summary = per_seed.groupby("method", sort=False).agg(
        n_seeds=("seed", "nunique"),
        r2_mean=("r2", "mean"),
        r2_sd=("r2", lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else np.nan),
        mae_mean=("mae", "mean"),
        rmse_mean=("rmse", "mean"),
    )
    summary.to_csv(out / "metrics.csv")
    write_json(
        out / "timing.json",
        {"total_seconds": time.perf_counter() - started, "seeds": seeds, "n_jobs": int(n_jobs)},
    )


def run_feature_strategies(X, y, out, *, seed, n_rounds):
    from .fusion import STRATEGY_NAMES, require_strategy_dependencies, run_strategy_bandit

    if n_rounds < len(STRATEGY_NAMES):
        raise ValueError("n_rounds must evaluate all eight strategies")
    if len(y) < 7:
        raise ValueError("five-fold scoring inside outer LOO needs at least seven rows")
    require_strategy_dependencies(STRATEGY_NAMES)
    rows, pulls_log, failures_log = ([], [], [])
    for fold, (tr, te) in enumerate(LeaveOneOut().split(X)):
        bandit, pulls, failures = run_strategy_bandit(
            X[tr], y[tr], n_rounds, cv_splits=5, delta=0.1, random_state=seed + 1000 * fold
        )
        chosen = bandit.best_arm()
        raw = bandit.raw_mean_rewards
        normalized = bandit.mean_rewards
        disabled = bandit.disabled
        rows.extend(
            (
                {
                    "fold": fold,
                    "test_index": int(te[0]),
                    "strategy": name,
                    "raw_r2_reward": float(raw[j]) if np.isfinite(raw[j]) else None,
                    "normalized_reward": float(normalized[j])
                    if np.isfinite(normalized[j])
                    else None,
                    "policy_pulls": int(bandit.pulls[j]),
                    "failed": bool(disabled[j]),
                    "selected_for_evidence_batch": j == chosen,
                }
                for j, name in enumerate(STRATEGY_NAMES)
            )
        )
        pulls_log.extend({"fold": fold, **entry} for entry in pulls)
        failures_log.extend({"fold": fold, "strategy": k, "error": v} for k, v in failures.items())
        pd.DataFrame(rows).to_csv(out / "strategy_folds.csv", index=False)
        print(f"feature strategies {fold + 1}/{len(y)}", flush=True)
    result = pd.DataFrame(rows)
    pd.DataFrame(pulls_log).to_csv(out / "strategy_pulls.csv", index=False)
    write_json(out / "strategy_failures.json", failures_log)
    summary = result.groupby("strategy", sort=False).agg(
        selection_count=("selected_for_evidence_batch", "sum"),
        policy_pull_count=("policy_pulls", "sum"),
        failed_folds=("failed", "sum"),
        raw_r2_reward=("raw_r2_reward", "mean"),
    )
    summary.to_csv(out / "strategy_summary.csv")
