from __future__ import annotations
import importlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from .ensemble import SparseEvidenceEnsemble
from .models import build_library
from .preprocessing import Preprocessor
from .figure_io import check_sampling, compact_diagnostics, metrics_frame, write_json


def run_benchmark(
    X, y, ids, out, *, seed, family_priors=None, methods=None, adapters=None
):
    library = dict(build_library(seed))
    methods = methods or ["SPARC"]
    adapters = adapters or {}
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
    rows, records = ([], [])
    for fold, (tr, te) in enumerate(LeaveOneOut().split(X)):
        for name in methods:
            if name == "SPARC":
                model = SparseEvidenceEnsemble(
                    random_state=seed, family_priors=family_priors
                )
            elif name in library:
                model = make_pipeline(Preprocessor(), clone(library[name]))
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
            rows.extend(
                (
                    {
                        "method": name,
                        "fold": fold,
                        "row_index": int(i),
                        "row_id": ids[i],
                        "y_true": float(y[i]),
                        "y_pred": float(p),
                        "epistemic_sd": float(u),
                    }
                    for i, p, u in zip(te, pred, sd)
                )
            )
            records.append(
                {
                    "method": name,
                    "fold": fold,
                    "train_indices": tr,
                    "test_indices": te,
                    "weights": getattr(model, "weights_", None),
                    "selection_frequencies": getattr(
                        model, "selection_frequencies_", None
                    ),
                    "diagnostics": compact_diagnostics(diagnostic),
                    "comparison_records": getattr(model, "records_", None),
                    "selection": getattr(model, "selection_", None),
                }
            )
        pd.DataFrame(rows).to_csv(out / "predictions.partial.csv", index=False)
        write_json(out / "folds.json", records)
        print(f"outer LOO {fold + 1}/{len(y)}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(out / "predictions.csv", index=False)
    metrics_frame(result).to_csv(out / "metrics.csv", index=False)
    (out / "predictions.partial.csv").unlink()


def run_feature_strategies(X, y, out, *, seed, n_rounds):
    from .fusion import (
        STRATEGY_NAMES,
        _STRATEGIES,
        _strategy_cv_reward,
        require_strategy_dependencies,
        UCBBandit,
    )

    if n_rounds < len(STRATEGY_NAMES):
        raise ValueError("n_rounds must evaluate all eight strategies")
    if len(y) < 7:
        raise ValueError("five-fold scoring inside outer LOO needs at least seven rows")
    require_strategy_dependencies(STRATEGY_NAMES)
    rows = []
    for fold, (tr, te) in enumerate(LeaveOneOut().split(X)):
        raw = np.array(
            [
                _strategy_cv_reward(_STRATEGIES[s], X[tr], y[tr], 5, seed)
                for s in STRATEGY_NAMES
            ]
        )
        if not np.isfinite(raw).all():
            raise RuntimeError(
                "a feature strategy failed; no substituted rewards are allowed"
            )
        bandit = UCBBandit(len(raw), delta=0.1)
        for _ in range(n_rounds):
            arm = bandit.select()
            bandit.update(arm, raw[arm])
        chosen = int(np.argmax(bandit.mean_rewards))
        rows.extend(
            (
                {
                    "fold": fold,
                    "test_index": int(te[0]),
                    "strategy": name,
                    "raw_r2_reward": float(raw[j]),
                    "normalized_reward": float(bandit.mean_rewards[j]),
                    "policy_pulls": int(bandit.pulls[j]),
                    "selected_for_evidence_batch": j == chosen,
                }
                for j, name in enumerate(STRATEGY_NAMES)
            )
        )
        pd.DataFrame(rows).to_csv(out / "strategy_folds.csv", index=False)
        print(f"feature strategies {fold + 1}/{len(y)}", flush=True)
    result = pd.DataFrame(rows)
    summary = result.groupby("strategy", sort=False).agg(
        selection_count=("selected_for_evidence_batch", "sum"),
        policy_pull_count=("policy_pulls", "sum"),
        raw_r2_reward=("raw_r2_reward", "mean"),
    )
    summary.to_csv(out / "strategy_summary.csv")
