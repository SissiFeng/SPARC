from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from .preprocessing import Preprocessor


def quality_report(
    X: np.ndarray,
    y: np.ndarray,
    oof_predictions: np.ndarray,
    oof_uncertainty: Optional[np.ndarray] = None,
    conformal_alpha: float = 0.1,
    influence_percentile: float = 90.0,
    iforest_contamination: float = 0.1,
    mahalanobis_alpha: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.float64)
    oof = np.asarray(oof_predictions, dtype=np.float64)
    if X.ndim != 2 or y.ndim != 1 or oof.ndim != 1:
        raise ValueError("X must be 2-dimensional and y/oof_predictions 1-dimensional")
    if len(X) != len(y) or len(oof) != len(y):
        raise ValueError("X, y and oof_predictions have inconsistent lengths")
    if not np.isfinite(y).all() or not np.isfinite(oof).all():
        raise ValueError("y and oof_predictions must contain only finite values")
    n = len(y)
    resid = y - oof
    if oof_uncertainty is not None:
        sigma = np.asarray(oof_uncertainty, dtype=np.float64)
        if sigma.shape != y.shape or not np.isfinite(sigma).all() or (sigma <= 0).any():
            raise ValueError("OOF uncertainty must be positive, finite and aligned")
    else:
        sigma = np.full(n, max(float(np.median(np.abs(resid))), 1e-08))
    Xs = Preprocessor().fit(X).transform(X)
    p = Xs.shape[1]
    if p == 0:
        raise ValueError("X has no non-constant usable features")
    nonconformity = np.abs(resid) / (sigma + 1e-08)
    conf_p = np.array([(nonconformity >= nonconformity[i]).mean() for i in range(n)])
    flag_conformal = conf_p < conformal_alpha
    ridge_full = Ridge(alpha=1.0).fit(Xs, y)
    full_pred = ridge_full.predict(Xs)
    loo_pred = np.empty(n, dtype=np.float64)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        fold_preprocessor = Preprocessor().fit(X[mask])
        loo_pred[i] = (
            Ridge(alpha=1.0)
            .fit(fold_preprocessor.transform(X[mask]), y[mask])
            .predict(fold_preprocessor.transform(X[[i]]))[0]
        )
    influence = np.abs(full_pred - loo_pred)
    flag_influence = influence >= np.percentile(influence, influence_percentile)
    iforest = IsolationForest(
        n_estimators=200, contamination=iforest_contamination, random_state=random_state
    ).fit(Xs)
    iforest_score = -iforest.score_samples(Xs)
    flag_iforest = iforest.predict(Xs) == -1
    lw = LedoitWolf().fit(Xs)
    mahal_sq = lw.mahalanobis(Xs)
    mahal_p = stats.chi2.sf(mahal_sq, df=p)
    flag_mahal = mahal_p < mahalanobis_alpha
    report = pd.DataFrame(
        {
            "idx": np.arange(n),
            "y": y,
            "oof_prediction": oof,
            "residual": resid,
            "nonconformity": nonconformity,
            "conformal_pvalue": conf_p,
            "influence": influence,
            "iforest_score": iforest_score,
            "mahalanobis_sq": mahal_sq,
            "mahalanobis_pvalue": mahal_p,
            "flag_conformal": flag_conformal,
            "flag_influence": flag_influence,
            "flag_iforest": flag_iforest,
            "flag_mahalanobis": flag_mahal,
        }
    )
    report["n_flags"] = report[
        ["flag_conformal", "flag_influence", "flag_iforest", "flag_mahalanobis"]
    ].sum(axis=1)
    return report.sort_values("n_flags", ascending=False).reset_index(drop=True)


def matched_cohort(report, X, ids, priority_ids, n_controls, seed):
    ids = pd.Index(ids)
    priority = ids.get_indexer(priority_ids)
    if (
        len(priority) == 0
        or (priority < 0).any()
        or len(set(priority)) != len(priority)
    ):
        raise ValueError("priority_ids must be unique known IDs")
    counts = report.set_index("idx")["n_flags"].reindex(range(len(X))).to_numpy()
    if (counts[priority] < 2).any():
        raise ValueError("every selected priority ID must have at least two flags")
    clean = np.flatnonzero(counts == 0).tolist()
    if not len(priority) <= n_controls <= len(clean):
        raise ValueError("need at least one distinct unflagged control per priority")
    Z = Preprocessor().fit_transform(X)
    rows = [
        {
            "row_id": ids[i],
            "row_index": int(i),
            "group": "priority",
            "original": float(report.set_index("idx").loc[i, "y"]),
        }
        for i in priority
    ]
    selected = []
    for i in priority:
        distance = np.linalg.norm(Z[clean] - Z[i], axis=1)
        pos = int(np.argmin(distance))
        j = clean.pop(pos)
        selected.append((j, ids[i], float(distance[pos])))
    rng = np.random.default_rng(seed)
    extras = rng.choice(clean, n_controls - len(selected), replace=False)
    selected += [(int(j), None, np.nan) for j in extras]
    values = report.set_index("idx")["y"]
    rows += [
        {
            "row_id": ids[j],
            "row_index": j,
            "group": "control",
            "original": float(values[j]),
            "matched_priority_id": partner,
            "distance": distance,
        }
        for j, partner, distance in selected
    ]
    return pd.DataFrame(rows)


def repeatability(cohort, repeats):
    if not {"row_id", "group", "original"}.issubset(cohort):
        raise ValueError("cohort must contain row_id, group and original")
    if not {"row_id", "repeated"}.issubset(repeats):
        raise ValueError("repeats must contain row_id and repeated")
    if cohort.row_id.duplicated().any() or repeats.row_id.duplicated().any():
        raise ValueError("one row per measurement ID is required")
    if set(cohort.row_id) != set(repeats.row_id):
        raise ValueError("repeat IDs must exactly match the fixed cohort")
    if set(cohort.group) != {"priority", "control"}:
        raise ValueError("both priority and control groups are required")
    joined = cohort.merge(
        repeats[["row_id", "repeated"]], on="row_id", validate="one_to_one"
    )
    if not np.isfinite(joined[["original", "repeated"]].to_numpy(dtype=float)).all():
        raise ValueError("all original and repeated measurements must be finite")
    joined["absolute_difference"] = np.abs(joined.original - joined.repeated)
    summary = joined.groupby("group").absolute_difference.agg(
        ["count", "mean", "median"]
    )
    return (joined, summary)
