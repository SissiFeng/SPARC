from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from .preprocessing import Preprocessor
from .uncertainty import resolve_reference_scale, trust_score


def quality_report(
    X: np.ndarray,
    y: np.ndarray,
    oof_predictions: np.ndarray,
    oof_uncertainty: Optional[np.ndarray] = None,
    trust_uncertainty: Optional[np.ndarray] = None,
    reference_scale="training_mad",
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
    conf_p = np.array(
        [
            (1.0 + np.sum(np.delete(nonconformity, i) >= nonconformity[i])) / n
            for i in range(n)
        ]
    )
    flag_conformal = conf_p < conformal_alpha
    ridge_full = Ridge(alpha=1.0).fit(Xs, y)
    full_pred = ridge_full.predict(Xs)
    influence = np.abs(full_pred - oof)
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
    if trust_uncertainty is None:
        raise ValueError("trust requires the OOF predictive dispersion of each record")
    trust_u = np.asarray(trust_uncertainty, dtype=np.float64)
    if trust_u.shape != y.shape or not np.isfinite(trust_u).all() or (trust_u < 0).any():
        raise ValueError("trust uncertainty must be finite, non-negative and aligned")
    residual_scale = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
    q = resolve_reference_scale(reference_scale, y)
    trust = trust_score(trust_u, residual_scale, q)
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
            "trust_uncertainty": trust_u,
            "residual_scale": residual_scale,
            "reference_scale": q,
            "trust": trust,
        }
    )
    report["n_flags"] = report[
        ["flag_conformal", "flag_influence", "flag_iforest", "flag_mahalanobis"]
    ].sum(axis=1)
    return report.sort_values("n_flags", ascending=False).reset_index(drop=True)


def select_priority(report, n_priority=None, min_flags=2):
    eligible = report[report.n_flags >= min_flags].sort_values(
        ["n_flags", "conformal_pvalue", "influence", "idx"],
        ascending=[False, True, False, True],
        kind="mergesort",
    )
    if eligible.empty:
        raise ValueError("no record has the required number of diagnostic flags")
    if n_priority is None:
        chosen = eligible
    else:
        if not isinstance(n_priority, (int, np.integer)) or not 1 <= n_priority <= len(eligible):
            raise ValueError("n_priority must be an integer within the eligible count")
        chosen = eligible.head(int(n_priority))
    return chosen.idx.to_numpy(dtype=int), int(len(eligible))


def matched_cohort(
    report, X, ids, n_controls, seed, n_priority=None, n_residual_baseline=0,
    priority_ids=None, additional_control_ids=None,
):
    ids = pd.Index(ids)
    if len(ids) != len(X) or not ids.is_unique or ids.hasnans:
        raise ValueError("IDs must be unique, nonmissing and aligned with X")
    if not report.idx.is_unique or set(report.idx) != set(range(len(X))):
        raise ValueError("diagnostic rows must cover X exactly once")
    indexed = report.set_index("idx").reindex(range(len(X)))
    counts = indexed["n_flags"].to_numpy()
    if priority_ids is None:
        priority, n_eligible = select_priority(report, n_priority)
    else:
        priority = ids.get_indexer(priority_ids)
        if not len(priority) or (priority < 0).any() or len(set(priority)) != len(priority):
            raise ValueError("priority_ids must be nonempty, unique and known")
        if (counts[priority] < 2).any():
            raise ValueError("priority_ids must have at least two diagnostic flags")
        if n_priority is not None and n_priority != len(priority):
            raise ValueError("n_priority must match the explicit priority_ids")
        n_eligible = int((counts >= 2).sum())
    clean = np.flatnonzero(counts == 0).tolist()
    if not isinstance(n_controls, (int, np.integer)) or not len(priority) <= n_controls <= len(clean):
        raise ValueError("need at least one distinct unflagged control per priority")
    extras = ids.get_indexer([] if additional_control_ids is None else additional_control_ids)
    if (extras < 0).any() or len(set(extras)) != len(extras) or (counts[extras] != 0).any():
        raise ValueError("additional_control_ids must be unique, known and unflagged")
    if len(extras) != n_controls - len(priority):
        raise ValueError("controls beyond one per priority require explicit additional_control_ids")
    clean = [i for i in clean if i not in set(extras)]
    Z = Preprocessor().fit_transform(X)
    rows = [
        {
            "row_id": ids[i],
            "row_index": int(i),
            "group": "priority",
            "original": float(indexed.loc[i, "y"]),
            "n_flags": int(indexed.loc[i, "n_flags"]),
            "trust": float(indexed.loc[i, "trust"]),
            "eligible_priority_records": n_eligible,
        }
        for i in priority
    ]
    selected = []
    for i in priority:
        distance = np.linalg.norm(Z[clean] - Z[i], axis=1)
        pos = int(np.argmin(distance))
        j = clean.pop(pos)
        selected.append((j, ids[i], float(distance[pos])))
    selected.extend((int(j), None, np.nan) for j in extras)
    rows += [
        {
            "row_id": ids[j],
            "row_index": int(j),
            "group": "control",
            "original": float(indexed.loc[j, "y"]),
            "n_flags": int(indexed.loc[j, "n_flags"]),
            "trust": float(indexed.loc[j, "trust"]),
            "matched_priority_id": partner,
            "distance": distance,
            "selection_rule": "nearest_unflagged_without_replacement" if partner is not None else "explicit_additional_control",
        }
        for j, partner, distance in selected
    ]
    n_baseline = 0 if n_residual_baseline is None else n_residual_baseline
    if not isinstance(n_baseline, (int, np.integer)) or n_baseline < 0:
        raise ValueError("n_residual_baseline must be a nonnegative integer")
    if n_baseline > 0:
        used = set(int(i) for i in priority) | {int(j) for j, _, _ in selected}
        candidates = [
            int(i)
            for i in np.argsort(-np.abs(indexed["residual"].to_numpy()), kind="mergesort")
            if int(i) not in used and int(counts[int(i)]) < 2
        ][:n_baseline]
        if len(candidates) < n_baseline:
            raise ValueError("not enough records for the residual-only baseline")
        rows += [
            {
                "row_id": ids[j],
                "row_index": int(j),
                "group": "residual_baseline",
                "original": float(indexed.loc[j, "y"]),
                "n_flags": int(indexed.loc[j, "n_flags"]),
                "trust": float(indexed.loc[j, "trust"]),
            }
            for j in candidates
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
    if not {"priority", "control"}.issubset(set(cohort.group)) or not set(
        cohort.group
    ).issubset({"priority", "control", "residual_baseline"}):
        raise ValueError("priority and control groups are required")
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
