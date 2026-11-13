from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
from .acquisition import trust_weighted_thompson
from .ood import GPExpertMixture, uncertainty_inflation_ratio
from .uncertainty import trust_score

__all__ = [
    "central_interval_split",
    "ExtrapolationRun",
    "run_controlled_extrapolation",
    "true_top_set",
    "precision_at_k",
    "batch_hit_rates",
]


def central_interval_split(
    X: np.ndarray, lower: float = 20.0, upper: float = 80.0
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a non-empty 2-dimensional array")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values for percentile splitting")
    if not 0.0 <= lower < upper <= 100.0:
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")
    lo = np.percentile(X, lower, axis=0)
    hi = np.percentile(X, upper, axis=0)
    return np.all((X >= lo) & (X <= hi), axis=1)


@dataclass
class ExtrapolationRun:
    batches: List[np.ndarray]
    rounds: pd.DataFrame
    uncertainty_ratio: float
    expert_weights: List[np.ndarray] = field(default_factory=list)

    @property
    def acquisition_order(self) -> np.ndarray:
        return np.concatenate(self.batches) if self.batches else np.zeros(0, dtype=int)


def run_controlled_extrapolation(
    X: np.ndarray,
    y: np.ndarray,
    initial_mask: np.ndarray,
    batch_size: int,
    n_rounds: int = 4,
    maximize: bool = True,
    random_state: int = 0,
    reference_scale: float = 1.0,
    model_factory: Optional[Callable[[int], object]] = None,
) -> ExtrapolationRun:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    initial_mask = np.asarray(initial_mask, dtype=bool)
    if (
        X.ndim != 2
        or y.ndim != 1
        or len(X) != len(y)
        or (initial_mask.shape != y.shape)
    ):
        raise ValueError("X, y and initial_mask must be aligned")
    if not np.isfinite(y).all():
        raise ValueError("y must contain only finite values")
    if int(batch_size) < 1 or int(n_rounds) < 1:
        raise ValueError("batch_size and n_rounds must be positive")
    if initial_mask.all() or not initial_mask.any():
        raise ValueError("initial_mask must leave both labelled and pool candidates")
    if model_factory is None:
        model_factory = lambda seed: GPExpertMixture(random_state=seed)
    pool = np.flatnonzero(~initial_mask)
    labelled = list(np.flatnonzero(initial_mask))
    remaining = np.arange(len(pool))
    batches: List[np.ndarray] = []
    records: List[Dict] = []
    weights_by_round: List[np.ndarray] = []
    uncertainty_ratio = float("nan")
    for round_no in range(int(n_rounds)):
        if len(remaining) == 0:
            break
        model = model_factory(int(random_state))
        model.fit(X[labelled], y[labelled])
        mean, uncertainty = model.predict_with_uncertainty(X[pool[remaining]])
        residual_scale = float(getattr(model, "residual_scale_"))
        trust = trust_score(uncertainty, residual_scale, reference_scale)
        if round_no == 0:
            _, sd_initial = model.predict_with_uncertainty(X[labelled])
            uncertainty_ratio = uncertainty_inflation_ratio(uncertainty, sd_initial)
        if hasattr(model, "expert_weights_"):
            weights_by_round.append(np.asarray(model.expert_weights_).copy())
        score = trust_weighted_thompson(
            mean,
            uncertainty,
            trust,
            maximize=maximize,
            random_state=int(random_state) + round_no,
        )
        order = np.argsort(score, kind="mergesort")[::-1][: int(batch_size)]
        batch = remaining[order]
        batches.append(batch.copy())
        for rank, (local, pos) in enumerate(zip(order, batch), start=1):
            records.append(
                {
                    "round": round_no + 1,
                    "rank_in_batch": rank,
                    "pool_index": int(pos),
                    "candidate_index": int(pool[pos]),
                    "predicted_mean": float(mean[local]),
                    "uncertainty": float(uncertainty[local]),
                    "residual_scale": residual_scale,
                    "trust": float(trust[local]),
                    "acquisition_score": float(score[local]),
                }
            )
        labelled.extend((int(pool[pos]) for pos in batch))
        remaining = np.setdiff1d(remaining, batch, assume_unique=True)
    rounds = pd.DataFrame(records)
    if len(rounds):
        rounds["y_revealed"] = y[rounds["candidate_index"].to_numpy()]
    return ExtrapolationRun(
        batches=batches,
        rounds=rounds,
        uncertainty_ratio=uncertainty_ratio,
        expert_weights=weights_by_round,
    )


def true_top_set(y_pool: np.ndarray, fraction: float, maximize: bool) -> np.ndarray:
    y_pool = np.asarray(y_pool, dtype=np.float64)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    n_top = int(round(fraction * len(y_pool)))
    key = -y_pool if maximize else y_pool
    return np.argsort(key, kind="mergesort")[:n_top]


def precision_at_k(
    batches: Sequence[np.ndarray], y_pool: np.ndarray, maximize: bool
) -> pd.DataFrame:
    y_pool = np.asarray(y_pool, dtype=np.float64)
    n = len(y_pool)
    rows = []
    acquired: List[int] = []
    for b, batch in enumerate(batches, start=1):
        acquired.extend((int(i) for i in batch))
        size = len(acquired)
        k = size / n
        top = set(true_top_set(y_pool, k, maximize).tolist())
        hits = len(top.intersection(acquired))
        rows.append(
            {
                "cutoff_batches": b,
                "n_acquired": size,
                "fraction_K": k,
                "precision_at_K": hits / size if size else float("nan"),
                "random_expectation": k,
            }
        )
    return pd.DataFrame(rows)


def batch_hit_rates(
    batches: Sequence[np.ndarray],
    y_pool: np.ndarray,
    maximize: bool,
    top_fraction: float = 0.25,
    z: float = 1.96,
) -> pd.DataFrame:
    y_pool = np.asarray(y_pool, dtype=np.float64)
    top = set(true_top_set(y_pool, top_fraction, maximize).tolist())
    rows = []
    for b, batch in enumerate(batches, start=1):
        n = len(batch)
        hits = sum((int(i) in top for i in batch))
        p = hits / n if n else float("nan")
        se = float(np.sqrt(p * (1.0 - p) / n)) if n else float("nan")
        rows.append(
            {
                "round": b,
                "batch_size": n,
                "hits": hits,
                "hit_rate": p,
                "se": se,
                "lower": p - z * se,
                "upper": p + z * se,
                "random_expectation": top_fraction,
            }
        )
    return pd.DataFrame(rows)
