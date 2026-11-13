from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.utils.validation import check_is_fitted

__all__ = ["MonotonicGradientBoosting", "infer_monotonic_constraints"]


def infer_monotonic_constraints(
    X: np.ndarray, y: np.ndarray, corr_threshold: float = 0.3
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_samples, n_features = X.shape
    cst = np.zeros(n_features, dtype=int)
    if n_samples < 5:
        return cst
    for j in range(n_features):
        col = X[:, j]
        if np.all(col == col[0]):
            continue
        rho = spearmanr(col, y).statistic
        if np.isnan(rho):
            continue
        if rho >= corr_threshold:
            cst[j] = 1
        elif rho <= -corr_threshold:
            cst[j] = -1
    return cst


class MonotonicGradientBoosting(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        monotonic_cst: Optional[Sequence[int]] = None,
        corr_threshold: float = 0.3,
        max_iter: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        min_samples_leaf: int = 5,
        random_state: int = 42,
    ) -> None:
        self.monotonic_cst = monotonic_cst
        self.corr_threshold = corr_threshold
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> "MonotonicGradientBoosting":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if self.monotonic_cst is not None:
            cst = np.asarray(self.monotonic_cst, dtype=int)
            if cst.shape[0] != X.shape[1]:
                raise ValueError(
                    f"monotonic_cst has length {cst.shape[0]} but X has {X.shape[1]} features."
                )
        else:
            cst = infer_monotonic_constraints(X, y, self.corr_threshold)
        self.monotonic_cst_ = cst
        self.model_ = HistGradientBoostingRegressor(
            monotonic_cst=cst.tolist(),
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.model_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "model_")
        return self.model_.predict(np.asarray(X, dtype=np.float64))
