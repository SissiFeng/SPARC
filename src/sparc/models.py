from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.utils.validation import check_is_fitted
from .monotonic import MonotonicGradientBoosting

__all__ = ["FAMILY_MAP", "FAMILIES", "build_library", "AdaptiveKNN"]
FAMILIES: Tuple[str, ...] = ("linear", "tree_based", "locally_smooth", "monotonic")
FAMILY_MAP: Dict[str, str] = {
    "Ridge": "linear",
    "Lasso": "linear",
    "ElasticNet": "linear",
    "BayesianRidge": "linear",
    "Huber": "linear",
    "RF": "tree_based",
    "ET": "tree_based",
    "GBR": "tree_based",
    "SVR": "locally_smooth",
    "KNN": "locally_smooth",
    "GPR": "locally_smooth",
    "Mono": "monotonic",
}


class AdaptiveKNN(BaseEstimator, RegressorMixin):
    def __init__(self, n_neighbors: int = 5, weights: str = "distance") -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> "AdaptiveKNN":
        del sample_weight
        X = np.asarray(X)
        y = np.asarray(y)
        k = int(min(self.n_neighbors, max(1, len(y))))
        self.model_ = KNeighborsRegressor(n_neighbors=k, weights=self.weights)
        self.model_.fit(X, y)
        self.n_neighbors_ = k
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "model_")
        return self.model_.predict(np.asarray(X))


def build_library(random_state: int = 42) -> List[Tuple[str, object]]:
    models: List[Tuple[str, object]] = [
        ("Ridge", Ridge(alpha=1.0, random_state=random_state)),
        ("Lasso", Lasso(alpha=0.001, max_iter=5000, random_state=random_state)),
        (
            "ElasticNet",
            ElasticNet(
                alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=random_state
            ),
        ),
        ("BayesianRidge", BayesianRidge()),
        ("Huber", HuberRegressor(epsilon=1.35, max_iter=500)),
        (
            "RF",
            RandomForestRegressor(
                n_estimators=200, max_depth=10, random_state=random_state, n_jobs=-1
            ),
        ),
        (
            "ET",
            ExtraTreesRegressor(
                n_estimators=200, max_depth=10, random_state=random_state, n_jobs=-1
            ),
        ),
        (
            "GBR",
            GradientBoostingRegressor(
                n_estimators=150, max_depth=4, random_state=random_state
            ),
        ),
        ("SVR", SVR(kernel="rbf", C=10.0, epsilon=0.1)),
        ("KNN", AdaptiveKNN(n_neighbors=5, weights="distance")),
        ("Mono", MonotonicGradientBoosting(random_state=random_state)),
        (
            "GPR",
            GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5),
                n_restarts_optimizer=2,
                alpha=0.01,
                normalize_y=True,
                random_state=random_state,
            ),
        ),
    ]
    return models
