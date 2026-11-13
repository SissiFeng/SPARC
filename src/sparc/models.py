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

__all__ = [
    "FAMILY_MAP",
    "FAMILIES",
    "build_library",
    "AdaptiveKNN",
    "HYPERPARAMETER_GRIDS",
    "tune_library",
]
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


HYPERPARAMETER_GRIDS: Dict[str, Dict[str, list]] = {
    "Ridge": {"alpha": [0.1, 1.0, 10.0]},
    "Lasso": {"alpha": [0.0001, 0.001, 0.01, 0.1]},
    "ElasticNet": {"alpha": [0.0001, 0.001, 0.01, 0.1], "l1_ratio": [0.2, 0.5, 0.8]},
    "BayesianRidge": {},
    "Huber": {"epsilon": [1.35, 1.75, 2.5]},
    "RF": {"max_depth": [5, 10, None], "min_samples_leaf": [1, 3]},
    "ET": {"max_depth": [5, 10, None], "min_samples_leaf": [1, 3]},
    "GBR": {"max_depth": [2, 4], "learning_rate": [0.05, 0.1]},
    "SVR": {"C": [1.0, 10.0, 100.0], "epsilon": [0.01, 0.1]},
    "KNN": {"n_neighbors": [3, 5, 10]},
    "Mono": {"max_depth": [2, 3], "learning_rate": [0.05, 0.1]},
    "GPR": {},
}


def tune_library(
    library: List[Tuple[str, object]],
    X: np.ndarray,
    y: np.ndarray,
    transformer_factory=None,
    cv_splits: int = 5,
    random_state: int = 42,
    grids: Optional[Dict[str, Dict[str, list]]] = None,
) -> Tuple[List[Tuple[str, object]], Dict[str, dict]]:
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.pipeline import Pipeline

    grids = HYPERPARAMETER_GRIDS if grids is None else grids
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    n_splits = min(int(cv_splits), len(y))
    if n_splits < 2:
        raise ValueError("hyperparameter selection requires at least two rows")
    tuned: List[Tuple[str, object]] = []
    chosen: Dict[str, dict] = {}
    for name, template in library:
        grid = grids.get(name, {})
        if not grid:
            tuned.append((name, clone(template)))
            chosen[name] = {}
            continue
        steps = []
        if transformer_factory is not None:
            steps.append(("transform", transformer_factory()))
        steps.append(("model", clone(template)))
        search = GridSearchCV(
            Pipeline(steps),
            {f"model__{key}": values for key, values in grid.items()},
            scoring="neg_mean_squared_error",
            cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_state),
            error_score="raise",
            refit=False,
        )
        search.fit(X, y)
        best = {key.split("__", 1)[1]: value for key, value in search.best_params_.items()}
        tuned.append((name, clone(template).set_params(**best)))
        chosen[name] = best
    return tuned, chosen
