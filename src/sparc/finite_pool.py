from __future__ import annotations
from typing import Callable, Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from .acquisition import BootstrapEnsemble, trust_weighted_confidence_bound
from .preprocessing import Preprocessor
from .uncertainty import trust_score

__all__ = [
    "run_finite_pool_search",
    "discovery_count",
    "GPReferenceModel",
    "RFReferenceModel",
    "gp_reference_model",
    "rf_reference_model",
    "bootstrap_model",
]


class GPReferenceModel:
    def __init__(self, random_state: int = 42, n_restarts_optimizer: int = 2) -> None:
        self.random_state = random_state
        self.n_restarts_optimizer = n_restarts_optimizer

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPReferenceModel":
        self.preprocessor_ = Preprocessor().fit(X)
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(
            noise_level=0.1, noise_level_bounds=(1e-06, 10.0)
        )
        self.model_ = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
            alpha=1e-06,
        ).fit(self.preprocessor_.transform(X), y)
        return self

    def predict_with_uncertainty(self, X: np.ndarray):
        mean, sd = self.model_.predict(self.preprocessor_.transform(X), return_std=True)
        return (np.asarray(mean, dtype=np.float64), np.asarray(sd, dtype=np.float64))


class RFReferenceModel:
    def __init__(self, random_state: int = 42, n_estimators: int = 300) -> None:
        self.random_state = random_state
        self.n_estimators = n_estimators

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFReferenceModel":
        self.preprocessor_ = Preprocessor().fit(X)
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1
        ).fit(self.preprocessor_.transform(X), y)
        return self

    def predict_with_uncertainty(self, X: np.ndarray):
        Z = self.preprocessor_.transform(X)
        per_tree = np.stack([tree.predict(Z) for tree in self.model_.estimators_])
        return (per_tree.mean(axis=0), per_tree.std(axis=0))


def bootstrap_model(seed: int) -> BootstrapEnsemble:
    return BootstrapEnsemble(n_bootstrap=150, random_state=seed)


def gp_reference_model(seed: int) -> GPReferenceModel:
    return GPReferenceModel(random_state=seed)


def rf_reference_model(seed: int) -> RFReferenceModel:
    return RFReferenceModel(random_state=seed)


def run_finite_pool_search(
    X: np.ndarray,
    y: np.ndarray,
    initial_indices: Sequence[int],
    maximize: bool,
    model_factory: Optional[Callable[[int], object]] = None,
    use_trust: bool = True,
    beta: float = 1.96,
    reference_scale: float = 1.0,
    n_evaluations: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X must be 2-dimensional and aligned with 1-dimensional y")
    if not np.isfinite(y).all():
        raise ValueError("y must contain only finite values")
    initial = [int(i) for i in initial_indices]
    if len(set(initial)) != len(initial) or not initial:
        raise ValueError("initial_indices must be non-empty and unique")
    if min(initial) < 0 or max(initial) >= len(y):
        raise ValueError("initial_indices out of range")
    budget = len(y) if n_evaluations is None else int(n_evaluations)
    if budget < len(initial) or budget > len(y):
        raise ValueError(
            "n_evaluations must lie between len(initial_indices) and len(y)"
        )
    if model_factory is None:
        model_factory = bootstrap_model
    labelled = list(initial)
    remaining = np.setdiff1d(np.arange(len(y)), np.asarray(initial))
    better = max if maximize else min
    rows = []
    best = None
    for i in labelled:
        best = y[i] if best is None else better(best, y[i])
        rows.append(
            {
                "evaluation": len(rows) + 1,
                "candidate_index": i,
                "y": float(y[i]),
                "best_so_far": float(best),
                "phase": "initial",
            }
        )
    while len(labelled) < budget and len(remaining):
        model = model_factory(int(random_state))
        model.fit(X[labelled], y[labelled])
        mean, uncertainty = model.predict_with_uncertainty(X[remaining])
        mean = np.asarray(mean, dtype=np.float64)
        uncertainty = np.asarray(uncertainty, dtype=np.float64)
        if use_trust:
            trust = trust_score(
                uncertainty, float(model.residual_scale_), reference_scale
            )
        else:
            trust = np.ones_like(mean)
        score = trust_weighted_confidence_bound(
            mean, uncertainty, trust, beta=beta, maximize=maximize
        )
        position = int(np.argmax(score))
        chosen = int(remaining[position])
        labelled.append(chosen)
        remaining = np.delete(remaining, position)
        best = better(best, y[chosen])
        rows.append(
            {
                "evaluation": len(rows) + 1,
                "candidate_index": chosen,
                "y": float(y[chosen]),
                "best_so_far": float(best),
                "phase": "acquired",
                "predicted_mean": float(mean[position]),
                "uncertainty": float(uncertainty[position]),
                "trust": float(trust[position]),
                "acquisition_score": float(score[position]),
            }
        )
    return pd.DataFrame(rows)


def discovery_count(
    trajectory: pd.DataFrame,
    pool_optimum: float,
    maximize: bool,
    rel_tol: float = 0.001,
) -> Optional[int]:
    if rel_tol < 0:
        raise ValueError("rel_tol must be non-negative")
    best = trajectory["best_so_far"].to_numpy(dtype=np.float64)
    gap = pool_optimum - best if maximize else best - pool_optimum
    hit = np.flatnonzero(gap <= rel_tol * abs(float(pool_optimum)) + 1e-15)
    if len(hit) == 0:
        return None
    return int(trajectory["evaluation"].to_numpy()[hit[0]])
