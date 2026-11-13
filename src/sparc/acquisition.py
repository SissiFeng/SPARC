from __future__ import annotations
from typing import List, Optional, Sequence
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from .preprocessing import Preprocessor

__all__ = [
    "trust_weighted_confidence_bound",
    "trust_weighted_lcb",
    "trust_weighted_thompson",
    "trust_weighted_ucb",
    "lcb",
    "ucb",
    "diversity_filter",
    "BootstrapEnsemble",
    "DEFAULT_BOOTSTRAP_MODELS",
]


def _validate_trust_inputs(
    mean: np.ndarray, uncertainty: np.ndarray, trust: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(mean, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    trust = np.asarray(trust, dtype=np.float64)
    if not mean.shape == uncertainty.shape == trust.shape:
        raise ValueError("mean, uncertainty and trust must share the same shape")
    if not (
        np.isfinite(mean).all()
        and np.isfinite(uncertainty).all()
        and np.isfinite(trust).all()
    ):
        raise ValueError("acquisition inputs must contain only finite values")
    if np.any(uncertainty < 0):
        raise ValueError("uncertainty must be non-negative")
    if np.any((trust < 0) | (trust > 1)):
        raise ValueError("trust must lie in [0, 1]")
    return (mean, uncertainty, trust)


def trust_weighted_confidence_bound(
    mean: np.ndarray,
    uncertainty: np.ndarray,
    trust: np.ndarray,
    beta: float = 1.96,
    maximize: bool = True,
) -> np.ndarray:
    mean, uncertainty, trust = _validate_trust_inputs(mean, uncertainty, trust)
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and non-negative")
    direction = 1.0 if maximize else -1.0
    return direction * mean + float(beta) * uncertainty * trust


def trust_weighted_ucb(
    mean: np.ndarray, uncertainty: np.ndarray, trust: np.ndarray, beta: float = 1.96
) -> np.ndarray:
    return trust_weighted_confidence_bound(
        mean, uncertainty, trust, beta=beta, maximize=True
    )


def trust_weighted_lcb(
    mean: np.ndarray, uncertainty: np.ndarray, trust: np.ndarray, beta: float = 1.96
) -> np.ndarray:
    mean, uncertainty, trust = _validate_trust_inputs(mean, uncertainty, trust)
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and non-negative")
    return mean - float(beta) * uncertainty * trust


def trust_weighted_thompson(
    mean: np.ndarray,
    trust: np.ndarray,
    joint_sample: np.ndarray,
    maximize: bool = True,
) -> np.ndarray:
    joint_sample = np.asarray(joint_sample, dtype=np.float64)
    mean, _, trust = _validate_trust_inputs(mean, np.zeros_like(joint_sample), trust)
    if joint_sample.shape != mean.shape or not np.isfinite(joint_sample).all():
        raise ValueError("joint_sample must be a finite draw aligned with mean")
    direction = 1.0 if maximize else -1.0
    return direction * (mean + trust * (joint_sample - mean))


def lcb(mean: np.ndarray, uncertainty: np.ndarray, kappa: float = 2.0) -> np.ndarray:
    return np.asarray(mean) - kappa * np.asarray(uncertainty)


def ucb(mean: np.ndarray, uncertainty: np.ndarray, kappa: float = 2.0) -> np.ndarray:
    return np.asarray(mean) + kappa * np.asarray(uncertainty)


def diversity_filter(
    X: np.ndarray,
    scores: np.ndarray,
    n_select: int,
    radius: float = 0.15,
    ascending: bool = False,
) -> List[int]:
    X = np.asarray(X, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    if len(scores) != len(X):
        raise ValueError("scores must align with X")
    if not np.isfinite(X).all() or not np.isfinite(scores).all():
        raise ValueError("diversity inputs must be finite")
    if n_select < 1 or radius < 0:
        raise ValueError("n_select must be positive and radius non-negative")
    Xs = StandardScaler().fit_transform(X)
    order = np.argsort(scores) if ascending else np.argsort(scores)[::-1]
    min_dist = radius * np.sqrt(Xs.shape[1])
    chosen: List[int] = []
    for idx in order:
        if len(chosen) >= n_select:
            break
        if all((np.linalg.norm(Xs[idx] - Xs[j]) >= min_dist for j in chosen)):
            chosen.append(int(idx))
    return chosen


DEFAULT_BOOTSTRAP_MODELS: List[object] = [
    Ridge(alpha=1.0),
    Ridge(alpha=10.0),
    RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]),
    RandomForestRegressor(n_estimators=100, max_features=0.5, random_state=42),
    RandomForestRegressor(
        n_estimators=100, max_features=0.8, min_samples_leaf=3, random_state=42
    ),
    GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    ),
]


class BootstrapEnsemble:
    def __init__(
        self,
        n_bootstrap: int = 150,
        base_models: Optional[Sequence[object]] = None,
        random_state: int = 42,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.base_models = (
            list(base_models)
            if base_models is not None
            else list(DEFAULT_BOOTSTRAP_MODELS)
        )
        self.random_state = random_state
        self._fitted: List[object] = []
        self._scalers: List[Preprocessor] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BootstrapEnsemble":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X must be 2-dimensional and aligned with 1-dimensional y")
        if len(y) < 3:
            raise ValueError("BootstrapEnsemble requires at least 3 samples")
        if not np.isfinite(y).all():
            raise ValueError("y must contain only finite values")
        if int(self.n_bootstrap) < 1:
            raise ValueError("n_bootstrap must be positive")
        if not self.base_models:
            raise ValueError("base_models must contain at least one estimator")
        rng = np.random.RandomState(self.random_state)
        self._fitted, self._scalers = ([], [])
        n = len(y)
        oob_sum = np.zeros(n, dtype=np.float64)
        oob_count = np.zeros(n, dtype=np.int64)
        for b in range(self.n_bootstrap):
            idx = rng.randint(0, n, size=n)
            scaler = Preprocessor().fit(X[idx])
            model = clone(self.base_models[b % len(self.base_models)])
            model.fit(scaler.transform(X[idx]), y[idx])
            self._fitted.append(model)
            self._scalers.append(scaler)
            in_bag = np.zeros(n, dtype=bool)
            in_bag[idx] = True
            oob = ~in_bag
            if oob.any():
                oob_sum[oob] += model.predict(scaler.transform(X[oob]))
                oob_count[oob] += 1
        valid = oob_count > 0
        self.oob_counts_ = oob_count
        self.oob_predictions_ = np.full(n, np.nan, dtype=np.float64)
        self.oob_predictions_[valid] = oob_sum[valid] / oob_count[valid]
        if not valid.all():
            raise RuntimeError(
                "every training row needs an out-of-bag prediction; increase n_bootstrap"
            )
        residual = y - self.oob_predictions_
        self.residual_scale_ = float(
            1.4826 * np.median(np.abs(residual - np.median(residual)))
        )
        self.aleatoric_ = self.residual_scale_
        self.y_train_ = y.copy()
        return self

    def predict(self, X: np.ndarray):
        if not self._fitted:
            raise RuntimeError("BootstrapEnsemble is not fitted")
        X = np.asarray(X, dtype=np.float64)
        preds = np.vstack(
            [m.predict(s.transform(X)) for m, s in zip(self._fitted, self._scalers)]
        )
        return (preds.mean(axis=0), preds.std(axis=0))

    def predict_with_uncertainty(self, X: np.ndarray):
        return self.predict(X)

    def sample_joint(self, X: np.ndarray, random_state=None) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("BootstrapEnsemble is not fitted")
        rng = (
            random_state
            if isinstance(random_state, np.random.RandomState)
            else np.random.RandomState(random_state)
        )
        b = int(rng.randint(len(self._fitted)))
        X = np.asarray(X, dtype=np.float64)
        return self._fitted[b].predict(self._scalers[b].transform(X))

    def predict_with_trust(self, X: np.ndarray, reference_scale="training_mad"):
        from .uncertainty import resolve_reference_scale, trust_score

        mean, uncertainty = self.predict(X)
        q = resolve_reference_scale(reference_scale, self.y_train_)
        trust = trust_score(uncertainty, self.residual_scale_, q)
        return (mean, uncertainty, trust)
