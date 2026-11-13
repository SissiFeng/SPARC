from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

__all__ = ["GPExpertMixture", "EXPERT_NAMES", "uncertainty_inflation_ratio"]
EXPERT_NAMES: Tuple[str, ...] = ("matern52_isotropic", "rbf_ard", "rbf_rf_importance")
_GP_JITTER = 1e-06


def uncertainty_inflation_ratio(
    uncertainty_holdout: np.ndarray, uncertainty_train: np.ndarray
) -> float:
    u_h = np.asarray(uncertainty_holdout, dtype=np.float64)
    u_t = np.asarray(uncertainty_train, dtype=np.float64)
    if u_h.size == 0 or u_t.size == 0:
        raise ValueError("uncertainty arrays must be non-empty")
    if not np.isfinite(u_h).all() or not np.isfinite(u_t).all():
        raise ValueError("uncertainty arrays must contain only finite values")
    train_mean = float(np.mean(u_t))
    if train_mean <= 0:
        raise ValueError("mean training uncertainty must be positive")
    return float(np.mean(u_h) / train_mean)


class _FittedExpert:
    def __init__(
        self,
        model: GaussianProcessRegressor,
        scaler: StandardScaler,
        feature_scale: np.ndarray,
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.feature_scale = feature_scale

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Z = self.scaler.transform(X) * self.feature_scale
        mean, sd = self.model.predict(Z, return_std=True)
        return (np.asarray(mean, dtype=np.float64), np.asarray(sd, dtype=np.float64))


def _noise_kernel() -> WhiteKernel:
    return WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-06, 10.0))


def _fit_expert(
    index: int, X: np.ndarray, y: np.ndarray, n_restarts: int, random_state: int
) -> _FittedExpert:
    n_features = X.shape[1]
    feature_scale = np.ones(n_features, dtype=np.float64)
    if index == 0:
        kernel = 1.0 * Matern(length_scale=1.0, nu=2.5) + _noise_kernel()
    elif index == 1:
        kernel = (
            1.0
            * RBF(length_scale=np.ones(n_features), length_scale_bounds=(0.01, 200.0))
            + _noise_kernel()
        )
    elif index == 2:
        rf = RandomForestRegressor(
            n_estimators=200, random_state=random_state, n_jobs=-1
        )
        rf.fit(X, y)
        importance = np.asarray(rf.feature_importances_, dtype=np.float64)
        feature_scale = np.sqrt(
            importance / max(float(importance.max()), 1e-12) + 0.0001
        )
        kernel = 1.0 * RBF(length_scale=1.0) + _noise_kernel()
    else:
        raise IndexError(index)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X) * feature_scale
    model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=int(n_restarts),
        normalize_y=True,
        random_state=random_state,
        alpha=_GP_JITTER,
    )
    model.fit(Z, y)
    return _FittedExpert(model, scaler, feature_scale)


def _mixture_moments(
    means: np.ndarray, sds: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = np.asarray(weights, dtype=np.float64)
    mean = np.sum(w[:, None] * means, axis=0)
    within = np.sum(w[:, None] * sds**2, axis=0)
    between = np.sum(w[:, None] * (means - mean[None, :]) ** 2, axis=0)
    return (mean, within, between)


class GPExpertMixture(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        cv_splits: int = 3,
        random_state: int = 42,
        n_restarts_optimizer: int = 0,
        cv_n_restarts: int = 0,
    ) -> None:
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.n_restarts_optimizer = n_restarts_optimizer
        self.cv_n_restarts = cv_n_restarts

    def _cross_validated_experts(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_splits = int(self.cv_splits)
        if not 2 <= n_splits <= len(y):
            raise ValueError("cv_splits must lie between 2 and the number of rows")
        n_experts = len(EXPERT_NAMES)
        mu_oof = np.empty((n_experts, len(y)), dtype=np.float64)
        sd_oof = np.empty((n_experts, len(y)), dtype=np.float64)
        splitter = KFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        for train, test in splitter.split(X):
            imputer = SimpleImputer(strategy="median").fit(X[train])
            X_train, X_test = (imputer.transform(X[train]), imputer.transform(X[test]))
            for index in range(n_experts):
                expert = _fit_expert(
                    index, X_train, y[train], self.cv_n_restarts, self.random_state
                )
                mean, sd = expert.predict(X_test)
                mu_oof[index, test] = mean
                sd_oof[index, test] = np.maximum(sd, 1e-09)
        loglik = (
            -0.5 * np.log(2.0 * np.pi)
            - np.log(sd_oof)
            - 0.5 * ((y[None, :] - mu_oof) / sd_oof) ** 2
        )
        return (np.mean(loglik, axis=1), mu_oof, sd_oof)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPExpertMixture":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X must be 2-dimensional and aligned with 1-dimensional y")
        if len(y) < 5:
            raise ValueError("GPExpertMixture requires at least 5 samples")
        if not np.isfinite(y).all():
            raise ValueError("y must contain only finite values")
        scores, mu_oof, sd_oof = self._cross_validated_experts(X, y)
        shifted = scores - np.max(scores)
        weights = np.exp(shifted)
        weights /= weights.sum()
        self.expert_log_likelihoods_ = scores
        self.expert_weights_ = weights
        self.oof_expert_predictions_ = mu_oof
        self.oof_expert_sd_ = sd_oof
        oof_mean = np.empty(len(y), dtype=np.float64)
        oof_within = np.empty(len(y), dtype=np.float64)
        oof_between = np.empty(len(y), dtype=np.float64)
        self.oof_mixture_weights_ = np.empty((len(y), len(EXPERT_NAMES)))
        splitter = KFold(
            n_splits=int(self.cv_splits), shuffle=True, random_state=self.random_state
        )
        for train, test in splitter.split(X):
            inner = GPExpertMixture(
                cv_splits=min(int(self.cv_splits), len(train)),
                random_state=self.random_state,
                cv_n_restarts=self.cv_n_restarts,
            )
            inner_scores, _, _ = inner._cross_validated_experts(X[train], y[train])
            fold_weights = np.exp(inner_scores - np.max(inner_scores))
            fold_weights /= fold_weights.sum()
            mean, within, between = _mixture_moments(
                mu_oof[:, test], sd_oof[:, test], fold_weights
            )
            oof_mean[test], oof_within[test], oof_between[test] = (
                mean,
                within,
                between,
            )
            self.oof_mixture_weights_[test] = fold_weights
        self.oof_predictions_ = oof_mean
        self.oof_uncertainty_ = np.sqrt(np.maximum(oof_within + oof_between, 0.0))
        residual = y - oof_mean
        self.residual_scale_ = float(
            1.4826 * np.median(np.abs(residual - np.median(residual)))
        )
        self.aleatoric_ = self.residual_scale_
        self.residual_method_ = "nested_cross_fitted_experts_and_weights"
        self.imputer_ = SimpleImputer(strategy="median").fit(X)
        X_model = self.imputer_.transform(X)
        self.experts_: List[_FittedExpert] = [
            _fit_expert(index, X_model, y, self.n_restarts_optimizer, self.random_state)
            for index in range(len(EXPERT_NAMES))
        ]
        self.expert_names_ = list(EXPERT_NAMES)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        mean, _ = self.predict_with_uncertainty(X)
        return mean

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean, _, _, total = self.predict_components(X)
        return (mean, total)

    def predict_components(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        check_is_fitted(self, "experts_")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features; mixture was fitted on {self.n_features_in_}"
            )
        X_model = self.imputer_.transform(X)
        predictions = [expert.predict(X_model) for expert in self.experts_]
        means = np.vstack([p[0] for p in predictions])
        sds = np.vstack([p[1] for p in predictions])
        mean, within, between = _mixture_moments(means, sds, self.expert_weights_)
        return (
            mean,
            np.sqrt(np.maximum(within, 0.0)),
            np.sqrt(np.maximum(between, 0.0)),
            np.sqrt(np.maximum(within + between, 0.0)),
        )

    def predict_with_trust(
        self, X: np.ndarray, reference_scale: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from .uncertainty import trust_score

        mean, uncertainty = self.predict_with_uncertainty(X)
        trust = trust_score(uncertainty, self.residual_scale_, reference_scale)
        return (mean, uncertainty, trust)

    def expert_report(self) -> Dict[str, float]:
        check_is_fitted(self, "expert_weights_")
        return {
            name: float(weight)
            for name, weight in zip(EXPERT_NAMES, self.expert_weights_)
        }
