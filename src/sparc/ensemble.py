from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from .cache import EvidenceCache
from .composition import (
    evidence_matrix,
    importance_loo_predictive,
    solve_weights_bayes,
    stability_screen,
)
from .fusion import STRATEGY_NAMES, FeatureFusion, select_feature_strategy
from .models import FAMILY_MAP, build_library
from .preprocessing import Preprocessor, feature_matrix
from .figure_io import check_sampling, validate_family_priors

__all__ = ["SparseEvidenceEnsemble"]


class SparseEvidenceEnsemble(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        family_priors=None,
        random_state=42,
        feature_strategy=None,
        feature_n_rounds=None,
        model_names=None,
        n_jobs=1,
        evidence_cache=None,
    ):
        self.family_priors = family_priors
        self.random_state = random_state
        self.feature_strategy = feature_strategy
        self.feature_n_rounds = feature_n_rounds
        self.model_names = model_names
        self.n_jobs = n_jobs
        self.evidence_cache = evidence_cache

    def _resolve_feature_strategy(self, X, y):
        self.feature_bandit_pulls_ = None
        self.feature_bandit_rewards_ = None
        if self.feature_strategy == "auto":
            chosen, pulls, rewards = select_feature_strategy(
                X,
                y,
                n_rounds=self.feature_n_rounds,
                cv_splits=5,
                delta=0.1,
                random_state=self.random_state,
            )
            self.feature_bandit_pulls_ = pulls
            self.feature_bandit_rewards_ = rewards
        else:
            chosen = self.feature_strategy
            if chosen is not None and chosen not in STRATEGY_NAMES:
                raise ValueError(
                    f"feature_strategy must be None, 'auto' or one of {STRATEGY_NAMES}"
                )
        self.feature_strategy_ = chosen
        return chosen

    def _new_transformer(self, strategy):
        steps = [("preprocess", Preprocessor())]
        if strategy is not None:
            steps.append(
                (
                    "fusion",
                    FeatureFusion(strategy=strategy, random_state=self.random_state),
                )
            )
        return Pipeline(steps)

    def fit(self, X, y):
        for name in list(vars(self)):
            if name.endswith("_") or name in {
                "_weight_samples",
                "_active_names",
                "_model_names",
            }:
                delattr(self, name)
        X, y = (feature_matrix(X), np.asarray(y, dtype=float))
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or (len(y) < 3):
            raise ValueError(
                "X and y must align and contain at least three observations"
            )
        if not np.isfinite(y).all():
            raise ValueError("targets must be finite")
        validate_family_priors(self.family_priors)
        library = build_library(self.random_state)
        if self.model_names is not None:
            names = list(self.model_names)
            available = {name for name, _ in library}
            if not names or len(set(names)) != len(names) or set(names) - available:
                raise ValueError(
                    "model_names must be distinct members of the manuscript library"
                )
            library = [(name, model) for name, model in library if name in names]
        strategy = self._resolve_feature_strategy(X, y)
        factory = lambda: self._new_transformer(strategy)
        if len(y) >= 20:
            active, frequencies = stability_screen(
                library,
                X,
                y,
                n_trials=50,
                subsample_ratio=0.8,
                threshold=0.6,
                inner_splits="loo",
                random_state=self.random_state,
                transformer_factory=factory,
            )
        else:
            active = np.ones(len(library), dtype=bool)
            frequencies = np.ones(len(library))
        retained = [pair for pair, keep in zip(library, active) if keep]
        compute = lambda: evidence_matrix(
            retained,
            X,
            y,
            inner_splits="loo",
            transformer_factory=factory,
            n_jobs=self.n_jobs,
        )
        self.evidence_cache_hit_ = False
        if self.evidence_cache is None:
            P = compute()
        else:
            cache = (
                self.evidence_cache
                if isinstance(self.evidence_cache, EvidenceCache)
                else EvidenceCache(self.evidence_cache)
            )
            key = cache.make_key(
                X,
                y,
                model_names=[name for name, _ in retained],
                inner_splits="loo",
                extra={
                    "schema": 4,
                    "strategy": strategy,
                    "seed": self.random_state,
                    "library": [
                        (
                            name,
                            type(model).__module__,
                            type(model).__qualname__,
                            model.get_params(deep=True),
                        )
                        for name, model in retained
                    ],
                },
            )
            P, self.evidence_cache_hit_ = cache.get_or_compute(key, compute)
        if P.shape != (len(y), len(retained)) or not np.isfinite(P).all():
            raise ValueError("invalid LOO evidence matrix")
        scales = np.array(
            [
                (self.family_priors or {}).get(FAMILY_MAP[name], 1.0)
                for name, _ in retained
            ]
        )
        weights, _, diagnostics = solve_weights_bayes(
            P,
            y,
            num_warmup=1000,
            num_samples=2000,
            num_chains=4,
            random_seed=self.random_state,
            family_scales=scales,
        )
        check_sampling(diagnostics)
        samples = np.asarray(diagnostics.pop("weight_samples"), dtype=float)
        sigma2 = np.asarray(diagnostics.pop("sigma2_samples"), dtype=float)
        oof = importance_loo_predictive(P, y, samples, sigma2_samples=sigma2)
        transformer = factory()
        transformed = transformer.fit_transform(X, y)
        fitted = {}
        for name, template in retained:
            try:
                fitted[name] = clone(template).fit(transformed, y)
            except Exception as exc:
                raise RuntimeError(
                    f"retained model {name} failed its final refit"
                ) from exc
        self.models_ = fitted
        self.transformer_ = transformer
        self._active_names = [name for name, _ in retained]
        self._model_names = [name for name, _ in library]
        self._weight_samples = samples
        self.weights_ = np.zeros(len(library))
        self.weights_[active] = weights
        self.weight_std_ = samples.std(axis=0)
        self.active_mask_ = active
        self.selection_frequencies_ = frequencies
        self.evidence_matrix_ = P
        self.evidence_strategy_ = strategy
        self.oof_predictions_ = oof
        residual = y - oof
        self.aleatoric_ = float(
            1.4826 * np.median(np.abs(residual - np.median(residual)))
        )
        self.diagnostics_ = {
            **diagnostics,
            "aleatoric_residual_method": "importance_loo",
        }
        self.family_weights_ = {}
        for name, weight in zip(self._model_names, self.weights_):
            family = FAMILY_MAP[name]
            self.family_weights_[family] = self.family_weights_.get(
                family, 0.0
            ) + float(weight)
        self.n_features_in_ = X.shape[1]
        self.n_transformed_features_ = transformed.shape[1]
        return self

    def predict(self, X):
        return self.predict_with_uncertainty(X)[0]

    def predict_with_uncertainty(self, X):
        check_is_fitted(self, "models_")
        X = feature_matrix(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("input feature shape differs from the training data")
        transformed = self.transformer_.transform(X)
        surfaces = np.column_stack(
            [self.models_[name].predict(transformed) for name in self._active_names]
        )
        if not np.isfinite(surfaces).all():
            raise ValueError("retained model produced a nonfinite prediction")
        mean = surfaces @ self.weights_[self.active_mask_]
        epistemic = (self._weight_samples @ surfaces.T).std(axis=0)
        self.weight_uncertainty_ = epistemic
        return (mean, epistemic)

    def diagnostics_report(self):
        check_is_fitted(self, "models_")
        return {
            "weight_method": "bayes",
            "posterior_update": "batch",
            "feature_strategy": self.feature_strategy_,
            "n_features_in": self.n_features_in_,
            "n_active_models": int(self.active_mask_.sum()),
            "weights": dict(zip(self._model_names, self.weights_.tolist())),
            "family_weights": self.family_weights_,
            "aleatoric": self.aleatoric_,
            "selection_frequencies": dict(
                zip(self._model_names, self.selection_frequencies_.tolist())
            ),
            "retained_models": list(self._active_names),
            "soft_gating": False,
            "solver_diagnostics": dict(self.diagnostics_),
        }

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
