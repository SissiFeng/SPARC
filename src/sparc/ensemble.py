from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from .preprocessing import feature_matrix
from .figure_io import validate_family_priors
from .weighting import validate_weight_strategy
from .workflow import Modeller, Planner

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
        tune_hyperparameters=True,
        weight_strategy="simplex",
        global_scale=1.0,
    ):
        self.family_priors = family_priors
        self.random_state = random_state
        self.feature_strategy = feature_strategy
        self.feature_n_rounds = feature_n_rounds
        self.model_names = model_names
        self.n_jobs = n_jobs
        self.evidence_cache = evidence_cache
        self.tune_hyperparameters = tune_hyperparameters
        self.weight_strategy = weight_strategy
        self.global_scale = global_scale

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
            raise ValueError("X and y must align and contain at least three observations")
        if not np.isfinite(y).all():
            raise ValueError("targets must be finite")
        validate_family_priors(self.family_priors)
        validate_weight_strategy(self.weight_strategy, self.global_scale)
        plan = Planner(
            feature_strategy=self.feature_strategy,
            feature_n_rounds=self.feature_n_rounds,
            random_state=self.random_state,
        ).plan(X, y)
        self.feature_strategy_ = plan.strategy
        self.feature_bandit_pulls_ = plan.bandit_pulls
        self.feature_bandit_rewards_ = plan.bandit_rewards
        composition = Modeller(
            family_priors=self.family_priors,
            random_state=self.random_state,
            model_names=self.model_names,
            n_jobs=self.n_jobs,
            evidence_cache=self.evidence_cache,
            tune_hyperparameters=self.tune_hyperparameters,
            weight_strategy=self.weight_strategy,
            global_scale=self.global_scale,
        ).fit(X, y, plan)
        self.composition_ = composition
        self.models_ = composition.models
        self.transformer_ = composition.transformer
        self._active_names = composition.active_names
        self._model_names = composition.library_names
        self._weight_samples = composition.weight_samples
        self.weights_ = composition.weights
        self.weight_std_ = composition.weight_samples.std(axis=0)
        self.active_mask_ = composition.active_mask
        self.selection_frequencies_ = composition.selection_frequencies
        self.hyperparameters_ = composition.hyperparameters
        self.evidence_matrix_ = composition.evidence
        self.evidence_strategy_ = plan.strategy
        self.evidence_cache_hit_ = composition.evidence_cache_hit
        self.oof_predictions_ = composition.oof_predictions
        self.aleatoric_ = composition.residual_scale
        self.residual_scale_ = composition.residual_scale
        self.diagnostics_ = composition.diagnostics
        self.family_weights_ = composition.family_weights
        self.weight_strategy_ = self.diagnostics_["weight_strategy"]
        self.relative_weights_ = self.weights_ / self.weights_.sum()
        self.sparsity_ = self.diagnostics_["sparsity"]
        self.n_features_in_ = X.shape[1]
        self.n_transformed_features_ = self.transformer_.transform(X[:1]).shape[1]
        return self

    def predict(self, X):
        return self.predict_with_uncertainty(X)[0]

    def _surfaces(self, X):
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
        return surfaces

    def predict_with_uncertainty(self, X):
        surfaces = self._surfaces(X)
        mean = surfaces @ self.weights_[self.active_mask_]
        epistemic = (self._weight_samples @ surfaces.T).std(axis=0)
        self.weight_uncertainty_ = epistemic
        return (mean, epistemic)

    def sample_joint(self, X, random_state=None):
        surfaces = self._surfaces(X)
        rng = (
            random_state
            if isinstance(random_state, np.random.RandomState)
            else np.random.RandomState(random_state)
        )
        draw = self._weight_samples[rng.randint(len(self._weight_samples))]
        return surfaces @ draw

    def diagnostics_report(self):
        check_is_fitted(self, "models_")
        return {
            "weight_method": "bayes",
            "weight_strategy": self.weight_strategy_,
            "relative_weights": dict(zip(self._model_names, self.relative_weights_.tolist())),
            "sparsity": dict(self.sparsity_),
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
            "hyperparameters": self.hyperparameters_,
            "soft_gating": False,
            "solver_diagnostics": dict(self.diagnostics_),
        }

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
