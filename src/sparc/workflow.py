from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from .cache import EvidenceCache
from .composition import (
    evidence_matrix,
    importance_loo_predictive,
    solve_weights_bayes,
    stability_screen,
)
from .figure_io import check_sampling
from .fusion import STRATEGY_NAMES, FeatureFusion, select_feature_strategy
from .models import FAMILY_MAP, build_library, tune_library
from .preprocessing import Preprocessor

__all__ = [
    "RepresentationPlan",
    "ModelComposition",
    "Planner",
    "Modeller",
    "Agent",
    "SCREENING_SETTINGS",
]

SCREENING_SETTINGS = {
    "n_trials": 50,
    "subsample_ratio": 0.8,
    "threshold": 0.6,
    "lasso_alpha": 0.01,
    "min_rows": 20,
    "fallback_models": 5,
}


@dataclass
class RepresentationPlan:
    strategy: Optional[str]
    transformer_factory: Callable[[], Pipeline]
    bandit_pulls: Optional[np.ndarray] = None
    bandit_rewards: Optional[np.ndarray] = None


@dataclass
class ModelComposition:
    library_names: List[str]
    active_names: List[str]
    active_mask: np.ndarray
    selection_frequencies: np.ndarray
    hyperparameters: Dict[str, dict]
    evidence: np.ndarray
    weights: np.ndarray
    weight_samples: np.ndarray
    oof_predictions: np.ndarray
    residual_scale: float
    transformer: Pipeline
    models: Dict[str, object]
    diagnostics: Dict
    evidence_cache_hit: bool = False
    family_weights: Dict[str, float] = field(default_factory=dict)


class Planner:
    def __init__(self, feature_strategy=None, feature_n_rounds=None, random_state=42):
        self.feature_strategy = feature_strategy
        self.feature_n_rounds = feature_n_rounds
        self.random_state = random_state

    def plan(self, X, y) -> RepresentationPlan:
        pulls = rewards = None
        if self.feature_strategy == "auto":
            chosen, pulls, rewards = select_feature_strategy(
                X,
                y,
                n_rounds=self.feature_n_rounds,
                cv_splits=5,
                delta=0.1,
                random_state=self.random_state,
            )
        else:
            chosen = self.feature_strategy
            if chosen is not None and chosen not in STRATEGY_NAMES:
                raise ValueError(
                    f"feature_strategy must be None, 'auto' or one of {STRATEGY_NAMES}"
                )
        random_state = self.random_state

        def factory():
            steps = [("preprocess", Preprocessor())]
            if chosen is not None:
                steps.append(
                    ("fusion", FeatureFusion(strategy=chosen, random_state=random_state))
                )
            return Pipeline(steps)

        return RepresentationPlan(chosen, factory, pulls, rewards)


class Modeller:
    def __init__(
        self,
        family_priors=None,
        random_state=42,
        model_names=None,
        n_jobs=1,
        evidence_cache=None,
        tune_hyperparameters=True,
        weight_strategy="simplex",
        global_scale=1.0,
    ):
        self.family_priors = family_priors
        self.random_state = random_state
        self.model_names = model_names
        self.n_jobs = n_jobs
        self.evidence_cache = evidence_cache
        self.tune_hyperparameters = tune_hyperparameters
        self.weight_strategy = weight_strategy
        self.global_scale = global_scale

    def library(self):
        library = build_library(self.random_state)
        if self.model_names is not None:
            names = list(self.model_names)
            available = {name for name, _ in library}
            if not names or len(set(names)) != len(names) or set(names) - available:
                raise ValueError(
                    "model_names must be distinct members of the manuscript library"
                )
            library = [(name, model) for name, model in library if name in names]
        return library

    def fit(self, X, y, plan: RepresentationPlan) -> ModelComposition:
        factory = plan.transformer_factory
        library = self.library()
        if self.tune_hyperparameters:
            library, hyperparameters = tune_library(
                library, X, y, transformer_factory=factory, random_state=self.random_state
            )
        else:
            hyperparameters = {name: {} for name, _ in library}
        if len(y) >= SCREENING_SETTINGS["min_rows"]:
            active, frequencies = stability_screen(
                library,
                X,
                y,
                n_trials=SCREENING_SETTINGS["n_trials"],
                subsample_ratio=SCREENING_SETTINGS["subsample_ratio"],
                threshold=SCREENING_SETTINGS["threshold"],
                inner_splits="loo",
                random_state=self.random_state,
                transformer_factory=factory,
                lasso_alpha=SCREENING_SETTINGS["lasso_alpha"],
                n_jobs=self.n_jobs,
            )
        else:
            active = np.ones(len(library), dtype=bool)
            frequencies = np.ones(len(library))
        retained = [pair for pair, keep in zip(library, active) if keep]

        def compute():
            return evidence_matrix(
                retained,
                X,
                y,
                inner_splits="loo",
                transformer_factory=factory,
                n_jobs=self.n_jobs,
            )

        cache_hit = False
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
                    "schema": 5,
                    "strategy": plan.strategy,
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
            P, cache_hit = cache.get_or_compute(key, compute)
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
            weight_strategy=self.weight_strategy,
            global_scale=self.global_scale,
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
                raise RuntimeError(f"retained model {name} failed its final refit") from exc
        names = [name for name, _ in library]
        full_weights = np.zeros(len(library))
        full_weights[active] = weights
        residual = y - oof
        family_weights: Dict[str, float] = {}
        for name, weight in zip(names, full_weights):
            family = FAMILY_MAP[name]
            family_weights[family] = family_weights.get(family, 0.0) + float(weight)
        monotonic = {
            name: model.monotonic_cst_.tolist()
            for name, model in fitted.items()
            if hasattr(model, "monotonic_cst_")
        }
        diagnostics = {
            **diagnostics,
            "retained_model_names": [name for name, _ in retained],
            "n_screened_out": int((~active).sum()),
            "residual_method": "truncated_importance_sampling_loo_sqrt_S",
            "screening": dict(SCREENING_SETTINGS),
            "hyperparameters": hyperparameters,
            "monotonic_constraints": monotonic,
            "monotonic_rule": "spearman_abs_rho_at_least_0.3_on_training_state",
        }
        return ModelComposition(
            library_names=names,
            active_names=[name for name, _ in retained],
            active_mask=active,
            selection_frequencies=frequencies,
            hyperparameters=hyperparameters,
            evidence=P,
            weights=full_weights,
            weight_samples=samples,
            oof_predictions=oof,
            residual_scale=float(1.4826 * np.median(np.abs(residual - np.median(residual)))),
            transformer=transformer,
            models=fitted,
            diagnostics=diagnostics,
            evidence_cache_hit=bool(cache_hit),
            family_weights=family_weights,
        )


class Agent:
    def __init__(
        self,
        rule="confidence_bound",
        maximize=True,
        beta=1.96,
        reference_scale="training_mad",
        diversity_radius=None,
    ):
        if rule not in {"confidence_bound", "thompson"}:
            raise ValueError("rule must be confidence_bound or thompson")
        self.rule = rule
        self.maximize = maximize
        self.beta = beta
        self.reference_scale = reference_scale
        self.diversity_radius = diversity_radius

    def recommend(self, model, X_candidates, y_train, n_select, random_state=None):
        from .paper_workflows import rank_campaign_candidates, rank_fig4_batch

        if self.rule == "thompson":
            ranking = rank_fig4_batch(
                model,
                X_candidates,
                n_select=n_select,
                maximize=self.maximize,
                y_train=y_train,
                random_state=random_state,
                diversity_radius=self.diversity_radius,
                reference_scale=self.reference_scale,
            )
        else:
            ranking = rank_campaign_candidates(
                model,
                X_candidates,
                n_select=n_select,
                y_train=y_train,
                maximize=self.maximize,
                beta=self.beta,
                diversity_radius=self.diversity_radius,
                reference_scale=self.reference_scale,
            )
        return ranking
