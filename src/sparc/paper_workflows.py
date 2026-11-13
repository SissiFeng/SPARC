from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from .acquisition import (
    diversity_filter,
    trust_weighted_confidence_bound,
    trust_weighted_thompson,
)
from .uncertainty import resolve_reference_scale, trust_score

__all__ = [
    "CandidateRanking",
    "rank_fig3_candidates",
    "rank_fig4_batch",
    "rank_campaign_candidates",
]


@dataclass(frozen=True)
class CandidateRanking:
    indices: np.ndarray
    mean: np.ndarray
    uncertainty: np.ndarray
    residual_scale: float
    trust: np.ndarray
    acquisition: np.ndarray
    rule: str
    maximize: bool
    reference_scale: float = 1.0

    def top(self, n_select: int) -> np.ndarray:
        if n_select < 1:
            raise ValueError("n_select must be positive")
        if n_select > len(self.indices):
            raise ValueError("not enough eligible candidates for the requested batch")
        return self.indices[:n_select].copy()

    def to_frame(self, candidate_ids: Optional[np.ndarray] = None) -> pd.DataFrame:
        n = len(self.mean)
        if candidate_ids is None:
            candidate_ids = np.arange(n)
        candidate_ids = np.asarray(candidate_ids)
        if len(candidate_ids) != n:
            raise ValueError("candidate_ids must align with the prediction arrays")
        rank = np.full(n, np.nan)
        rank[self.indices] = np.arange(1, len(self.indices) + 1)
        frame = pd.DataFrame(
            {
                "candidate_id": candidate_ids,
                "predicted_mean": self.mean,
                "epistemic_uncertainty": self.uncertainty,
                "residual_scale": self.residual_scale,
                "trust": self.trust,
                "acquisition_score": self.acquisition,
                "acquisition_rank": rank,
                "eligible_for_batch": np.isfinite(rank),
                "acquisition_rule": self.rule,
                "maximize": self.maximize,
                "reference_scale": self.reference_scale,
            }
        )
        return frame.sort_values("acquisition_rank").reset_index(drop=True)


def _model_prediction_and_trust(
    model: object, X_candidates: np.ndarray, y_train, reference_scale="training_mad"
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
    if hasattr(model, "predict_with_uncertainty"):
        mean, uncertainty = model.predict_with_uncertainty(X_candidates)
    else:
        mean, uncertainty = model.predict(X_candidates)
    residual_scale = getattr(model, "residual_scale_", None)
    if residual_scale is None:
        residual_scale = getattr(model, "aleatoric_", None)
    if residual_scale is None or not np.isfinite(residual_scale):
        raise ValueError(
            "model must expose a finite held-out residual_scale_ or aleatoric_"
        )
    mean = np.asarray(mean, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    if mean.shape != (len(X_candidates),) or uncertainty.shape != mean.shape:
        raise ValueError("model predictions must be one-dimensional and aligned with candidates")
    if not np.isfinite(mean).all():
        raise ValueError("model predictions must be finite")
    q = resolve_reference_scale(reference_scale, y_train)
    trust = trust_score(uncertainty, float(residual_scale), q)
    return (mean, uncertainty, float(residual_scale), trust, q)


def _ordered_indices(
    X_candidates: np.ndarray, acquisition: np.ndarray, diversity_radius: Optional[float]
) -> np.ndarray:
    order = np.argsort(acquisition, kind="mergesort")[::-1]
    if diversity_radius is None or diversity_radius <= 0:
        return order.astype(int)
    selected = diversity_filter(
        X_candidates,
        acquisition,
        n_select=len(acquisition),
        radius=float(diversity_radius),
        ascending=False,
    )
    return np.asarray(selected, dtype=int)


def _confidence_bound_ranking(
    model: object,
    X_candidates: np.ndarray,
    maximize: bool,
    beta: float,
    diversity_radius: Optional[float],
    rule: str,
    y_train,
    reference_scale="training_mad",
) -> CandidateRanking:
    X_candidates = np.asarray(X_candidates, dtype=np.float64)
    mean, uncertainty, residual_scale, trust, q = _model_prediction_and_trust(
        model, X_candidates, y_train, reference_scale
    )
    acquisition = trust_weighted_confidence_bound(
        mean, uncertainty, trust, beta=beta, maximize=maximize
    )
    return CandidateRanking(
        indices=_ordered_indices(X_candidates, acquisition, diversity_radius),
        mean=mean,
        uncertainty=uncertainty,
        residual_scale=residual_scale,
        trust=trust,
        acquisition=acquisition,
        rule=rule,
        maximize=maximize,
        reference_scale=q,
    )


def rank_fig3_candidates(
    bootstrap_model: object,
    X_candidates: np.ndarray,
    maximize: bool,
    y_train,
    beta: float = 1.96,
    reference_scale="training_mad",
) -> CandidateRanking:
    return _confidence_bound_ranking(
        bootstrap_model,
        X_candidates,
        maximize=maximize,
        beta=beta,
        diversity_radius=None,
        rule="fig3_trust_weighted_confidence_bound",
        y_train=y_train,
        reference_scale=reference_scale,
    )


def rank_fig4_batch(
    gp_expert_model: object,
    X_candidates: np.ndarray,
    n_select: int,
    maximize: bool,
    y_train,
    random_state: Optional[int] = None,
    diversity_radius: Optional[float] = None,
    reference_scale="training_mad",
) -> CandidateRanking:
    if n_select < 1:
        raise ValueError("n_select must be positive")
    X_candidates = np.asarray(X_candidates, dtype=np.float64)
    mean, uncertainty, residual_scale, trust, q = _model_prediction_and_trust(
        gp_expert_model, X_candidates, y_train, reference_scale
    )
    draw = gp_expert_model.sample_joint(X_candidates, random_state=random_state)
    acquisition = trust_weighted_thompson(mean, trust, draw, maximize=maximize)
    ranking = CandidateRanking(
        indices=_ordered_indices(X_candidates, acquisition, diversity_radius),
        mean=mean,
        uncertainty=uncertainty,
        residual_scale=residual_scale,
        trust=trust,
        acquisition=acquisition,
        rule="fig4_trust_modulated_joint_thompson",
        maximize=maximize,
        reference_scale=q,
    )
    if n_select > len(ranking.indices):
        raise ValueError("n_select cannot exceed the number of candidates")
    return ranking


def rank_campaign_candidates(
    bootstrap_model: object,
    X_candidates: np.ndarray,
    n_select: int,
    y_train,
    maximize: bool = False,
    beta: float = 1.96,
    diversity_radius: Optional[float] = 0.15,
    reference_scale="training_mad",
) -> CandidateRanking:
    if n_select < 1:
        raise ValueError("n_select must be positive")
    ranking = _confidence_bound_ranking(
        bootstrap_model,
        X_candidates,
        maximize=maximize,
        beta=beta,
        diversity_radius=diversity_radius,
        rule="campaign_trust_weighted_confidence_bound",
        y_train=y_train,
        reference_scale=reference_scale,
    )
    if n_select > len(ranking.indices):
        raise ValueError("n_select cannot exceed the number of candidates")
    return ranking
