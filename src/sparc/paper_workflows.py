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
from .uncertainty import trust_score

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
            }
        )
        return frame.sort_values("acquisition_rank").reset_index(drop=True)


def _model_prediction_and_trust(
    model: object, X_candidates: np.ndarray, reference_scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
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
    trust = trust_score(uncertainty, float(residual_scale), reference_scale)
    return (mean, uncertainty, float(residual_scale), trust)


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
    reference_scale: float = 1.0,
) -> CandidateRanking:
    X_candidates = np.asarray(X_candidates, dtype=np.float64)
    mean, uncertainty, residual_scale, trust = _model_prediction_and_trust(
        model, X_candidates, reference_scale
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
    )


def rank_fig3_candidates(
    bootstrap_model: object,
    X_candidates: np.ndarray,
    maximize: bool,
    beta: float = 1.96,
    reference_scale: float = 1.0,
) -> CandidateRanking:
    return _confidence_bound_ranking(
        bootstrap_model,
        X_candidates,
        maximize=maximize,
        beta=beta,
        diversity_radius=None,
        rule="fig3_trust_weighted_confidence_bound",
        reference_scale=reference_scale,
    )


def rank_fig4_batch(
    gp_expert_model: object,
    X_candidates: np.ndarray,
    n_select: int,
    maximize: bool,
    random_state: Optional[int] = None,
    diversity_radius: Optional[float] = None,
    reference_scale: float = 1.0,
) -> CandidateRanking:
    if n_select < 1:
        raise ValueError("n_select must be positive")
    X_candidates = np.asarray(X_candidates, dtype=np.float64)
    mean, uncertainty, residual_scale, trust = _model_prediction_and_trust(
        gp_expert_model, X_candidates, reference_scale
    )
    acquisition = trust_weighted_thompson(
        mean, uncertainty, trust, maximize=maximize, random_state=random_state
    )
    ranking = CandidateRanking(
        indices=_ordered_indices(X_candidates, acquisition, diversity_radius),
        mean=mean,
        uncertainty=uncertainty,
        residual_scale=residual_scale,
        trust=trust,
        acquisition=acquisition,
        rule="fig4_trust_modulated_thompson",
        maximize=maximize,
    )
    if n_select > len(ranking.indices):
        raise ValueError("n_select cannot exceed the number of candidates")
    return ranking


def rank_campaign_candidates(
    bootstrap_model: object,
    X_candidates: np.ndarray,
    n_select: int,
    maximize: bool = False,
    beta: float = 1.96,
    diversity_radius: Optional[float] = 0.15,
    reference_scale: float = 1.0,
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
        reference_scale=reference_scale,
    )
    if n_select > len(ranking.indices):
        raise ValueError("n_select cannot exceed the number of candidates")
    return ranking
