from __future__ import annotations
import numpy as np

__all__ = [
    "total_uncertainty",
    "trust_score",
    "response_scale",
    "resolve_reference_scale",
    "trust_summary",
]

REFERENCE_SCALE_RULES = ("training_mad",)


def total_uncertainty(
    epistemic: np.ndarray, aleatoric: float | np.ndarray
) -> np.ndarray:
    epistemic = np.asarray(epistemic, dtype=np.float64)
    aleatoric = np.asarray(aleatoric, dtype=np.float64)
    if any((not np.isfinite(a).all() or np.any(a < 0) for a in (epistemic, aleatoric))):
        raise ValueError("uncertainty terms must be finite and non-negative")
    return np.sqrt(epistemic**2 + aleatoric**2)


def response_scale(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("response scale requires at least two finite responses")
    scale = 1.4826 * float(np.median(np.abs(y - np.median(y))))
    if scale <= 1e-12:
        scale = float(np.std(y))
    if scale <= 1e-12:
        raise ValueError("training responses have no spread; reference scale undefined")
    return scale


def resolve_reference_scale(reference_scale, y_train) -> float:
    if isinstance(reference_scale, str):
        if reference_scale not in REFERENCE_SCALE_RULES:
            raise ValueError(
                f"reference_scale must be positive or one of {REFERENCE_SCALE_RULES}"
            )
        return response_scale(y_train)
    q = float(reference_scale)
    if not np.isfinite(q) or q <= 0:
        raise ValueError("reference_scale must be finite and strictly positive")
    return q


def trust_score(
    epistemic: np.ndarray, aleatoric: float | np.ndarray, reference_scale: float = 1.0
) -> np.ndarray:
    q = float(reference_scale)
    if not np.isfinite(q) or q <= 0:
        raise ValueError("reference_scale must be finite and strictly positive")
    u = total_uncertainty(epistemic, aleatoric)
    if np.any(~np.isfinite(u)) or np.any(u < 0):
        raise ValueError("uncertainty terms must be finite")
    return np.clip(1.0 / (1.0 + u / q), 0.0, 1.0)


def trust_summary(trust: np.ndarray, uncertainty: np.ndarray, q: float, beta: float = 1.96):
    trust = np.asarray(trust, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    bonus = beta * uncertainty * trust
    return {
        "reference_scale": float(q),
        "trust_min": float(trust.min()),
        "trust_median": float(np.median(trust)),
        "trust_max": float(trust.max()),
        "bonus_median": float(np.median(bonus)),
        "bonus_max": float(bonus.max()),
        "bonus_bound": float(beta * q),
    }
