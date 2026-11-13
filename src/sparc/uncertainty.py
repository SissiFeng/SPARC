from __future__ import annotations
import numpy as np

__all__ = ["total_uncertainty", "trust_score"]


def total_uncertainty(
    epistemic: np.ndarray, aleatoric: float | np.ndarray
) -> np.ndarray:
    epistemic = np.asarray(epistemic, dtype=np.float64)
    aleatoric = np.asarray(aleatoric, dtype=np.float64)
    if any((not np.isfinite(a).all() or np.any(a < 0) for a in (epistemic, aleatoric))):
        raise ValueError("uncertainty terms must be finite and non-negative")
    return np.sqrt(epistemic**2 + aleatoric**2)


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
