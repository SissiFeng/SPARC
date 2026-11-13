from __future__ import annotations
import numpy as np

WEIGHT_STRATEGIES = ("simplex", "global_local")


def validate_weight_strategy(weight_strategy, global_scale):
    if weight_strategy not in WEIGHT_STRATEGIES:
        raise ValueError(f"weight_strategy must be one of {WEIGHT_STRATEGIES}")
    if not np.isfinite(global_scale) or global_scale <= 0:
        raise ValueError("global_scale must be finite and strictly positive")


def compose_weights(rho, scales, local, magnitude, weight_strategy, xp=np):
    if weight_strategy == "global_local":
        return rho * scales * local * magnitude
    if weight_strategy != "simplex":
        raise ValueError(f"weight_strategy must be one of {WEIGHT_STRATEGIES}")
    logits = xp.log(scales) + xp.log(local) + xp.log(magnitude)
    positive = xp.exp(logits - xp.max(logits))
    return positive / xp.sum(positive)


def summarize_coefficients(samples, threshold=0.01):
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or not samples.size:
        raise ValueError("coefficient samples must be a nonempty matrix")
    if not np.isfinite(samples).all() or np.any(samples < 0):
        raise ValueError("coefficient samples must be finite and non-negative")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    total = samples.sum(axis=1)
    if np.any(total <= 0) or not np.isfinite(total).all():
        raise ValueError("every draw must have a finite positive coefficient sum")
    relative = samples / total[:, None]
    effective = 1.0 / np.sum(relative**2, axis=1)
    return {
        "coefficient_threshold": float(threshold),
        "coefficient_mean": samples.mean(axis=0).tolist(),
        "coefficient_interval_90": np.quantile(samples, [0.05, 0.95], axis=0).tolist(),
        "near_zero_fraction_mean": float((samples < threshold).mean()),
        "coefficient_below_threshold_probability": (samples < threshold).mean(axis=0).tolist(),
        "coefficient_sum_mean": float(total.mean()),
        "coefficient_sum_interval_90": np.quantile(total, [0.05, 0.95]).tolist(),
        "relative_weight_mean": relative.mean(axis=0).tolist(),
        "effective_models_mean": float(effective.mean()),
        "effective_models_interval_90": np.quantile(effective, [0.05, 0.95]).tolist(),
        "relative_weight_of_posterior_mean": (samples.mean(axis=0) / total.mean()).tolist(),
    }
