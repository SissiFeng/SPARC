from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(spec, base):
    path = (Path(base) / spec["data"]).resolve()
    frame = pd.read_csv(path)
    allowed = {
        "data",
        "target",
        "target_unit",
        "features",
        "row_id",
        "excluded_outcomes",
    }
    if set(spec) - allowed:
        raise ValueError(f"unknown dataset keys: {sorted(set(spec) - allowed)}")
    target, features = (spec["target"], list(spec["features"]))
    if not spec.get("target_unit"):
        raise ValueError("target_unit is required; trust depends on response units")
    if not features or len(set(features)) != len(features):
        raise ValueError("features must be an explicit nonempty list of unique columns")
    forbidden = {target, spec.get("row_id"), *spec.get("excluded_outcomes", [])}
    if forbidden.intersection(features):
        raise ValueError("target, row ID and other outcomes cannot be input features")
    if not set(features + [target]).issubset(frame.columns):
        raise ValueError("missing feature or target columns")
    ids = (
        frame[spec["row_id"]].to_numpy()
        if spec.get("row_id")
        else np.arange(len(frame))
    )
    if pd.isna(ids).any() or not pd.Index(ids).is_unique:
        raise ValueError("row IDs must be unique and nonmissing")
    X = frame[features].to_numpy()
    y = frame[target].to_numpy(dtype=float)
    if len(y) < 3 or not np.isfinite(y).all():
        raise ValueError(
            "at least three finite targets required; rows are never silently dropped"
        )
    for col in features:
        if (
            pd.api.types.is_numeric_dtype(frame[col])
            and np.isinf(frame[col].to_numpy(dtype=float)).any()
        ):
            raise ValueError(f"infinite input values in {col}")
    return (
        frame,
        X,
        y,
        ids,
        {"path": str(path), "sha256": sha256(path), "rows": len(y), **spec},
    )


def metrics_frame(predictions):
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    rows = []
    for method, group in predictions.groupby("method", sort=False):
        y, p = (group["y_true"], group["y_pred"])
        if not np.isfinite(p).all():
            raise ValueError(f"nonfinite predictions for {method}")
        if group["row_index"].duplicated().any():
            raise ValueError(f"duplicate test predictions for {method}")
        rows.append(
            {
                "method": method,
                "n": len(y),
                "r2": r2_score(y, p),
                "mae": mean_absolute_error(y, p),
                "rmse": np.sqrt(mean_squared_error(y, p)),
            }
        )
    return pd.DataFrame(rows)


def compact_diagnostics(diagnostics):
    return {
        k: v
        for k, v in diagnostics.items()
        if k not in {"weight_samples", "sigma2_samples"}
    }


def validate_family_priors(priors):
    from .models import FAMILIES

    if priors is None:
        return
    if not isinstance(priors, dict) or set(priors) - set(FAMILIES):
        raise ValueError(f"family_priors must use family names {FAMILIES}")
    values = np.array(list(priors.values()), dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("family scales must be finite and positive")


def check_sampling(diagnostics):
    if diagnostics.get("converged") is False:
        raise RuntimeError("posterior did not converge")
    attempts = diagnostics.get("sampling_attempts", diagnostics.get("attempts", []))
    if attempts and attempts[-1].get("divergences", 0):
        raise RuntimeError(
            "final posterior run has divergent transitions; result not exported"
        )
