from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
from .preprocessing import Preprocessor

_SATURATION_BOUNDS = ([-np.inf, 0.0, 1e-08], [1.0, np.inf, np.inf])


def _bottleneck_models(seed: int, n_trees: int) -> List[object]:
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )

    return [
        ExtraTreesRegressor(
            n_estimators=n_trees,
            min_samples_leaf=2,
            max_features=0.75,
            random_state=seed,
            n_jobs=1,
        ),
        RandomForestRegressor(
            n_estimators=n_trees,
            min_samples_leaf=2,
            max_features=0.75,
            random_state=seed,
            n_jobs=1,
        ),
        GradientBoostingRegressor(
            n_estimators=n_trees, max_depth=2, loss="huber", random_state=seed
        ),
    ]


def _saturation_curve(
    n_samples: np.ndarray, ceiling: float, amplitude: float, tau: float
) -> np.ndarray:
    n_samples = np.asarray(n_samples, dtype=np.float64)
    n0 = float(np.min(n_samples))
    return ceiling - amplitude * np.exp(-(n_samples - n0) / tau)


def monotone_envelope(values: Sequence[float]) -> np.ndarray:
    return np.maximum.accumulate(np.asarray(values, dtype=np.float64))


def fit_r2_saturation(
    n_train: np.ndarray, r2_mean: np.ndarray, min_amplitude: float = 0.0001
) -> Dict[str, float]:
    from scipy.optimize import curve_fit

    n = np.asarray(n_train, dtype=np.float64)
    r = np.asarray(r2_mean, dtype=np.float64)
    if n.ndim != 1 or r.shape != n.shape:
        raise ValueError("n_train and r2_mean must be aligned 1-dimensional arrays")
    if len(n) < 3:
        raise ValueError("need at least 3 sizes")
    if not np.isfinite(n).all() or not np.isfinite(r).all():
        raise ValueError("learning-curve values must be finite")
    order = np.argsort(n, kind="mergesort")
    n, r = (n[order], r[order])
    if len(np.unique(n)) != len(n):
        raise ValueError("training sizes must be distinct")
    envelope = monotone_envelope(r)
    observed_max = float(np.max(envelope))
    observed_min = float(np.min(envelope))
    p0 = [
        min(0.95, observed_max + 0.03),
        max(0.02, observed_max - observed_min + 0.03),
        max(10.0, 0.4 * float(np.ptp(n))),
    ]
    converged = True
    if np.ptp(envelope) <= min_amplitude:
        return {
            "r2_ceiling": observed_max,
            "A": 0.0,
            "tau": float("nan"),
            "n_min": float(n[0]),
            "n_star": float("nan"),
            "identifiable": False,
            "converged": True,
            "fit_r2": float("nan"),
        }
    try:
        params, _ = curve_fit(
            _saturation_curve,
            n,
            envelope,
            p0=p0,
            bounds=_SATURATION_BOUNDS,
            maxfev=20000,
        )
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "saturation fit failed; no fitted curve was produced"
        ) from exc
    ceiling, amplitude, tau = (float(v) for v in params)
    n_min = float(n[0])
    identifiable = amplitude > float(min_amplitude)
    fitted = _saturation_curve(n, ceiling, amplitude, tau)
    ss_res = float(np.sum((fitted - envelope) ** 2))
    ss_tot = float(np.sum((envelope - envelope.mean()) ** 2))
    return {
        "r2_ceiling": ceiling,
        "A": amplitude,
        "tau": tau,
        "n_min": n_min,
        "n_star": bottleneck_transition_size(n_min, tau)
        if identifiable
        else float("nan"),
        "identifiable": bool(identifiable),
        "converged": bool(converged),
        "form": "R2_ceiling - A*exp(-(N-N_min)/tau), fitted to the monotone envelope",
        "fit_r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def relative_bottleneck_index(
    n_train: np.ndarray, tau: float, n_min: float
) -> np.ndarray:
    n = np.asarray(n_train, dtype=np.float64)
    tau = float(tau)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    return 1.0 - np.exp(-(n - float(n_min)) / tau)


def bottleneck_transition_size(n_min: float, tau: float) -> float:
    tau = float(tau)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    return float(n_min) + tau * float(np.log(2.0))


def bottleneck_learning_curves(
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: Sequence[int],
    n_repeats: int = 20,
    test_fraction: float = 0.2,
    n_trees: int = 60,
    random_state: int = 2026,
) -> pd.DataFrame:
    from sklearn.metrics import r2_score
    from sklearn.pipeline import make_pipeline

    X = np.asarray(X)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X must be 2-dimensional and aligned with 1-dimensional y")
    if not np.isfinite(y).all():
        raise ValueError("y must contain only finite values")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between 0 and 1")
    if int(n_repeats) < 1:
        raise ValueError("n_repeats must be positive")
    sizes = sorted({int(s) for s in train_sizes})
    if len(sizes) < 3:
        raise ValueError("need at least 3 distinct train_sizes")
    n_pool = int(np.floor((1.0 - test_fraction) * len(y)))
    if sizes[0] < 2 or sizes[-1] > n_pool:
        raise ValueError(
            f"train_sizes must lie in [2, {n_pool}] (training pool after the held-out split)"
        )
    if len(y) - n_pool < 2:
        raise ValueError("held-out test set must contain at least 2 rows")
    rows: List[Dict] = []
    predictions_raw, splits = ([], [])
    for repeat in range(int(n_repeats)):
        order = np.random.default_rng(random_state + repeat).permutation(len(y))
        pool_idx, test_idx = (order[:n_pool], order[n_pool:])
        y_test = y[test_idx]
        for n_samples in sizes:
            train_idx = pool_idx[:n_samples]
            predictions = []
            for model in _bottleneck_models(random_state + repeat, int(n_trees)):
                estimator = make_pipeline(Preprocessor(), model)
                estimator.fit(X[train_idx], y[train_idx])
                predictions.append(estimator.predict(X[test_idx]))
            prediction = np.mean(np.vstack(predictions), axis=0)
            predictions_raw.extend(
                (
                    {
                        "repeat": repeat,
                        "n_samples": int(n_samples),
                        "row_index": int(i),
                        "y_true": float(y[i]),
                        "y_pred": float(p),
                    }
                    for i, p in zip(test_idx, prediction)
                )
            )
            splits.append(
                {
                    "repeat": repeat,
                    "n_samples": int(n_samples),
                    "train_indices": train_idx.tolist(),
                    "test_indices": test_idx.tolist(),
                }
            )
            rows.append(
                {
                    "repeat": repeat,
                    "n_samples": int(n_samples),
                    "r2": float(r2_score(y_test, prediction)),
                }
            )
    result = pd.DataFrame(rows)
    result.attrs["predictions"] = predictions_raw
    result.attrs["splits"] = splits
    return result


def summarize_bottleneck(
    curves: pd.DataFrame,
    n_bootstrap: int = 2000,
    random_state: int = 2026,
    increment: float = 10.0,
    interval: float = 0.9,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    required = {"repeat", "n_samples", "r2"}
    if not required.issubset(curves.columns):
        raise ValueError(f"curves must contain columns {sorted(required)}")
    if int(n_bootstrap) < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < interval < 1.0:
        raise ValueError("interval must lie strictly between 0 and 1")
    pivot = curves.pivot(index="repeat", columns="n_samples", values="r2").sort_index(
        axis=1
    )
    if pivot.isna().any().any():
        raise ValueError("every repeat must contain every training size")
    n_values = pivot.columns.to_numpy(dtype=np.float64)
    matrix = pivot.to_numpy(dtype=np.float64)
    mean_r2 = matrix.mean(axis=0)
    fit = fit_r2_saturation(n_values, mean_r2)
    lo_q, hi_q = (0.5 * (1.0 - interval), 0.5 * (1.0 + interval))
    rng = np.random.default_rng(random_state)
    boot_index = np.full((int(n_bootstrap), len(n_values)), np.nan)
    boot_ceiling = np.full(int(n_bootstrap), np.nan)
    boot_tau = np.full(int(n_bootstrap), np.nan)
    failures = 0
    for b in range(int(n_bootstrap)):
        selected = rng.integers(0, len(matrix), size=len(matrix))
        try:
            boot_fit = fit_r2_saturation(n_values, matrix[selected].mean(axis=0))
        except RuntimeError:
            failures += 1
            continue
        if boot_fit["identifiable"]:
            boot_index[b] = relative_bottleneck_index(
                n_values, boot_fit["tau"], boot_fit["n_min"]
            )
        boot_ceiling[b] = boot_fit["r2_ceiling"]
        boot_tau[b] = boot_fit["tau"]
    centre = (
        relative_bottleneck_index(n_values, fit["tau"], fit["n_min"])
        if fit["identifiable"]
        else np.full(len(n_values), np.nan)
    )
    intervals_identifiable = bool(np.isfinite(boot_index).all())
    summary = pd.DataFrame(
        {
            "n_samples": n_values.astype(int),
            "r2_mean": mean_r2,
            "r2_sd": matrix.std(axis=0, ddof=1) if len(matrix) > 1 else np.nan,
            "bottleneck_index": centre,
            "bottleneck_ci_low": np.quantile(boot_index, lo_q, axis=0)
            if intervals_identifiable
            else np.nan,
            "bottleneck_ci_high": np.quantile(boot_index, hi_q, axis=0)
            if intervals_identifiable
            else np.nan,
        }
    )
    n_min, n_last = (float(n_values.min()), float(n_values.max()))
    gain_last = float(
        fit["A"]
        * np.exp(-(n_last - n_min) / fit["tau"])
        * (1.0 - np.exp(-float(increment) / fit["tau"]))
    )
    diagnostics = {
        "r2_ceiling": fit["r2_ceiling"],
        "r2_ceiling_ci_low": float(np.quantile(boot_ceiling, lo_q)),
        "r2_ceiling_ci_high": float(np.quantile(boot_ceiling, hi_q)),
        "amplitude": fit["A"],
        "tau": fit["tau"],
        "tau_ci_low": float(np.quantile(boot_tau, lo_q)),
        "tau_ci_high": float(np.quantile(boot_tau, hi_q)),
        "n_min": n_min,
        "n_switch": fit["n_star"],
        "identifiable": fit["identifiable"],
        "converged": fit["converged"],
        "marginal_r2_gain_next_increment": gain_last,
        "increment": float(increment),
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_failed": failures,
        "bootstrap_unidentifiable": int(np.isnan(boot_tau).sum()) - failures,
        "intervals_identifiable": intervals_identifiable,
        "interval": float(interval),
    }
    return (summary, diagnostics)


def representation_bottleneck_analysis(
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: Sequence[int],
    n_repeats: int = 20,
    test_fraction: float = 0.2,
    n_bootstrap: int = 2000,
    n_trees: int = 60,
    random_state: int = 2026,
) -> Dict[str, object]:
    curves = bottleneck_learning_curves(
        X,
        y,
        train_sizes=train_sizes,
        n_repeats=n_repeats,
        test_fraction=test_fraction,
        n_trees=n_trees,
        random_state=random_state,
    )
    summary, diagnostics = summarize_bottleneck(
        curves, n_bootstrap=n_bootstrap, random_state=random_state
    )
    diagnostics.update(
        {
            "n_total": int(len(y)),
            "n_repeats": int(n_repeats),
            "test_fraction": float(test_fraction),
            "n_trees": int(n_trees),
            "ensemble": "ExtraTrees + RandomForest + GradientBoosting (mean)",
        }
    )
    return {"curves": curves, "summary": summary, "diagnostics": diagnostics}
