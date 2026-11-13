from __future__ import annotations
import logging
from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, LeaveOneOut

logger = logging.getLogger(__name__)
__all__ = [
    "evidence_matrix",
    "stability_screen",
    "solve_weights_bayes",
    "importance_loo_predictive",
]


def _splitter(inner_splits, n_samples):
    if inner_splits == "loo":
        return LeaveOneOut()
    if not isinstance(inner_splits, int) or not 2 <= inner_splits <= n_samples:
        raise ValueError("inner_splits must be 'loo' or an integer between 2 and n")
    return KFold(n_splits=inner_splits, shuffle=True, random_state=0)


def evidence_matrix(
    models, X, y, inner_splits="loo", transformer_factory=None, n_jobs=1
):
    X, y = (np.asarray(X), np.asarray(y, dtype=float))
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or (len(y) < 3) or (not models):
        raise ValueError(
            "aligned X and y, at least three rows and a nonempty library required"
        )
    if not np.isfinite(y).all():
        raise ValueError("targets must be finite")
    if len({name for name, _ in models}) != len(models):
        raise ValueError("model names must be unique")
    folds = list(_splitter(inner_splits, len(y)).split(X))

    def fit_fold(train, test):
        Xt, Xv = (X[train], X[test])
        if transformer_factory is not None:
            transform = transformer_factory()
            Xt = transform.fit_transform(Xt, y[train])
            Xv = transform.transform(Xv)
        values = np.empty((len(test), len(models)))
        for j, (name, template) in enumerate(models):
            try:
                prediction = np.asarray(clone(template).fit(Xt, y[train]).predict(Xv))
                if (
                    prediction.shape != (len(test),)
                    or not np.isfinite(prediction).all()
                ):
                    raise ValueError("invalid prediction")
                values[:, j] = prediction
            except Exception as exc:
                raise RuntimeError(
                    f"evidence fit failed for {name}, test rows {test.tolist()}"
                ) from exc
        return (test, values)

    if int(n_jobs) == 1:
        results = [fit_fold(train, test) for train, test in folds]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=int(n_jobs), prefer="processes")(
            (delayed(fit_fold)(train, test) for train, test in folds)
        )
    evidence = np.empty((len(y), len(models)))
    for test, values in results:
        evidence[test] = values
    return evidence


def stability_screen(
    models,
    X,
    y,
    n_trials=50,
    subsample_ratio=0.8,
    threshold=0.6,
    inner_splits="loo",
    random_state=42,
    transformer_factory=None,
):
    X, y = (np.asarray(X), np.asarray(y, dtype=float))
    n, m = (len(y), len(models))
    if n < 20:
        return (np.ones(m, dtype=bool), np.ones(m))
    if int(n_trials) != n_trials or n_trials < 1:
        raise ValueError("n_trials must be a positive integer")
    if not 0 < subsample_ratio <= 1 or not 0 <= threshold <= 1:
        raise ValueError("invalid subsample ratio or threshold")
    rng = np.random.RandomState(random_state)
    selected = np.zeros((int(n_trials), m), dtype=bool)
    for trial in range(int(n_trials)):
        indices = rng.choice(n, size=int(n * subsample_ratio), replace=False)
        P = evidence_matrix(
            models,
            X[indices],
            y[indices],
            inner_splits=inner_splits,
            transformer_factory=transformer_factory,
        )
        scale = 1.4826 * np.median(np.abs(y[indices] - np.median(y[indices])))
        if scale <= 1e-12:
            scale = max(float(np.std(y[indices])), 1.0)
        support = Lasso(alpha=0.01, positive=True, max_iter=10000).fit(
            P / scale, y[indices] / scale
        )
        selected[trial] = support.coef_ > 1e-05
    frequencies = selected.mean(axis=0)
    active = frequencies >= threshold
    if not active.any():
        active[np.argsort(frequencies)[-min(5, m) :]] = True
    return (active, frequencies)


def importance_loo_predictive(P, y, weight_samples, sigma2_samples, max_rel_weight=0.1):
    P, y = (np.asarray(P, dtype=float), np.asarray(y, dtype=float))
    W, variance = (
        np.asarray(weight_samples, dtype=float),
        np.asarray(sigma2_samples, dtype=float),
    )
    if P.ndim != 2 or y.shape != (len(P),) or W.ndim != 2 or (W.shape[1] != P.shape[1]):
        raise ValueError("evidence, targets and posterior draws must align")
    if variance.shape != (len(W),) or not len(W) or np.any(variance <= 0):
        raise ValueError("one strictly positive variance per posterior draw required")
    if not all((np.isfinite(a).all() for a in (P, y, W, variance))):
        raise ValueError("importance-sampling inputs must be finite")
    if not 0 < max_rel_weight <= 1:
        raise ValueError("max_rel_weight must lie in (0, 1]")
    predictions = W @ P.T
    result = np.empty(len(y))
    for i in range(len(y)):
        ratios = 0.5 * np.log(variance) + (y[i] - predictions[:, i]) ** 2 / (
            2 * variance
        )
        if not np.isfinite(ratios).all():
            raise FloatingPointError("inverse likelihood overflow")
        ratios = np.exp(ratios - ratios.max())
        ratios /= ratios.sum()
        ratios = np.minimum(ratios, max_rel_weight)
        ratios /= ratios.sum()
        result[i] = ratios @ predictions[:, i]
    return result


def solve_weights_bayes(
    P: np.ndarray,
    y: np.ndarray,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
    target_accept_prob: float = 0.85,
    rhat_threshold: float = 1.01,
    neff_threshold: int = 400,
    random_seed: int = 0,
    family_scales: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    try:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.diagnostics import effective_sample_size, gelman_rubin
        from numpyro.infer import MCMC, NUTS
    except Exception as exc:
        raise ImportError(
            "Install the pinned requirements, including JAX and NumPyro."
        ) from exc
    y_np = np.asarray(y, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or y_np.ndim != 1 or P.shape[0] != len(y_np) or (P.shape[1] == 0):
        raise ValueError("P and y must align with at least one model")
    if not np.isfinite(P).all() or not np.isfinite(y_np).all():
        raise ValueError("evidence and targets must be finite")
    m = P.shape[1]
    if family_scales is None:
        scales = np.ones(m, dtype=np.float64)
    else:
        scales = np.asarray(family_scales, dtype=np.float64)
        if scales.shape != (m,):
            raise ValueError(f"family_scales must have shape {(m,)}")
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("family_scales must be finite and strictly positive")
    if m == 1:
        from scipy.stats import invgamma

        n_draws = int(num_samples) * int(num_chains)
        if n_draws < 1:
            raise ValueError("num_samples and num_chains must be positive")
        sigma_flat = invgamma.rvs(
            a=1.0 + len(y_np) / 2.0,
            scale=1.0 + 0.5 * np.sum((y_np - P[:, 0]) ** 2),
            size=n_draws,
            random_state=random_seed,
        )
        return (
            np.ones(1),
            np.zeros(1),
            {
                "rhat": [],
                "neff": [],
                "converged": True,
                "method": "analytic_single_model",
                "weight_samples": np.ones((n_draws, 1)),
                "sigma2_samples": sigma_flat,
                "prior": "saas_halfcauchy_invgamma_1_1_simplex",
                "family_scales": scales.tolist(),
            },
        )

    def model(P_, y_, scales_):
        rho = numpyro.sample("rho", dist.HalfCauchy(scale=1.0))
        local_unit = numpyro.sample("local_unit", dist.HalfCauchy(scale=jnp.ones(m)))
        magnitude_unit = numpyro.sample("magnitude_unit", dist.HalfNormal(jnp.ones(m)))
        tau = numpyro.deterministic("tau", rho * scales_ * local_unit)
        numpyro.deterministic("alpha", tau * magnitude_unit)
        log_alpha = jnp.log(scales_) + jnp.log(local_unit) + jnp.log(magnitude_unit)
        weights = numpyro.deterministic("weights", jax.nn.softmax(log_alpha))
        sigma_sq = numpyro.sample(
            "sigma_sq", dist.InverseGamma(concentration=1.0, rate=1.0)
        )
        obs_sd = jnp.sqrt(sigma_sq)
        numpyro.sample("y", dist.Normal(P_ @ weights, obs_sd), obs=y_)

    P_j = jnp.asarray(P, dtype=jnp.float32)
    y_j = jnp.asarray(y, dtype=jnp.float32)
    scales_j = jnp.asarray(scales, dtype=jnp.float32)
    attempts = []

    def run(warmup: int, seed: int):
        mcmc = MCMC(
            NUTS(model, target_accept_prob=target_accept_prob),
            num_warmup=warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(seed), P_=P_j, y_=y_j, scales_=scales_j)
        grouped = mcmc.get_samples(group_by_chain=True)
        chains = np.asarray(grouped["weights"])
        sigma_chains = np.asarray(grouped["sigma_sq"])
        flat = chains.reshape(-1, chains.shape[-1])
        sigma_flat = sigma_chains.reshape(-1)
        rhat = np.array(
            [gelman_rubin(chains[:, :, j]) for j in range(chains.shape[-1])]
        )
        neff = np.array(
            [effective_sample_size(chains[:, :, j]) for j in range(chains.shape[-1])]
        )
        extra = mcmc.get_extra_fields() if hasattr(mcmc, "get_extra_fields") else {}
        attempts.append(
            {
                "warmup": int(warmup),
                "seed": int(seed),
                "chains": int(num_chains),
                "samples_per_chain": int(num_samples),
                "weight_rhat": rhat.tolist(),
                "weight_ess": neff.tolist(),
                "noise_rhat": float(gelman_rubin(sigma_chains)),
                "noise_ess": float(effective_sample_size(sigma_chains)),
                "divergences": int(np.asarray(extra["diverging"]).sum())
                if "diverging" in extra
                else None,
            }
        )
        return (flat, sigma_flat, rhat, neff)

    def passes(rhat, neff):
        return bool(
            np.isfinite(rhat).all()
            and np.isfinite(neff).all()
            and (rhat < rhat_threshold).all()
            and (neff > neff_threshold).all()
        )

    flat, sigma_flat, rhat, neff = run(num_warmup, random_seed)
    if not passes(rhat, neff):
        flat, sigma_flat, rhat, neff = run(num_warmup * 2, random_seed + 1)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    total = mean.sum()
    weights = mean / total if total > 1e-12 else np.ones_like(mean) / len(mean)
    converged = passes(rhat, neff)
    if not converged:
        raise RuntimeError(
            f"Bayesian weight sampler did not meet convergence thresholds after retry; rhat={rhat.tolist()}, neff={neff.tolist()}. Unconverged draws must not produce reportable benchmark predictions."
        )
    diagnostics = {
        "rhat": rhat.tolist(),
        "neff": neff.tolist(),
        "weight_samples": flat,
        "sigma2_samples": sigma_flat,
        "converged": converged,
        "prior": "saas_halfcauchy_invgamma_1_1_simplex",
        "family_scales": scales.tolist(),
        "sampling_attempts": attempts,
        "parameterization": "noncentred_magnitudes_signs_marginalized_rho_cancelled",
    }
    if any(
        (
            attempt["divergences"]
            for attempt in attempts
            if attempt["divergences"] is not None
        )
    ):
        logger.warning(
            "NUTS reported divergent transitions; inspect sampling_attempts before using these draws"
        )
    return (weights, std, diagnostics)
