from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, LeaveOneOut
from .weighting import compose_weights, summarize_coefficients, validate_weight_strategy

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
    lasso_alpha=0.01,
    n_jobs=1,
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
            n_jobs=n_jobs,
        )
        scale = 1.4826 * np.median(np.abs(y[indices] - np.median(y[indices])))
        if scale <= 1e-12:
            scale = max(float(np.std(y[indices])), 1.0)
        support = Lasso(alpha=float(lasso_alpha), positive=True, max_iter=10000).fit(
            P / scale, y[indices] / scale
        )
        selected[trial] = support.coef_ > 1e-05
    frequencies = selected.mean(axis=0)
    active = frequencies >= threshold
    if not active.any():
        active[np.argsort(frequencies)[-min(5, m) :]] = True
    return (active, frequencies)


def importance_loo_predictive(P, y, weight_samples, sigma2_samples):
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
    predictions = W @ P.T
    cap = 1.0 / np.sqrt(len(W))
    result = np.empty(len(y))
    for i in range(len(y)):
        ratios = 0.5 * np.log(variance) + (y[i] - predictions[:, i]) ** 2 / (
            2 * variance
        )
        if not np.isfinite(ratios).all():
            raise FloatingPointError("inverse likelihood overflow")
        ratios = np.exp(ratios - ratios.max())
        ratios /= ratios.sum()
        ratios = np.minimum(ratios, cap)
        ratios /= ratios.sum()
        result[i] = ratios @ predictions[:, i]
    return result



def _sampling_converged(rhat, neff, divergences, noise_rhat, noise_ess,
                        rhat_threshold=1.01, neff_threshold=400):
    rhat, neff = np.asarray(rhat), np.asarray(neff)
    return bool(
        np.isfinite(rhat).all()
        and np.isfinite(neff).all()
        and (rhat < rhat_threshold).all()
        and (neff > neff_threshold).all()
        and np.isfinite(noise_rhat)
        and np.isfinite(noise_ess)
        and noise_rhat < rhat_threshold
        and noise_ess > neff_threshold
        and divergences == 0
    )


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
    weight_strategy: str = "simplex",
    global_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    validate_weight_strategy(weight_strategy, global_scale)
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
    if m == 1 and weight_strategy == "simplex":
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
                "weight_strategy": weight_strategy,
                "global_scale": float(global_scale),
                "global_shrinkage_effective": False,
                "sparsity": summarize_coefficients(np.ones((n_draws, 1))),
            },
        )

    def model(P_, y_, scales_):
        rho = numpyro.sample("rho", dist.HalfCauchy(scale=global_scale))
        local_unit = numpyro.sample("local_unit", dist.HalfCauchy(scale=jnp.ones(m)))
        magnitude_unit = numpyro.sample("magnitude_unit", dist.HalfNormal(jnp.ones(m)))
        tau = numpyro.deterministic("tau", rho * scales_ * local_unit)
        numpyro.deterministic("alpha", tau * magnitude_unit)
        weights = numpyro.deterministic(
            "weights", compose_weights(rho, scales_, local_unit, magnitude_unit, weight_strategy, xp=jnp)
        )
        sigma_sq = numpyro.sample(
            "sigma_sq", dist.InverseGamma(concentration=1.0, rate=1.0)
        )
        obs_sd = jnp.sqrt(sigma_sq)
        numpyro.sample("y", dist.Normal(P_ @ weights, obs_sd), obs=y_)

    P_j = jnp.asarray(P, dtype=jnp.float32)
    y_j = jnp.asarray(y, dtype=jnp.float32)
    scales_j = jnp.asarray(scales, dtype=jnp.float32)
    attempts = []
    posterior_scales = {}

    def run(warmup: int, seed: int):
        mcmc = MCMC(
            NUTS(model, target_accept_prob=target_accept_prob),
            num_warmup=warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=False,
        )
        mcmc.run(
            jax.random.PRNGKey(seed),
            P_=P_j,
            y_=y_j,
            scales_=scales_j,
            extra_fields=("diverging",),
        )
        grouped = mcmc.get_samples(group_by_chain=True)
        chains = np.asarray(grouped["weights"])
        sigma_chains = np.asarray(grouped["sigma_sq"])
        flat = chains.reshape(-1, chains.shape[-1])
        sigma_flat = sigma_chains.reshape(-1)
        rho_chains = np.asarray(grouped["rho"])
        local_chains = np.asarray(grouped["local_unit"])
        posterior_scales.update({
            "rho": rho_chains.reshape(-1),
            "local": local_chains.reshape(-1, m),
        })
        rhat = np.array(
            [gelman_rubin(chains[:, :, j]) for j in range(chains.shape[-1])]
        )
        neff = np.array(
            [effective_sample_size(chains[:, :, j]) for j in range(chains.shape[-1])]
        )
        extra = mcmc.get_extra_fields()
        divergences = int(np.asarray(extra["diverging"]).sum())
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
                "rho_rhat": float(gelman_rubin(rho_chains)),
                "rho_ess": float(effective_sample_size(rho_chains)),
                "local_rhat": np.asarray(gelman_rubin(local_chains)).tolist(),
                "local_ess": np.asarray(effective_sample_size(local_chains)).tolist(),
                "divergences": divergences,
            }
        )
        return (flat, sigma_flat, rhat, neff, divergences)

    def passes(rhat, neff, divergences):
        passed = _sampling_converged(
            rhat, neff, divergences,
            attempts[-1]["noise_rhat"], attempts[-1]["noise_ess"],
            rhat_threshold, neff_threshold,
        )
        if weight_strategy == "global_local":
            passed = passed and _sampling_converged(
                attempts[-1]["local_rhat"], attempts[-1]["local_ess"], divergences,
                attempts[-1]["rho_rhat"], attempts[-1]["rho_ess"],
                rhat_threshold, neff_threshold,
            )
        return passed

    flat, sigma_flat, rhat, neff, divergences = run(num_warmup, random_seed)
    if not passes(rhat, neff, divergences):
        flat, sigma_flat, rhat, neff, divergences = run(num_warmup * 2, random_seed + 1)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    total = mean.sum()
    weights = mean
    if weight_strategy == "simplex":
        weights = mean / total if total > 1e-12 else np.ones_like(mean) / len(mean)
    converged = passes(rhat, neff, divergences)
    if not converged:
        raise RuntimeError(
            f"Bayesian weight sampler did not meet convergence thresholds after retry: {attempts[-1]}. Unconverged draws must not produce reportable benchmark predictions."
        )
    diagnostics = {
        "rhat": rhat.tolist(),
        "neff": neff.tolist(),
        "weight_samples": flat,
        "sigma2_samples": sigma_flat,
        "converged": converged,
        "prior": "saas_halfcauchy_invgamma_1_1_simplex" if weight_strategy == "simplex" else "positive_global_local_halfcauchy_invgamma_1_1",
        "weight_strategy": weight_strategy,
        "global_scale": float(global_scale),
        "global_shrinkage_effective": weight_strategy == "global_local",
        "sparsity": summarize_coefficients(flat),
        "rho_posterior": {
            "median": float(np.median(posterior_scales["rho"])),
            "interval_90": np.quantile(posterior_scales["rho"], [0.05, 0.95]).tolist(),
            "informs_likelihood": weight_strategy == "global_local",
        },
        "local_scale_posterior": {
            "median": np.median(posterior_scales["local"], axis=0).tolist(),
            "interval_90": np.quantile(posterior_scales["local"], [0.05, 0.95], axis=0).tolist(),
        },
        "family_scales": scales.tolist(),
        "sampling_attempts": attempts,
        "convergence_rule": {
            "rhat_below": float(rhat_threshold),
            "ess_above": int(neff_threshold),
            "divergences": 0,
            "retry_warmup": int(num_warmup * 2),
        },
        "parameterization": "noncentred_magnitudes_signs_marginalized_rho_cancelled" if weight_strategy == "simplex" else "noncentred_positive_global_local_unnormalized",
    }
    return (weights, std, diagnostics)
