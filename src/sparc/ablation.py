from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut, KFold
from .composition import evidence_matrix, stability_screen, solve_weights_bayes, _sampling_converged
from .models import build_library, tune_library, FAMILY_MAP
from .preprocessing import Preprocessor
from .workflow import SCREENING_SETTINGS
from .figure_io import check_sampling, compact_diagnostics, metrics_frame, write_json
from .weighting import validate_weight_strategy


def altered_posterior(P, y, mode, seed, family_scales, weight_strategy="simplex", global_scale=1.0):
    validate_weight_strategy(weight_strategy, global_scale)
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.diagnostics import effective_sample_size, gelman_rubin
    from numpyro.infer import MCMC, NUTS, init_to_value
    from jax.scipy.special import logsumexp

    m = P.shape[1]
    scales = jnp.asarray(family_scales)
    if mode not in {"dense", "signed", "no_global", "no_local"}:
        raise ValueError("unknown posterior variant")
    if mode in {"no_global", "no_local"} and weight_strategy != "global_local":
        raise ValueError("separate global/local ablations require global_local")
    if m == 1 and weight_strategy == "simplex":
        return (np.ones(1), {"method": "analytic_single_model", "converged": True})
    signs = jnp.asarray(
        [[1, *tail] for tail in itertools.product((-1, 1), repeat=m - 1)]
        if weight_strategy == "simplex"
        else list(itertools.product((-1, 1), repeat=m))
    )

    def model(P_, y_):
        sigma_sq = numpyro.sample("sigma_sq", dist.InverseGamma(1.0, 1.0))
        if mode != "signed":
            magnitude = numpyro.sample("z", dist.HalfNormal(jnp.ones(m)))
            coefficient = scales * magnitude
            if mode == "no_global":
                local = numpyro.sample("local", dist.HalfCauchy(jnp.ones(m)))
                coefficient = coefficient * local
            elif mode == "no_local":
                rho = numpyro.sample("rho", dist.HalfCauchy(global_scale))
                coefficient = rho * coefficient
            weights = coefficient / coefficient.sum() if weight_strategy == "simplex" else coefficient
            numpyro.sample("y", dist.Normal(P_ @ weights, jnp.sqrt(sigma_sq)), obs=y_)
        else:
            local = numpyro.sample("local", dist.HalfCauchy(jnp.ones(m)))
            magnitude = numpyro.sample("z", dist.HalfNormal(jnp.ones(m)))
            if weight_strategy == "simplex":
                u = jax.nn.softmax(jnp.log(scales) + jnp.log(local) + jnp.log(magnitude))
                signed = signs * u[None, :]
                denominator = signed.sum(axis=1)
                denominator = jnp.where(
                    jnp.abs(denominator) < 1e-12,
                    jnp.where(denominator < 0, -1e-12, 1e-12),
                    denominator,
                )
                alternatives = signed / denominator[:, None]
            else:
                rho = numpyro.sample("rho", dist.HalfCauchy(global_scale))
                alternatives = signs * (rho * scales * local * magnitude)[None, :]
            residual = y_[:, None] - P_ @ alternatives.T
            log_likelihood = -0.5 * jnp.sum(
                residual**2 / sigma_sq + jnp.log(2 * jnp.pi * sigma_sq), axis=0
            )
            numpyro.factor(
                "marginal_y", logsumexp(log_likelihood) - jnp.log(len(signs))
            )
            probabilities = jax.nn.softmax(log_likelihood)
            weights = probabilities @ alternatives
        numpyro.deterministic("weights", weights)

    attempts = []
    for retry in (0, 1):
        sampler = MCMC(
            NUTS(
                model,
                target_accept_prob=0.95,
                init_strategy=init_to_value(
                    values={"z": jnp.ones(m), "local": jnp.ones(m)}
                ),
            ),
            num_warmup=1000 * (retry + 1),
            num_samples=2000,
            num_chains=4,
            progress_bar=False,
        )
        sampler.run(
            jax.random.PRNGKey(seed + retry),
            P_=jnp.asarray(P, dtype=jnp.float32),
            y_=jnp.asarray(y, dtype=jnp.float32),
        )
        draws = sampler.get_samples(group_by_chain=True)
        w = np.asarray(draws["weights"])
        rhat = np.asarray(gelman_rubin(w))
        ess = np.asarray(effective_sample_size(w))
        divergences = int(np.asarray(sampler.get_extra_fields()["diverging"]).sum())
        diagnostic = {
            "rhat": rhat.tolist(),
            "ess": ess.tolist(),
            "noise_rhat": float(gelman_rubin(np.asarray(draws["sigma_sq"]))),
            "noise_ess": float(effective_sample_size(np.asarray(draws["sigma_sq"]))),
            "divergences": divergences,
            "warmup": 1000 * (retry + 1),
        }
        if "rho" in draws:
            diagnostic["rho_rhat"] = float(gelman_rubin(np.asarray(draws["rho"])))
            diagnostic["rho_ess"] = float(effective_sample_size(np.asarray(draws["rho"])))
        scale_converged = True
        if "rho" in draws:
            local_rhat = np.asarray(gelman_rubin(np.asarray(draws["local"]))) if "local" in draws else [1.0]
            local_ess = np.asarray(effective_sample_size(np.asarray(draws["local"]))) if "local" in draws else [1000.0]
            diagnostic["local_rhat"] = np.asarray(local_rhat).tolist()
            diagnostic["local_ess"] = np.asarray(local_ess).tolist()
            scale_converged = _sampling_converged(local_rhat, local_ess, divergences, diagnostic["rho_rhat"], diagnostic["rho_ess"])
        attempts.append(diagnostic)
        if np.isfinite(w).all() and scale_converged and _sampling_converged(
            rhat, ess, divergences, diagnostic["noise_rhat"], diagnostic["noise_ess"]
        ):
            mean = w.reshape(-1, m).mean(axis=0)
            if weight_strategy == "simplex" and not np.isclose(mean.sum(), 1, atol=2e-05):
                raise RuntimeError("simplex ablation weights must sum to one")
            return (
                mean,
                {
                    "method": mode,
                    "weight_strategy": weight_strategy,
                    "global_scale": float(global_scale),
                    "converged": True,
                    "attempts": attempts,
                    "sign_integration": "exact" if mode == "signed" else None,
                },
            )
    raise RuntimeError(f"{mode} posterior did not converge: {attempts}")


def _screen(library, X, y, inner_splits, seed, n_jobs):
    if len(y) >= SCREENING_SETTINGS["min_rows"]:
        return stability_screen(
            library,
            X,
            y,
            n_trials=SCREENING_SETTINGS["n_trials"],
            threshold=SCREENING_SETTINGS["threshold"],
            subsample_ratio=SCREENING_SETTINGS["subsample_ratio"],
            inner_splits=inner_splits,
            random_state=seed,
            transformer_factory=Preprocessor,
            lasso_alpha=SCREENING_SETTINGS["lasso_alpha"],
            n_jobs=n_jobs,
        )
    return (np.ones(len(library), bool), np.ones(len(library)))


def run_ablation(X, y, ids, out, *, seed, definitions, family_priors=None, n_jobs=1, weight_strategy="simplex", global_scale=1.0):
    validate_weight_strategy(weight_strategy, global_scale)
    required = {
        "sparse_removal": "fixed_local_scales",
        "nonnegative_removal": "signed_affine",
        "nonnested": "kfold_composition_fitted_on_all_rows",
    }
    if weight_strategy == "global_local":
        required.update({"sparse_removal": "fixed_global_and_local_scales", "nonnegative_removal": "signed_unnormalized"})
    if any((definitions.get(k) != v for k, v in required.items())):
        raise ValueError(f"explicit ablation definitions required: {required}")
    k = definitions.get("nonnested_folds")
    if not isinstance(k, int) or not 2 <= k <= len(y):
        raise ValueError("explicit nonnested_folds between 2 and n required")
    base_library = build_library(seed)
    names = [name for name, _ in base_library]
    scales = np.array(
        [(family_priors or {}).get(FAMILY_MAP[name], 1.0) for name in names]
    )
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("family scales must be finite and positive")
    family_variant = not np.allclose(scales, scales[0], rtol=0, atol=0)
    predictions, details, outer_Q = ([], [], {})

    def posterior(P, target, c):
        w, _, diagnostics = solve_weights_bayes(
            P, target, random_seed=seed, family_scales=c,
            weight_strategy=weight_strategy, global_scale=global_scale
        )
        check_sampling(diagnostics)
        return (w, compact_diagnostics(diagnostics))

    for fold, (tr, te) in enumerate(LeaveOneOut().split(X)):
        library, hyperparameters = tune_library(
            base_library, X[tr], y[tr], transformer_factory=Preprocessor, random_state=seed
        )
        active, frequency = _screen(library, X[tr], y[tr], "loo", seed, n_jobs)
        P = evidence_matrix(
            library,
            X[tr],
            y[tr],
            inner_splits="loo",
            transformer_factory=Preprocessor,
            n_jobs=n_jobs,
        )
        transform = Preprocessor().fit(X[tr])
        Xt, Xv = (transform.transform(X[tr]), transform.transform(X[te]))
        Q = np.column_stack(
            [clone(model).fit(Xt, y[tr]).predict(Xv) for _, model in library]
        )
        outer_Q[int(te[0])] = Q[0]
        w, full_diag = posterior(P[:, active], y[tr], scales[active])
        current = {"Full SPARC": (Q[:, active] @ w, full_diag)}
        variants = [("No Sparse Selection", "dense"), ("No Non-negativity", "signed")]
        if weight_strategy == "global_local":
            variants += [("No Global Shrinkage", "no_global"), ("No Local Shrinkage", "no_local")]
        for label, mode in variants:
            altered, diagnostic = altered_posterior(
                P[:, active], y[tr], mode, seed, scales[active],
                weight_strategy=weight_strategy, global_scale=global_scale
            )
            current[label] = (
                Q[:, active] @ altered,
                {**diagnostic, "weights": altered},
            )
        if family_variant:
            w_family, diagnostic = posterior(P[:, active], y[tr], np.ones(active.sum()))
            current["No Family Priors"] = (
                Q[:, active] @ w_family,
                {**diagnostic, "weights": w_family},
            )
        if active.all():
            current["No Stability Screening"] = (
                current["Full SPARC"][0].copy(),
                {
                    "same_as_full": True,
                    "reason": "full support already retains every model",
                },
            )
        else:
            w_all, diagnostic = posterior(P, y[tr], scales)
            current["No Stability Screening"] = (
                Q @ w_all,
                {**diagnostic, "weights": w_all},
            )
        best = int(np.argmin(np.mean((P - y[tr, None]) ** 2, axis=0)))
        current["Best Single Model"] = (Q[:, best], {"chosen_model": names[best]})
        for method, (pred, diagnostic) in current.items():
            predictions.append(
                {
                    "method": method,
                    "fold": fold,
                    "row_index": int(te[0]),
                    "row_id": ids[te[0]],
                    "y_true": float(y[te[0]]),
                    "y_pred": float(pred[0]),
                    "evaluation": "nested_outer_loo",
                }
            )
            details.append(
                {
                    "method": method,
                    "fold": fold,
                    "train_indices": tr,
                    "test_indices": te,
                    "active_models": np.asarray(names)[active],
                    "selection_frequency": frequency,
                    "hyperparameters": hyperparameters,
                    "diagnostics": diagnostic,
                    "full_weights": w,
                }
            )
        pd.DataFrame(predictions).to_csv(out / "predictions.partial.csv", index=False)
        write_json(out / "folds.json", details)
        print(f"ablation outer LOO {fold + 1}/{len(y)}", flush=True)
    library, hyperparameters = tune_library(
        base_library, X, y, transformer_factory=Preprocessor, random_state=seed
    )
    P = evidence_matrix(
        library, X, y, inner_splits=k, transformer_factory=Preprocessor, n_jobs=n_jobs
    )
    active, _ = _screen(library, X, y, k, seed, n_jobs)
    w, diagnostic = posterior(P[:, active], y, scales[active])
    for row in range(len(y)):
        predictions.append(
            {
                "method": "Non-nested k-fold",
                "fold": row,
                "row_index": row,
                "row_id": ids[row],
                "y_true": float(y[row]),
                "y_pred": float(outer_Q[row][active] @ w),
                "evaluation": "outer_loo_base_models_composition_fitted_on_all_rows",
            }
        )
    frame = pd.DataFrame(predictions)
    frame.to_csv(out / "predictions.csv", index=False)
    metrics = metrics_frame(frame)
    full = float(metrics.loc[metrics.method == "Full SPARC", "r2"].iloc[0])
    metrics["delta_r2_percent"] = (
        100 * (metrics.r2 - full) / full if full != 0 else np.nan
    )
    metrics.to_csv(out / "ablation_metrics.csv", index=False)
    (out / "predictions.partial.csv").unlink()
    write_json(
        out / "nonnested_fit.json",
        {
            "folds": [
                {"train": tr, "test": te}
                for tr, te in KFold(k, shuffle=True, random_state=0).split(X)
            ],
            "active_models": np.asarray(names)[active],
            "weights": w,
            "hyperparameters": hyperparameters,
            "diagnostics": diagnostic,
            "definitions": definitions,
            "family_prior_variant": "computed" if family_variant else "not defined for uniform family scales",
        },
    )
