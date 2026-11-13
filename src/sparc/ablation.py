from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut, KFold
from .composition import evidence_matrix, stability_screen, solve_weights_bayes
from .models import build_library, FAMILY_MAP
from .preprocessing import Preprocessor
from .figure_io import check_sampling, compact_diagnostics, metrics_frame, write_json


def altered_posterior(P, y, mode, seed, family_scales):
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.diagnostics import effective_sample_size, gelman_rubin
    from numpyro.infer import MCMC, NUTS, init_to_value
    from jax.scipy.special import logsumexp

    m = P.shape[1]
    scales = jnp.asarray(family_scales)
    if mode not in {"dense", "signed"}:
        raise ValueError("unknown posterior variant")
    if m == 1:
        return (np.ones(1), {"method": "analytic_single_model", "converged": True})
    signs = jnp.asarray(
        [[1, *tail] for tail in itertools.product((-1, 1), repeat=m - 1)]
    )

    def model(P_, y_):
        sigma_sq = numpyro.sample("sigma_sq", dist.InverseGamma(1.0, 1.0))
        if mode == "dense":
            magnitude = numpyro.sample("z", dist.HalfNormal(jnp.ones(m)))
            coefficient = scales * magnitude
            weights = coefficient / coefficient.sum()
            numpyro.sample("y", dist.Normal(P_ @ weights, jnp.sqrt(sigma_sq)), obs=y_)
        else:
            local = numpyro.sample("local", dist.HalfCauchy(jnp.ones(m)))
            magnitude = numpyro.sample("z", dist.HalfNormal(jnp.ones(m)))
            u = jax.nn.softmax(jnp.log(scales) + jnp.log(local) + jnp.log(magnitude))
            signed = signs * u[None, :]
            denominator = signed.sum(axis=1)
            denominator = jnp.where(
                jnp.abs(denominator) < 1e-12,
                jnp.where(denominator < 0, -1e-12, 1e-12),
                denominator,
            )
            alternatives = signed / denominator[:, None]
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
            "divergences": divergences,
            "warmup": 1000 * (retry + 1),
        }
        attempts.append(diagnostic)
        if (
            np.isfinite(w).all()
            and np.isfinite(rhat).all()
            and np.isfinite(ess).all()
            and (rhat < 1.01).all()
            and (ess > 400).all()
            and (divergences == 0)
        ):
            mean = w.reshape(-1, m).mean(axis=0)
            assert np.isclose(mean.sum(), 1, atol=2e-05)
            return (
                mean,
                {
                    "method": mode,
                    "converged": True,
                    "attempts": attempts,
                    "sign_integration": "exact" if mode == "signed" else None,
                },
            )
    raise RuntimeError(f"{mode} posterior did not converge: {attempts}")


def run_ablation(X, y, ids, out, *, seed, definitions, family_priors=None):
    required = {
        "sparse_removal": "fixed_local_scales",
        "nonnegative_removal": "signed_affine",
        "nonnested": "fit_and_score_same_oof",
    }
    if any((definitions.get(k) != v for k, v in required.items())):
        raise ValueError(f"explicit ablation definitions required: {required}")
    k = definitions.get("nonnested_folds")
    if not isinstance(k, int) or not 2 <= k <= len(y):
        raise ValueError("explicit nonnested_folds between 2 and n required")
    library = build_library(seed)
    names = [name for name, _ in library]
    scales = np.array(
        [(family_priors or {}).get(FAMILY_MAP[name], 1.0) for name in names]
    )
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("family scales must be finite and positive")
    predictions, details = ([], [])

    def posterior(P, target, c):
        w, _, diagnostics = solve_weights_bayes(
            P, target, random_seed=seed, family_scales=c
        )
        check_sampling(diagnostics)
        return (w, compact_diagnostics(diagnostics))

    for fold, (tr, te) in enumerate(LeaveOneOut().split(X)):
        if len(tr) >= 20:
            active, frequency = stability_screen(
                library,
                X[tr],
                y[tr],
                n_trials=50,
                threshold=0.6,
                subsample_ratio=0.8,
                inner_splits="loo",
                random_state=seed,
                transformer_factory=Preprocessor,
            )
        else:
            active, frequency = (np.ones(len(library), bool), np.ones(len(library)))
        P = evidence_matrix(
            library, X[tr], y[tr], inner_splits="loo", transformer_factory=Preprocessor
        )
        transform = Preprocessor().fit(X[tr])
        Xt, Xv = (transform.transform(X[tr]), transform.transform(X[te]))
        Q = np.column_stack(
            [clone(model).fit(Xt, y[tr]).predict(Xv) for _, model in library]
        )
        w, full_diag = posterior(P[:, active], y[tr], scales[active])
        current = {"Full SPARC": (Q[:, active] @ w, full_diag)}
        for label, mode in [
            ("No Sparse Selection", "dense"),
            ("No Non-negativity", "signed"),
        ]:
            altered, diagnostic = altered_posterior(
                P[:, active], y[tr], mode, seed, scales[active]
            )
            current[label] = (
                Q[:, active] @ altered,
                {**diagnostic, "weights": altered},
            )
        if np.allclose(scales[active], scales[active][0], rtol=0, atol=0):
            current["No Family Priors"] = (
                current["Full SPARC"][0].copy(),
                {
                    "same_as_full": True,
                    "reason": "common family scale cancels under normalization",
                },
            )
        else:
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
                    "diagnostics": diagnostic,
                    "full_weights": w,
                }
            )
        pd.DataFrame(predictions).to_csv(out / "predictions.partial.csv", index=False)
        write_json(out / "folds.json", details)
        print(f"ablation outer LOO {fold + 1}/{len(y)}", flush=True)
    frame = pd.DataFrame(predictions)
    frame.to_csv(out / "predictions.csv", index=False)
    metrics = metrics_frame(frame)
    full = float(metrics.loc[metrics.method == "Full SPARC", "r2"].iloc[0])
    metrics["delta_r2_percent"] = (
        100 * (metrics.r2 - full) / full if full != 0 else np.nan
    )
    metrics.to_csv(out / "ablation_metrics.csv", index=False)
    (out / "predictions.partial.csv").unlink()
    folds = list(KFold(k, shuffle=True, random_state=0).split(X))
    P = evidence_matrix(library, X, y, inner_splits=k, transformer_factory=Preprocessor)
    if len(y) >= 20:
        active, _ = stability_screen(
            library,
            X,
            y,
            n_trials=50,
            threshold=0.6,
            subsample_ratio=0.8,
            inner_splits=k,
            random_state=seed,
            transformer_factory=Preprocessor,
        )
    else:
        active = np.ones(len(library), bool)
    w, diagnostic = posterior(P[:, active], y, scales[active])
    biased = pd.DataFrame(
        {
            "method": "Non-nested k-fold",
            "row_index": np.arange(len(y)),
            "row_id": ids,
            "y_true": y,
            "y_pred": P[:, active] @ w,
            "evaluation": "same evidence used for fitting composition and scoring",
        }
    )
    biased.to_csv(out / "nonnested_predictions.csv", index=False)
    metric = metrics_frame(biased)
    metric["delta_r2_percent"] = (
        100 * (metric.r2 - full) / full if full != 0 else np.nan
    )
    metric["evaluation"] = "biased_diagnostic_not_outer_heldout"
    metric.to_csv(out / "nonnested_metric.csv", index=False)
    write_json(
        out / "nonnested_fit.json",
        {
            "folds": [{"train": tr, "test": te} for tr, te in folds],
            "diagnostics": diagnostic,
            "definitions": definitions,
        },
    )
