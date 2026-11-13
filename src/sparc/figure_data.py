from __future__ import annotations
import argparse
import importlib.metadata
import os
from pathlib import Path
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from .figure_io import load_dataset, write_json, sha256, validate_family_priors
from .weighting import validate_weight_strategy

FINITE_POOL_METHODS = {
    "SPARC": ("bootstrap", True, 1.96),
    "SPARC-UCB": ("bootstrap", False, 1.96),
    "SPARC-greedy": ("bootstrap", True, 0.0),
    "SPARC-Bayes": ("bayesian", True, 1.96),
    "GP": ("gp", True, 1.96),
    "RF": ("rf", True, 1.96),
}

TASK_KEYS = {
    "benchmark": {"methods", "family_priors", "adapters", "repeat_seeds", "n_jobs"},
    "feature_strategies": {"policy_rounds"},
    "ablation": {"family_priors", "definitions", "n_jobs"},
    "learning_curve": {"train_sizes", "trees"},
    "finite_pool": {
        "task",
        "initial_id_sets",
        "n_evaluations",
        "repeat_seeds",
        "reference_scale",
        "methods",
    },
    "extrapolation": {"task", "maximize", "repeat_seeds", "reference_scale"},
    "campaign": {"candidates", "candidate_id", "model", "reference_scale"},
    "verification": {
        "oof_predictions",
        "oof_method",
        "oof_seed",
        "nonconformity",
        "n_priority",
        "priority_ids",
        "additional_control_ids",
        "n_controls",
        "n_residual_baseline",
        "reference_scale",
    },
    "repeatability": {"cohort", "repeats"},
    "campaign_outcomes": {"recommendations", "measurements", "threshold"},
}


for _workflow in ("benchmark", "ablation", "finite_pool", "campaign"):
    TASK_KEYS[_workflow].update({"weight_strategy", "global_scale"})


def run(config, base, out):
    workflow = config["workflow"]
    if workflow not in TASK_KEYS:
        raise ValueError(f"workflow must be one of {list(TASK_KEYS)}")
    allowed = TASK_KEYS[workflow] | {"workflow", "dataset", "seed"}
    if set(config) - allowed:
        raise ValueError(f"unknown configuration keys: {sorted(set(config) - allowed)}")
    validate_family_priors(config.get("family_priors"))
    weight_strategy = config.get("weight_strategy", "simplex")
    global_scale = config.get("global_scale", 1.0)
    validate_weight_strategy(weight_strategy, global_scale)
    if workflow == "campaign" and config.get("model", "bootstrap") != "bayesian" and ({"weight_strategy", "global_scale"} & set(config)):
        raise ValueError("weight_strategy and global_scale require campaign model=bayesian")
    seed = config["seed"]
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("an explicit nonnegative integer seed is required")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    from . import __version__

    state = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "sparc_version": __version__,
        "configuration": config,
        "bayesian_composition": {"weight_strategy": weight_strategy, "global_scale": global_scale},
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "scipy", "pandas", "scikit-learn", "jax", "numpyro"]
        },
        "code_sha256": {
            p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "inputs": {},
    }

    def read_table(key):
        path = (Path(base) / config[key]).resolve()
        state["inputs"][key] = {"path": str(path), "sha256": sha256(path)}
        return pd.read_csv(path)

    write_json(out / "run_parameters.json", state)
    try:
        if workflow not in {"repeatability", "campaign_outcomes"}:
            frame, X, y, ids, data_info = load_dataset(config["dataset"], base)
            state["inputs"]["dataset"] = data_info
            state["row_ids"] = ids
        if workflow == "benchmark":
            from .benchmark import run_benchmark

            state["protocol"] = {
                "outer": "loo",
                "inner": "loo",
                "screening_subsamples": 50,
                "screening_fraction": 0.8,
                "selection_frequency": 0.6,
                "nuts_chains": 4,
                "nuts_warmup": 1000,
                "nuts_draws": 2000,
                "soft_gating": False,
                "feature_policy": False,
                "hyperparameter_selection": "5-fold grid search within every training state",
                "repeat_seeds": config.get("repeat_seeds", [seed]),
            }
            write_json(out / "run_parameters.json", state)
            run_benchmark(
                X,
                y,
                ids,
                out,
                seed=seed,
                methods=config.get("methods"),
                family_priors=config.get("family_priors"),
                adapters=config.get("adapters"),
                repeat_seeds=config.get("repeat_seeds"),
                n_jobs=config.get("n_jobs", 1),
                weight_strategy=weight_strategy,
                global_scale=global_scale,
            )
        elif workflow == "feature_strategies":
            from .benchmark import run_feature_strategies

            state["protocol"] = {
                "outer": "loo",
                "reward_cv": 5,
                "delta": 0.1,
                "policy_rounds": config["policy_rounds"],
                "reward_per_pull": "R2 on a freshly shuffled five-fold split",
            }
            run_feature_strategies(
                X, y, out, seed=seed, n_rounds=config["policy_rounds"]
            )
        elif workflow == "ablation":
            from .ablation import run_ablation

            state["protocol"] = {
                "outer": "loo",
                "inner": "loo",
                "screening_subsamples": 50,
                "nuts_chains": 4,
                "nuts_warmup": 1000,
                "nuts_draws": 2000,
                "definitions": config["definitions"],
            }
            write_json(out / "run_parameters.json", state)
            run_ablation(
                X,
                y,
                ids,
                out,
                seed=seed,
                definitions=config["definitions"],
                family_priors=config.get("family_priors"),
                n_jobs=config.get("n_jobs", 1),
                weight_strategy=weight_strategy,
                global_scale=global_scale,
            )
        elif workflow == "learning_curve":
            from .learning import representation_bottleneck_analysis

            result = representation_bottleneck_analysis(
                X,
                y,
                config["train_sizes"],
                n_repeats=20,
                test_fraction=0.2,
                n_bootstrap=2000,
                n_trees=config["trees"],
                random_state=seed,
            )
            result["curves"].to_csv(out / "learning_curves.csv", index=False)
            result["summary"].to_csv(out / "bottleneck_index.csv", index=False)
            pd.DataFrame(result["curves"].attrs["predictions"]).to_csv(
                out / "predictions.csv", index=False
            )
            write_json(out / "splits.json", result["curves"].attrs["splits"])
            write_json(out / "fitted_parameters.json", result["diagnostics"])
            state["protocol"] = {
                "repeats": 20,
                "test_fraction": 0.2,
                "bootstrap": 2000,
                "interval": 0.9,
                "models": "ExtraTrees,RandomForest,GradientBoosting",
            }
        elif workflow == "finite_pool":
            from .finite_pool import (
                run_finite_pool_search,
                discovery_count,
                bootstrap_model,
                bayesian_model,
                gp_reference_model,
                rf_reference_model,
            )

            factories = {
                "bootstrap": bootstrap_model,
                "bayesian": lambda current: bayesian_model(current, weight_strategy, global_scale),
                "gp": gp_reference_model,
                "rf": rf_reference_model,
            }
            tasks = {"ru_dissolution": (20, False), "polymer_her": (10, True)}
            n_initial, maximize = tasks[config["task"]]
            sets = config["initial_id_sets"]
            if not isinstance(sets, list) or not sets:
                raise ValueError("initial_id_sets must be a nonempty list of ID lists")
            initial_sets = []
            for members in sets:
                initial = pd.Index(ids).get_indexer(members)
                if (
                    len(initial) != n_initial
                    or (initial < 0).any()
                    or len(set(initial)) != n_initial
                ):
                    raise ValueError(
                        f"{config['task']} requires {n_initial} unique known initial IDs per set"
                    )
                initial_sets.append(initial)
            seeds = [int(v) for v in config.get("repeat_seeds", [seed])]
            methods = config.get("methods", list(FINITE_POOL_METHODS))
            if set(methods) - set(FINITE_POOL_METHODS) or len(set(methods)) != len(methods):
                raise ValueError(f"methods must be distinct members of {list(FINITE_POOL_METHODS)}")
            reference = config.get("reference_scale", "training_mad")
            optimum = float(np.max(y) if maximize else np.min(y))
            summary = []
            for set_no, initial in enumerate(initial_sets):
                for current in seeds:
                    for name in methods:
                        kind, use_trust, beta = FINITE_POOL_METHODS[name]
                        trajectory = run_finite_pool_search(
                            X,
                            y,
                            initial,
                            maximize,
                            model_factory=factories[kind],
                            use_trust=use_trust,
                            beta=beta,
                            reference_scale=reference,
                            n_evaluations=config.get("n_evaluations"),
                            random_state=current,
                        )
                        trajectory["row_id"] = ids[trajectory.candidate_index]
                        trajectory.to_csv(
                            out / f"trajectory_{name}_set{set_no}_seed{current}.csv",
                            index=False,
                        )
                        summary.append(
                            {
                                "method": name,
                                "initial_set": set_no,
                                "seed": current,
                                "discovery_evaluation": discovery_count(
                                    trajectory, optimum, maximize
                                ),
                                "pool_optimum": optimum,
                                "relative_tolerance": 0.001,
                            }
                        )
                        pd.DataFrame(summary).to_csv(out / "discovery_counts.csv", index=False)
            frame = pd.DataFrame(summary)
            frame.to_csv(out / "discovery_counts.csv", index=False)
            frame.groupby("method", sort=False).discovery_evaluation.agg(
                runs="size",
                found="count",
                mean="mean",
                median="median",
                q25=lambda v: v.quantile(0.25),
                q75=lambda v: v.quantile(0.75),
            ).to_csv(out / "discovery_summary.csv")
            state["protocol"] = {
                "bootstrap": 150,
                "methods": {name: FINITE_POOL_METHODS[name] for name in methods},
                "reference_scale": reference,
                "n_initial": n_initial,
                "n_initial_sets": len(initial_sets),
                "repeat_seeds": seeds,
                "maximize": maximize,
                "acquisition": "identical trust-modulated confidence bound for every model",
                "counts_include_initialization": True,
            }
        elif workflow == "extrapolation":
            from .extrapolation import (
                central_interval_split,
                run_controlled_extrapolation,
                precision_at_k,
                batch_hit_rates,
            )
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            settings = {
                "formate_oxidation": (214, 92, 23),
                "oer_activity": (406, 56, 14),
            }
            n_initial, n_pool, batch_size = settings[config["task"]]
            mask = central_interval_split(X, 20, 80)
            if int(mask.sum()) != n_initial or int((~mask).sum()) != n_pool:
                raise ValueError(
                    f"20–80% split gives {mask.sum()}/{(~mask).sum()}; paper requires {n_initial}/{n_pool}. Check input variables."
                )
            maximize = config["maximize"]
            if not isinstance(maximize, bool):
                raise ValueError(
                    "maximize must be boolean and match the specified target"
                )
            seeds = [int(v) for v in config.get("repeat_seeds", [seed])]
            reference = config.get("reference_scale", "training_mad")
            rounds_all, precision_all, hits_all, ratios, weights = ([], [], [], [], {})
            for current in seeds:
                result = run_controlled_extrapolation(
                    X,
                    y,
                    mask,
                    batch_size,
                    4,
                    maximize,
                    current,
                    reference_scale=reference,
                )
                result.rounds["row_id"] = ids[result.rounds.candidate_index]
                rounds_all.append(result.rounds.assign(seed=current))
                precision_all.append(
                    precision_at_k(result.batches, y[~mask], maximize).assign(seed=current)
                )
                hits_all.append(
                    batch_hit_rates(result.batches, y[~mask], maximize).assign(seed=current)
                )
                ratios.append({"seed": current, "uncertainty_ratio": result.uncertainty_ratio})
                weights[str(current)] = result.expert_weights
            rounds = pd.concat(rounds_all, ignore_index=True)
            rounds.to_csv(out / "acquisition_rounds.csv", index=False)
            precision = pd.concat(precision_all, ignore_index=True)
            precision.to_csv(out / "precision_at_k.csv", index=False)
            hits = pd.concat(hits_all, ignore_index=True)
            hits.to_csv(out / "batch_hit_rates.csv", index=False)
            pd.DataFrame(ratios).to_csv(out / "uncertainty_ratio.csv", index=False)
            precision.groupby("cutoff_batches").agg(
                fraction_K=("fraction_K", "first"),
                precision_mean=("precision_at_K", "mean"),
                precision_sd=("precision_at_K", "std"),
                runs=("seed", "nunique"),
            ).to_csv(out / "precision_at_k_summary.csv")
            hits.groupby("round").agg(
                hit_rate_mean=("hit_rate", "mean"),
                hit_rate_sd=("hit_rate", "std"),
                runs=("seed", "nunique"),
            ).to_csv(out / "batch_hit_rates_summary.csv")
            coords = PCA(2, svd_solver="full").fit_transform(
                StandardScaler().fit_transform(X)
            )
            regions = pd.DataFrame(
                {
                    "row_id": ids,
                    "PC1": coords[:, 0],
                    "PC2": coords[:, 1],
                    "initial": mask,
                    "target": y,
                    "round": 0,
                }
            )
            first = rounds[rounds.seed == seeds[0]]
            for row in first.itertuples():
                regions.loc[row.candidate_index, "round"] = row.round
            regions.to_csv(out / "projection_coordinates.csv", index=False)
            regions.groupby("round").target.agg(["count", "mean"]).to_csv(
                out / "round_target_means.csv"
            )
            state["protocol"] = {
                "batch_size": batch_size,
                "rounds": 4,
                "expert_cv": 3,
                "expert_weighting": "pseudo-BMA: softmax of summed three-fold CV log predictive densities",
                "reference_scale": reference,
                "thompson_draw": "one joint latent posterior function per round from an expert sampled by its weight",
                "uncertainty": "latent mixture standard deviation excluding the fitted noise kernel",
                "uncertainty_ratio_denominator": "held-out cross-fitted uncertainty in the central region",
                "repeat_seeds": seeds,
                "initial_indices": np.flatnonzero(mask),
                "expert_weights": weights,
                "uncertainty_ratio": ratios,
            }
        elif workflow == "campaign":
            from .acquisition import BootstrapEnsemble
            from .ensemble import SparseEvidenceEnsemble
            from .paper_workflows import rank_campaign_candidates
            from .uncertainty import trust_summary

            candidates = read_table("candidates")
            candidate_ids = candidates[config["candidate_id"]].to_numpy()
            if pd.isna(candidate_ids).any() or not pd.Index(candidate_ids).is_unique:
                raise ValueError("candidate IDs must be unique and nonmissing")
            if config["candidate_id"] in config["dataset"]["features"]:
                raise ValueError("candidate ID cannot be a feature")
            if len(y) != 125:
                raise ValueError("paper campaign starts with 125 labelled observations")
            XC = candidates[config["dataset"]["features"]].to_numpy(dtype=float)
            kind = config.get("model", "bootstrap")
            if kind == "bootstrap":
                model = BootstrapEnsemble(n_bootstrap=150, random_state=seed).fit(X, y)
            elif kind == "bayesian":
                model = SparseEvidenceEnsemble(
                    random_state=seed, weight_strategy=weight_strategy, global_scale=global_scale
                ).fit(X, y)
            else:
                raise ValueError("model must be bootstrap or bayesian")
            if kind == "bayesian":
                write_json(out / "composition_diagnostics.json", model.diagnostics_report())
            reference = config.get("reference_scale", "training_mad")
            ranking = rank_campaign_candidates(
                model,
                XC,
                n_select=11,
                y_train=y,
                maximize=False,
                beta=1.96,
                diversity_radius=0.15,
                reference_scale=reference,
            )
            ranking.to_frame(candidate_ids).to_csv(
                out / "candidate_ranking.csv", index=False
            )
            selection = ranking.top(11)
            selected = candidates.iloc[selection][
                [config["candidate_id"], *config["dataset"]["features"]]
            ].copy()
            selected = selected.rename(columns={config["candidate_id"]: "row_id"})
            selected.to_csv(out / "recommendations.csv", index=False)
            pd.DataFrame(
                {
                    "row_id": ids,
                    "oof_prediction": model.oob_predictions_
                    if kind == "bootstrap"
                    else model.oof_predictions_,
                }
            ).to_csv(out / "oof_predictions.csv", index=False)
            greedy = np.argsort(ranking.mean, kind="mergesort")[: len(selection)]
            state["protocol"] = {
                "model": kind,
                "bootstrap": 150 if kind == "bootstrap" else None,
                "beta": 1.96,
                "reference_scale_rule": reference,
                "trust": trust_summary(
                    ranking.trust, ranking.uncertainty, ranking.reference_scale, 1.96
                ),
                "overlap_with_greedy_mean_top": int(len(set(selection) & set(greedy))),
                "diversity_distance": "0.15 * sqrt(d)",
                "n_select": 11,
                "diversity_standardization": "complete candidate feature pool",
                "candidate_labels_used": False,
                "residual_scale": model.residual_scale_,
            }
        elif workflow == "verification":
            from .verification import quality_report, matched_cohort

            if len(y) != 125:
                raise ValueError("paper verification starts with 125 observations")
            oof = read_table("oof_predictions")
            oof = oof[oof.method == config["oof_method"]].copy()
            if "seed" in oof:
                if "oof_seed" in config:
                    oof = oof[oof.seed == config["oof_seed"]]
                elif oof.seed.nunique() > 1:
                    raise ValueError("oof_seed is required when multiple prediction runs are available")
            if not oof.row_id.is_unique or set(oof.row_id) != set(ids):
                raise ValueError(
                    "OOF IDs must match each initial observation exactly once"
                )
            oof = oof.set_index("row_id").loc[ids]
            if not np.allclose(oof.y_true, y, rtol=0, atol=0):
                raise ValueError("OOF targets do not match the initial measurements")
            mode = config["nonconformity"]
            if mode not in {"absolute_residual", "residual_over_sd"}:
                raise ValueError("specify absolute_residual or residual_over_sd")
            dispersion = oof.epistemic_sd.to_numpy(dtype=float)
            if not np.isfinite(dispersion).all() or (dispersion < 0).any():
                raise ValueError("verification requires finite OOF predictive dispersion")
            sigma = dispersion if mode == "residual_over_sd" else None
            if sigma is not None and (sigma <= 0).any():
                raise ValueError(
                    "normalized nonconformity requires positive OOF standard deviations"
                )
            reference = config.get("reference_scale", "training_mad")
            report = quality_report(
                X,
                y,
                oof.y_pred.to_numpy(),
                sigma,
                trust_uncertainty=dispersion,
                reference_scale=reference,
                random_state=seed,
            )
            report["row_id"] = ids[report.idx]
            report.to_csv(out / "verification_flags.csv", index=False)
            cohort = matched_cohort(
                report,
                X,
                ids,
                config["n_controls"],
                seed,
                n_priority=config.get("n_priority"),
                n_residual_baseline=config.get("n_residual_baseline", 0),
                priority_ids=config.get("priority_ids"),
                additional_control_ids=config.get("additional_control_ids"),
            )
            cohort.to_csv(out / "fixed_cohort.csv", index=False)
            state["protocol"] = {
                "oof_method": config["oof_method"],
                "oof_seed": int(oof.seed.iloc[0]) if "seed" in oof else None,
                "nonconformity": mode,
                "conformal_p": 0.1,
                "conformal_pvalue": "jackknife (1 + #{j != i: s_j >= s_i}) / n on outer OOF residuals",
                "influence_percentile": 90,
                "isolation_contamination": 0.1,
                "mahalanobis_p": 0.05,
                "priority_min_flags": 2,
                "priority_rule": "explicit eligible priority_ids when provided; otherwise all or n_priority ranked eligible records",
                "n_eligible_priority": int((report.n_flags >= 2).sum()),
                "control_rule": "one unflagged nearest neighbour per priority without replacement; any additional controls require explicit IDs",
                "influence_rule": "absolute full-fit Ridge prediction minus supplied outer-OOF ensemble prediction",
                "trust_reported_not_flagged": True,
                "reference_scale": reference,
                "repeat_measurements_used": False,
            }
        elif workflow == "repeatability":
            from .verification import repeatability

            joined, summary = repeatability(read_table("cohort"), read_table("repeats"))
            joined.to_csv(out / "individual_differences.csv", index=False)
            summary.to_csv(out / "group_summary.csv")
        elif workflow == "campaign_outcomes":
            recommendations, measurements = (
                read_table("recommendations"),
                read_table("measurements"),
            )
            if (
                recommendations.row_id.duplicated().any()
                or measurements.row_id.duplicated().any()
            ):
                raise ValueError(
                    "recommendations and measurements require unique row_id"
                )
            if set(recommendations.row_id) != set(measurements.row_id):
                raise ValueError(
                    "measured IDs must match the fixed recommendation list"
                )
            joined = recommendations.merge(
                measurements[["row_id", "measured"]], on="row_id", validate="one_to_one"
            )
            if not np.isfinite(joined.measured).all():
                raise ValueError("measurements must be finite")
            joined.to_csv(out / "measured_recommendations.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "n": len(joined),
                        "below_threshold": int(
                            (joined.measured < config["threshold"]).sum()
                        ),
                        "threshold": config["threshold"],
                        "minimum": float(joined.measured.min()),
                    }
                ]
            ).to_csv(out / "summary.csv", index=False)
        state["status"] = "complete"
        state["outputs_sha256"] = {
            p.name: sha256(p)
            for p in sorted(out.iterdir())
            if p.name != "run_parameters.json"
        }
    except Exception as exc:
        state["status"], state["error"] = ("failed", repr(exc))
        raise
    finally:
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "run_parameters.json", state)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate SPARC numerical source data")
    parser.add_argument("configuration", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
    config = json.loads(args.configuration.read_text())
    run(config, args.configuration.resolve().parent, args.out)


if __name__ == "__main__":
    main()
