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

TASK_KEYS = {
    "benchmark": {"methods", "family_priors", "adapters"},
    "feature_strategies": {"policy_rounds"},
    "ablation": {"family_priors", "definitions"},
    "learning_curve": {"train_sizes", "trees"},
    "finite_pool": {"task", "initial_ids", "n_evaluations"},
    "extrapolation": {"task", "maximize"},
    "campaign": {"candidates", "candidate_id"},
    "verification": {
        "oof_predictions",
        "oof_method",
        "nonconformity",
        "priority_ids",
        "n_controls",
    },
    "repeatability": {"cohort", "repeats"},
    "campaign_outcomes": {"recommendations", "measurements", "threshold"},
}


def run(config, base, out):
    workflow = config["workflow"]
    if workflow not in TASK_KEYS:
        raise ValueError(f"workflow must be one of {list(TASK_KEYS)}")
    allowed = TASK_KEYS[workflow] | {"workflow", "dataset", "seed"}
    if set(config) - allowed:
        raise ValueError(f"unknown configuration keys: {sorted(set(config) - allowed)}")
    validate_family_priors(config.get("family_priors"))
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
            )
        elif workflow == "feature_strategies":
            from .benchmark import run_feature_strategies

            state["protocol"] = {
                "outer": "loo",
                "reward_cv": 5,
                "delta": 0.1,
                "policy_rounds": config["policy_rounds"],
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
                gp_reference_model,
                rf_reference_model,
            )

            tasks = {"ru_dissolution": (20, False), "polymer_her": (10, True)}
            n_initial, maximize = tasks[config["task"]]
            initial = pd.Index(ids).get_indexer(config["initial_ids"])
            if (
                len(initial) != n_initial
                or (initial < 0).any()
                or len(set(initial)) != n_initial
            ):
                raise ValueError(
                    f"{config['task']} requires {n_initial} unique known initial IDs"
                )
            summary = []
            for name, factory, use_trust in [
                ("SPARC", bootstrap_model, True),
                ("GP", gp_reference_model, False),
                ("RF", rf_reference_model, False),
            ]:
                trajectory = run_finite_pool_search(
                    X,
                    y,
                    initial,
                    maximize,
                    model_factory=factory,
                    use_trust=use_trust,
                    beta=1.96,
                    reference_scale=1.0,
                    n_evaluations=config.get("n_evaluations"),
                    random_state=seed,
                )
                trajectory["row_id"] = ids[trajectory.candidate_index]
                trajectory.to_csv(out / f"trajectory_{name}.csv", index=False)
                optimum = float(np.max(y) if maximize else np.min(y))
                summary.append(
                    {
                        "method": name,
                        "discovery_evaluation": discovery_count(
                            trajectory, optimum, maximize
                        ),
                        "pool_optimum": optimum,
                        "relative_tolerance": 0.001,
                    }
                )
            pd.DataFrame(summary).to_csv(out / "discovery_counts.csv", index=False)
            state["protocol"] = {
                "bootstrap": 150,
                "beta": 1.96,
                "q": 1.0,
                "n_initial": n_initial,
                "maximize": maximize,
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
            result = run_controlled_extrapolation(
                X, y, mask, batch_size, 4, maximize, seed
            )
            result.rounds["row_id"] = ids[result.rounds.candidate_index]
            result.rounds.to_csv(out / "acquisition_rounds.csv", index=False)
            precision_at_k(result.batches, y[~mask], maximize).to_csv(
                out / "precision_at_k.csv", index=False
            )
            batch_hit_rates(result.batches, y[~mask], maximize).to_csv(
                out / "batch_hit_rates.csv", index=False
            )
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
            for row in result.rounds.itertuples():
                regions.loc[row.candidate_index, "round"] = row.round
            regions.to_csv(out / "projection_coordinates.csv", index=False)
            regions.groupby("round").target.agg(["count", "mean"]).to_csv(
                out / "round_target_means.csv"
            )
            state["protocol"] = {
                "batch_size": batch_size,
                "rounds": 4,
                "expert_cv": 3,
                "q": 1.0,
                "thompson_joint_draw": "independent standard normal per candidate",
                "initial_indices": np.flatnonzero(mask),
                "expert_weights": result.expert_weights,
                "uncertainty_ratio": result.uncertainty_ratio,
            }
        elif workflow == "campaign":
            from .acquisition import BootstrapEnsemble
            from .paper_workflows import rank_campaign_candidates

            candidates = read_table("candidates")
            candidate_ids = candidates[config["candidate_id"]].to_numpy()
            if pd.isna(candidate_ids).any() or not pd.Index(candidate_ids).is_unique:
                raise ValueError("candidate IDs must be unique and nonmissing")
            if config["candidate_id"] in config["dataset"]["features"]:
                raise ValueError("candidate ID cannot be a feature")
            if len(y) != 125:
                raise ValueError("paper campaign starts with 125 labelled observations")
            XC = candidates[config["dataset"]["features"]].to_numpy(dtype=float)
            model = BootstrapEnsemble(n_bootstrap=150, random_state=seed).fit(X, y)
            ranking = rank_campaign_candidates(
                model,
                XC,
                n_select=11,
                maximize=False,
                beta=1.96,
                diversity_radius=0.15,
                reference_scale=1.0,
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
                    "oob_prediction": model.oob_predictions_,
                    "oob_count": model.oob_counts_,
                }
            ).to_csv(out / "oob_predictions.csv", index=False)
            state["protocol"] = {
                "bootstrap": 150,
                "beta": 1.96,
                "q": 1.0,
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
            sigma = oof.epistemic_sd.to_numpy() if mode == "residual_over_sd" else None
            if sigma is not None and (
                not np.isfinite(sigma).all() or (sigma <= 0).any()
            ):
                raise ValueError(
                    "normalized nonconformity requires positive OOF standard deviations"
                )
            report = quality_report(
                X, y, oof.y_pred.to_numpy(), sigma, random_state=seed
            )
            report["row_id"] = ids[report.idx]
            report.to_csv(out / "verification_flags.csv", index=False)
            cohort = matched_cohort(
                report, X, ids, config["priority_ids"], config["n_controls"], seed
            )
            cohort.to_csv(out / "fixed_cohort.csv", index=False)
            state["protocol"] = {
                "conformal_p": 0.1,
                "influence_percentile": 90,
                "isolation_contamination": 0.1,
                "mahalanobis_p": 0.05,
                "priority_min_flags": 2,
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
