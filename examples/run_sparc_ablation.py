from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_NAME = "dataset"
DATA_PATH = Path("data.csv")
SHEET_NAME = 0
TARGET_COLUMN = "target"
TARGET_UNIT = ""
FEATURE_COLUMNS = []
ROW_ID_COLUMN = None
EXCLUDED_OUTCOME_COLUMNS = []
FAMILY_PRIORS = {
    "linear": 1.0,
    "tree_based": 1.0,
    "locally_smooth": 1.0,
    "monotonic": 1.0,
}
SEED = 42
NONNESTED_FOLDS = 5
NONNESTED_DEFINITION = "kfold_composition_fitted_on_all_rows"
N_JOBS = 1
SPARC_SOURCE = None
OUTPUT_ROOT = Path("ablation_results")

METHODS = (
    "Full SPARC",
    "No Sparse Selection",
    "No Non-negativity",
    "No Family Priors",
    "No Stability Screening",
    "Non-nested k-fold",
    "Best Single Model",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_input():
    path = Path(DATA_PATH).expanduser().resolve()
    if not FEATURE_COLUMNS or len(set(FEATURE_COLUMNS)) != len(FEATURE_COLUMNS):
        raise ValueError("Set FEATURE_COLUMNS to an explicit nonempty list of unique input columns.")
    forbidden = {TARGET_COLUMN, ROW_ID_COLUMN, *EXCLUDED_OUTCOME_COLUMNS}
    if forbidden.intersection(FEATURE_COLUMNS):
        raise ValueError("Features cannot include the target, row ID or excluded outcomes.")
    if not TARGET_UNIT:
        raise ValueError("Set TARGET_UNIT, using 'dimensionless' if applicable.")
    dtypes = {ROW_ID_COLUMN: str} if ROW_ID_COLUMN else None
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype=dtypes, float_precision="round_trip")
    elif path.suffix.lower() in {".xlsx", ".xlsm"}:
        frame = pd.read_excel(path, sheet_name=SHEET_NAME, dtype=dtypes)
    else:
        raise ValueError("DATA_PATH must be a CSV or XLSX/XLSM file.")
    needed = {*FEATURE_COLUMNS, TARGET_COLUMN}
    if ROW_ID_COLUMN:
        needed.add(ROW_ID_COLUMN)
    if not frame.columns.is_unique or not needed.issubset(frame.columns):
        raise ValueError(f"Missing or duplicate input columns: {sorted(needed - set(frame.columns))}")
    y = pd.to_numeric(frame[TARGET_COLUMN], errors="raise").to_numpy(dtype=float)
    if len(y) < 3 or not np.isfinite(y).all() or np.ptp(y) == 0:
        raise ValueError("At least three finite, nonconstant targets are required; no rows are dropped.")
    if not isinstance(NONNESTED_FOLDS, int) or not 2 <= NONNESTED_FOLDS <= len(y):
        raise ValueError("NONNESTED_FOLDS must be between 2 and the number of observations.")
    ids = frame[ROW_ID_COLUMN] if ROW_ID_COLUMN else pd.Series(np.arange(len(y)))
    if ids.isna().any():
        raise ValueError("Row IDs cannot be missing.")
    ids = ids.astype(str).to_numpy()
    if not pd.Index(ids).is_unique:
        raise ValueError("Row IDs must be unique.")
    features = frame[FEATURE_COLUMNS]
    for column in FEATURE_COLUMNS:
        if pd.api.types.is_numeric_dtype(features[column]):
            if np.isinf(features[column].to_numpy(dtype=float)).any():
                raise ValueError(f"Infinite input values in {column}.")
    return path, features.to_numpy(), y, ids


def assemble_outputs(raw, y, ids, family_variant, definitions):
    from sparc.figure_io import metrics_frame

    def read_predictions(path):
        return pd.read_csv(
            path, dtype={"row_id": str}, keep_default_na=False,
            float_precision="round_trip",
        )

    predictions = read_predictions(raw / "predictions.csv")
    if "Non-nested k-fold" not in set(predictions["method"]):
        predictions = pd.concat(
            [predictions, read_predictions(raw / "nonnested_predictions.csv")],
            ignore_index=True,
        )
    required = {"method", "row_index", "row_id", "y_true", "y_pred"}
    if not required.issubset(predictions.columns):
        raise ValueError("Incomplete prediction columns returned by run_ablation.")
    expected = set(METHODS)
    if not family_variant:
        expected.remove("No Family Priors")
    observed = set(predictions["method"])
    permitted = expected | ({"No Family Priors"} if not family_variant else set())
    if not expected.issubset(observed) or observed - permitted:
        raise ValueError(f"Unexpected or missing ablation variants: {sorted(observed ^ expected)}")
    for method, group in predictions.groupby("method", sort=False):
        group = group.sort_values("row_index")
        if not np.array_equal(group["row_index"].to_numpy(), np.arange(len(y))):
            raise ValueError(f"Missing or duplicate test rows for {method}.")
        if not np.array_equal(group["row_id"].to_numpy(), ids):
            raise ValueError(f"Mismatched row IDs for {method}.")
        if not np.allclose(group["y_true"].to_numpy(dtype=float), y, rtol=1e-12, atol=1e-12):
            raise ValueError(f"Mismatched targets for {method}.")
        if not np.isfinite(group["y_pred"].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite predictions for {method}.")
    if not family_variant:
        predictions = predictions[predictions["method"] != "No Family Priors"].copy()
    predictions["evaluation"] = "nested_outer_loo"
    is_nonnested = predictions["method"] == "Non-nested k-fold"
    predictions.loc[is_nonnested, "evaluation"] = definitions["nonnested"]
    predictions["dataset"] = DATASET_NAME
    if "fold" not in predictions:
        predictions["fold"] = predictions["row_index"]
    predictions = predictions[
        ["dataset", "method", "fold", "row_index", "row_id", "y_true", "y_pred", "evaluation"]
    ]
    metrics = metrics_frame(predictions).set_index("method").reindex(METHODS)
    full_r2 = float(metrics.loc["Full SPARC", "r2"])
    if not np.isfinite(full_r2):
        raise ValueError("Full SPARC R2 is not finite.")
    metrics["full_r2"] = full_r2
    metrics["delta_r2_percent"] = (
        100.0 * (metrics["r2"] - full_r2) / full_r2 if full_r2 != 0 else np.nan
    )
    metrics["status"] = "computed"
    metrics["delta_status"] = "computed" if full_r2 != 0 else "undefined_zero_full_r2"
    metrics["evaluation"] = "nested_outer_loo"
    metrics.loc["Non-nested k-fold", "evaluation"] = definitions["nonnested"]
    if not family_variant:
        metrics.loc["No Family Priors", ["n", "r2", "mae", "rmse", "delta_r2_percent"]] = np.nan
        metrics.loc["No Family Priors", ["status", "delta_status"]] = "not_defined_uniform_family_scales"
        metrics.loc["No Family Priors", "evaluation"] = "not_applicable"
    metrics.insert(0, "dataset", DATASET_NAME)
    return predictions, metrics.reset_index()


def run():
    if not DATASET_NAME or DATASET_NAME in {".", ".."} or Path(DATASET_NAME).name != DATASET_NAME:
        raise ValueError("DATASET_NAME must be a single nonempty directory name.")
    if not isinstance(SEED, int) or SEED < 0:
        raise ValueError("SEED must be a nonnegative integer.")
    if not isinstance(N_JOBS, int) or N_JOBS == 0:
        raise ValueError("N_JOBS must be a nonzero integer.")
    path, X, y, ids = load_input()
    source = None
    if SPARC_SOURCE is not None:
        source = Path(SPARC_SOURCE).expanduser().resolve()
        if not (source / "sparc" / "ablation.py").is_file():
            raise ValueError("SPARC_SOURCE must point to the src directory containing sparc/ablation.py.")
        sys.path.insert(0, str(source))
    import sparc
    from sparc.ablation import run_ablation
    from sparc.figure_io import validate_family_priors, write_json
    from sparc.models import FAMILIES, FAMILY_MAP, build_library

    package = Path(sparc.__file__).resolve().parent
    if source is not None and package != source / "sparc":
        raise RuntimeError("A different SPARC package was already imported; restart Python.")
    validate_family_priors(FAMILY_PRIORS)
    priors = {family: float((FAMILY_PRIORS or {}).get(family, 1.0)) for family in FAMILIES}
    scales = np.array([priors[FAMILY_MAP[name]] for name, _ in build_library(SEED)])
    family_variant = bool(np.any(scales != scales[0]))
    definitions = {
        "sparse_removal": "fixed_local_scales",
        "nonnegative_removal": "signed_affine",
        "nonnested": NONNESTED_DEFINITION,
        "nonnested_folds": NONNESTED_FOLDS,
    }
    options = {"seed": SEED, "definitions": definitions, "family_priors": priors}
    modern = "n_jobs" in inspect.signature(run_ablation).parameters
    supported = "kfold_composition_fitted_on_all_rows" if modern else "fit_and_score_same_oof"
    if NONNESTED_DEFINITION != supported:
        raise ValueError(
            f"This SPARC package requires NONNESTED_DEFINITION={supported!r}; "
            "use the intended package and its matching non-nested definition."
        )
    if modern:
        options["n_jobs"] = N_JOBS
    elif N_JOBS != 1:
        raise ValueError("This SPARC package does not support N_JOBS; set it to 1.")
    out = Path(OUTPUT_ROOT).expanduser().resolve() / DATASET_NAME
    out.mkdir(parents=True, exist_ok=False)
    raw = out / "raw"
    raw.mkdir()
    state = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_NAME,
        "input": {"path": str(path), "sha256": digest(path), "sheet": SHEET_NAME},
        "target": TARGET_COLUMN,
        "target_unit": TARGET_UNIT,
        "features": list(FEATURE_COLUMNS),
        "excluded_outcomes": list(EXCLUDED_OUTCOME_COLUMNS),
        "row_id_column": ROW_ID_COLUMN,
        "rows": len(y),
        "seed": SEED,
        "n_jobs": N_JOBS,
        "family_priors": priors,
        "family_prior_variant": "computed" if family_variant else "not_defined_uniform_family_scales",
        "definitions": definitions,
        "sparc_path": str(package),
        "sparc_version": sparc.__version__,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "jax", "numpyro")
        },
        "source_sha256": {p.name: digest(p) for p in sorted(package.glob("*.py"))},
        "runner_sha256": digest(__file__),
    }
    write_json(out / "run_parameters.json", state)
    try:
        run_ablation(X, y, ids, raw, **options)
        predictions, metrics = assemble_outputs(raw, y, ids, family_variant, definitions)
        predictions.to_csv(out / "predictions.csv", index=False)
        metrics.to_csv(out / "ablation_metrics.csv", index=False, na_rep="")
        state["status"] = "complete"
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        state["outputs_sha256"] = {
            name: digest(out / name) for name in ("predictions.csv", "ablation_metrics.csv")
        }
        write_json(out / "run_parameters.json", state)
    except Exception as error:
        state["status"] = "failed"
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        state["error"] = f"{type(error).__name__}: {error}"
        write_json(out / "run_parameters.json", state)
        raise
    print(metrics[["method", "r2", "delta_r2_percent", "status"]].to_string(index=False, na_rep="n.a."))
    print(f"Results: {out}", flush=True)
    return out


if __name__ == "__main__":
    run()
