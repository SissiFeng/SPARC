from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut, LeaveOneOut

from .ensemble import SparseEvidenceEnsemble
from .figure_io import check_sampling
from .representations import molecular_features


def evaluate_representations(frame, *, feature_columns, target_column, id_column,
                             smiles_columns, representations, output_dir,
                             estimator=None, seeds=(42,), outer_folds=None,
                             group_column=None):
    features = list(feature_columns)
    smiles = list(smiles_columns)
    if not features or len(features) != len(set(features)):
        raise ValueError("Provide an explicit, nonempty list of distinct baseline feature columns")
    if len(smiles) not in {1, 2} or len(set(smiles)) != len(smiles):
        raise ValueError("Provide one or two distinct SMILES columns")
    reserved = [target_column, id_column, *smiles]
    if group_column is not None:
        reserved.append(group_column)
    if len(set(reserved)) != len(reserved) or set(features) & set(reserved):
        raise ValueError("Features, target, ID, group and SMILES columns must be distinct")
    required = list(dict.fromkeys([*features, *reserved]))
    if not frame.columns.is_unique or set(required) - set(frame.columns):
        raise ValueError("Missing or duplicated input columns")
    if len(frame) < 4 or frame[id_column].isna().any() or not frame[id_column].is_unique:
        raise ValueError("At least four rows with nonmissing unique IDs are required")
    y = frame[target_column].to_numpy(dtype=float)
    if not np.isfinite(y).all() or np.var(y) == 0:
        raise ValueError("Targets must be finite and nonconstant")
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Supply at least one distinct random seed")
    if outer_folds is not None and (not isinstance(outer_folds, int) or outer_folds < 2):
        raise ValueError("outer_folds must be None or an integer >= 2")
    if not representations or "baseline" in representations:
        raise ValueError("Supply named representations; baseline is reserved")
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")
    estimator = SparseEvidenceEnsemble() if estimator is None else estimator
    if "random_state" not in estimator.get_params(deep=True):
        raise ValueError("The downstream estimator must expose random_state")
    groups = None if group_column is None else frame[group_column].to_numpy()
    if groups is not None and pd.isna(groups).any():
        raise ValueError("Group labels must not be missing")
    matrices = {"baseline": frame[features].to_numpy()}
    names = {"baseline": features}
    representation_metadata = {}
    for name, (encoder, pair_mode) in representations.items():
        if not name or not all(c.isalnum() or c == "_" for c in name):
            raise ValueError("Representation names must contain only letters, digits and underscores")
        embedding = molecular_features(frame, smiles, encoder, pair_mode=pair_mode)
        added_names = [f"{name}_{i}" for i in range(embedding.shape[1])]
        if set(features) & set(added_names):
            raise ValueError("Generated feature names overlap baseline feature names")
        matrices[name] = np.column_stack((matrices["baseline"], embedding))
        names[name] = features + added_names
        representation_metadata[name] = {**encoder.metadata(), "pair_mode": pair_mode}
    manifest = {"status": "running", "target_column": target_column, "id_column": id_column,
                "smiles_columns": smiles, "group_column": group_column, "seeds": seeds,
                "outer_protocol": ("leave_one_out" if groups is None else "leave_one_group_out")
                if outer_folds is None else f"{outer_folds}_fold",
                "feature_columns": names, "representations": representation_metadata,
                "estimator": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
                "estimator_parameters": estimator.get_params(deep=False),
                "target_transform": "none", "metric_aggregation": "pooled held-out predictions per seed",
                "input_sha256": hashlib.sha256(frame[required].to_csv(index=False).encode()).hexdigest(),
                "versions": {p: version(p) for p in ("numpy", "pandas", "scikit-learn")}}
    manifest_text = json.dumps(manifest, indent=2, allow_nan=False)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_manifest.json").write_text(manifest_text + "\n")
    frame[required].to_csv(out / "inputs.csv", index=False)
    for name, matrix in matrices.items():
        exported = pd.DataFrame(matrix, columns=names[name])
        exported.insert(0, id_column, frame[id_column].to_numpy())
        exported.to_csv(out / f"features_{name}.csv", index=False)
    predictions, memberships, diagnostics = [], [], []
    for seed in seeds:
        if groups is None:
            splitter = LeaveOneOut() if outer_folds is None else KFold(outer_folds, shuffle=True, random_state=seed)
        else:
            splitter = LeaveOneGroupOut() if outer_folds is None else GroupKFold(outer_folds)
        for fold, (train, test) in enumerate(splitter.split(matrices["baseline"], y, groups)):
            if len(train) < 3:
                raise ValueError("Each outer training fold needs at least three observations")
            for role, indices in (("train", train), ("test", test)):
                memberships.extend({"seed": seed, "fold": fold, "role": role,
                                    "row_index": int(i), "row_id": frame.iloc[i][id_column]} for i in indices)
            for name, matrix in matrices.items():
                model = clone(estimator).set_params(random_state=seed)
                model.fit(matrix[train], y[train])
                check_sampling(getattr(model, "diagnostics_", {}))
                pred = np.asarray(model.predict(matrix[test]), dtype=float)
                if pred.shape != (len(test),) or not np.isfinite(pred).all():
                    raise ValueError(f"Invalid held-out predictions: {name}, fold {fold}")
                predictions.extend({"method": name, "seed": seed, "fold": fold,
                                    "row_index": int(i), "row_id": frame.iloc[i][id_column],
                                    "y_true": float(y[i]), "y_pred": float(p)} for i, p in zip(test, pred))
                if hasattr(model, "diagnostics_report"):
                    diagnostics.append({"method": name, "seed": seed, "fold": fold,
                                        "report": model.diagnostics_report()})
            pd.DataFrame(predictions).to_csv(out / "predictions.csv", index=False)
            pd.DataFrame(memberships).to_csv(out / "folds.csv", index=False)
    results = pd.DataFrame(predictions)
    metrics = []
    for (name, seed), values in results.groupby(["method", "seed"], sort=False):
        if len(values) != len(frame) or not values.row_index.is_unique:
            raise RuntimeError("Held-out predictions must cover every row exactly once per seed")
        metrics.append({"method": name, "seed": seed, "n": len(values),
                        "r2": r2_score(values.y_true, values.y_pred),
                        "mae": mean_absolute_error(values.y_true, values.y_pred),
                        "rmse": root_mean_squared_error(values.y_true, values.y_pred)})
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out / "metrics.csv", index=False)
    if diagnostics:
        from .figure_io import write_json

        write_json(out / "diagnostics.json", diagnostics)
    manifest["status"] = "complete"
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return metrics
