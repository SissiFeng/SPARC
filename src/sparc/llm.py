from __future__ import annotations
import importlib
from copy import deepcopy
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from .ensemble import SparseEvidenceEnsemble
from .fusion import STRATEGY_NAMES
from .models import build_library
from .preprocessing import feature_matrix
from .figure_io import json_safe


class LLMComparison(RegressorMixin, BaseEstimator):
    def __init__(self, configuration, random_state=42):
        self.configuration = configuration
        self.random_state = random_state

    def _call(self, operation, inputs, schema):
        request = {
            "operation": operation,
            "model_snapshot": self.configuration["model_snapshot"],
            "prompt": self.configuration["prompts"][operation],
            "decoding": deepcopy(self.configuration["decoding"]),
            "output_schema": schema,
            "input": json_safe(inputs),
        }
        response = self._complete(deepcopy(request))
        if not isinstance(response, dict):
            raise ValueError("LLM provider must return a parsed JSON object")
        self.records_.append({"request": request, "response": deepcopy(response)})
        return response

    def fit(self, X, y):
        for name in list(vars(self)):
            if name.endswith("_"):
                delattr(self, name)
        config = self.configuration
        required = {
            "mode",
            "provider",
            "model_snapshot",
            "feature_names",
            "prompts",
            "decoding",
        }
        optional = {"family_priors", "adaptive_rounds", "feedback_folds", "weight_strategy", "global_scale"}
        if set(config) - (required | optional) or required - set(config):
            raise ValueError(
                f"LLM configuration requires {sorted(required)}; optional keys are {sorted(optional)}"
            )
        mode = config["mode"]
        if mode == "standalone" and {"weight_strategy", "global_scale"} & set(config):
            raise ValueError("composition strategies require LLM mode=hybrid")
        if mode not in {"standalone", "hybrid"}:
            raise ValueError("LLM mode must be standalone or hybrid")
        rounds = config.get("adaptive_rounds")
        if mode == "standalone" and (not isinstance(rounds, int) or rounds < 1):
            raise ValueError("standalone mode requires adaptive_rounds >= 1")
        folds = config.get("feedback_folds", 5)
        if mode == "standalone" and (not isinstance(folds, int) or folds < 2):
            raise ValueError("feedback_folds must be an integer of at least 2")
        if (
            not isinstance(config["model_snapshot"], str)
            or not config["model_snapshot"]
        ):
            raise ValueError("model_snapshot is required")
        operations = {"select", "revise", "predict"} if mode == "standalone" else {"select"}
        if (
            not isinstance(config["prompts"], dict)
            or set(config["prompts"]) != operations
            or any(
                (not isinstance(p, str) or not p for p in config["prompts"].values())
            )
            or (not isinstance(config["decoding"], dict))
        ):
            raise ValueError(
                "supply complete prompts and decoding settings for this mode"
            )
        X, y = (feature_matrix(X), np.asarray(y, dtype=float))
        names = config["feature_names"]
        if (
            X.ndim != 2
            or y.shape != (len(X),)
            or len(y) < 3
            or (not np.isfinite(y).all())
            or (len(names) != X.shape[1])
            or (len(set(names)) != len(names))
        ):
            raise ValueError(
                "training observations and distinct feature_names must align"
            )
        module, name = config["provider"].split(":")
        self._complete = getattr(importlib.import_module(module), name)
        self.records_ = []
        inputs = {
            "feature_names": list(names),
            "X_train": X.tolist(),
            "y_train": y.tolist(),
        }
        if mode == "hybrid":
            inputs["feature_strategies"] = list(STRATEGY_NAMES)
            inputs["model_hypotheses"] = [
                name for name, _ in build_library(self.random_state)
            ]
            selection = self._call(
                "select",
                inputs,
                {
                    "feature_strategy": list(STRATEGY_NAMES),
                    "model_names": "nonempty unique array from model_hypotheses",
                },
            )
            if (
                set(selection) != {"feature_strategy", "model_names"}
                or selection["feature_strategy"] not in STRATEGY_NAMES
            ):
                raise ValueError(
                    "hybrid selection must name one feature strategy and model hypotheses"
                )
            self.model_ = SparseEvidenceEnsemble(
                random_state=self.random_state,
                family_priors=config.get("family_priors"),
                weight_strategy=config.get("weight_strategy", "simplex"),
                global_scale=config.get("global_scale", 1.0),
                feature_strategy=selection["feature_strategy"],
                model_names=selection["model_names"],
            ).fit(X, y)
            self.diagnostics_ = self.model_.diagnostics_
        else:
            selection = self._call(
                "select",
                inputs,
                {
                    "feature_indices": "nonempty unique integer array indexing feature_names"
                },
            )
            indices = self._indices(selection, X.shape[1])
            self.adaptation_history_ = []
            for round_no in range(int(rounds)):
                feedback = self._feedback(X, y, names, indices, folds, round_no)
                self.adaptation_history_.append(feedback)
                selection = self._call(
                    "revise",
                    {**inputs, "current_feature_indices": indices, "feedback": feedback},
                    {
                        "feature_indices": "nonempty unique integer array indexing feature_names"
                    },
                )
                indices = self._indices(selection, X.shape[1])
            self.selected_indices_ = indices
            self.training_X_ = X[:, indices].copy()
            self.training_y_ = y.copy()
            self.selected_names_ = [names[i] for i in indices]
            self.diagnostics_ = {}
        self.selection_ = selection
        self.n_features_in_ = X.shape[1]
        return self

    @staticmethod
    def _indices(selection, n_features):
        indices = selection.get("feature_indices")
        if (
            set(selection) != {"feature_indices"}
            or not isinstance(indices, list)
            or (not indices)
            or any((type(i) is not int or not 0 <= i < n_features for i in indices))
            or (len(set(indices)) != len(indices))
        ):
            raise ValueError("standalone selection must return valid feature_indices")
        return list(indices)

    def _in_context(self, names, X_train, y_train, X_predict):
        response = self._call(
            "predict",
            {
                "feature_names": names,
                "X_train": X_train.tolist(),
                "y_train": y_train.tolist(),
                "X_predict": X_predict.tolist(),
            },
            {"predictions": "one finite numeric prediction per X_predict row"},
        )
        prediction = np.asarray(response.get("predictions"), dtype=float)
        if (
            set(response) != {"predictions"}
            or prediction.shape != (len(X_predict),)
            or (not np.isfinite(prediction).all())
        ):
            raise ValueError(
                "LLM predictions must be finite and aligned with prediction rows"
            )
        return prediction

    def _feedback(self, X, y, names, indices, folds, round_no):
        from sklearn.metrics import r2_score
        from sklearn.model_selection import KFold

        selected = [names[i] for i in indices]
        oof = np.empty(len(y))
        splitter = KFold(
            n_splits=min(int(folds), len(y)),
            shuffle=True,
            random_state=int(self.random_state) + round_no,
        )
        for train, test in splitter.split(X):
            oof[test] = self._in_context(
                selected, X[train][:, indices], y[train], X[test][:, indices]
            )
        return {
            "round": round_no,
            "feature_indices": list(indices),
            "cv_r2": float(r2_score(y, oof)),
            "cv_mae": float(np.mean(np.abs(y - oof))),
            "residuals": (y - oof).tolist(),
        }

    def predict(self, X):
        check_is_fitted(self, "n_features_in_")
        X = feature_matrix(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("prediction feature shape differs from training")
        if self.configuration["mode"] == "hybrid":
            return self.model_.predict(X)
        return self._in_context(
            self.selected_names_,
            self.training_X_,
            self.training_y_,
            X[:, self.selected_indices_],
        )


def make_comparison(random_state, configuration):
    return LLMComparison(configuration=configuration, random_state=random_state)
