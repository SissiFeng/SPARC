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
        if set(config) - (required | {"family_priors"}) or required - set(config):
            raise ValueError(
                f"LLM configuration requires {sorted(required)}; family_priors is optional"
            )
        mode = config["mode"]
        if mode not in {"standalone", "hybrid"}:
            raise ValueError("LLM mode must be standalone or hybrid")
        if (
            not isinstance(config["model_snapshot"], str)
            or not config["model_snapshot"]
        ):
            raise ValueError("model_snapshot is required")
        operations = {"select", "predict"} if mode == "standalone" else {"select"}
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
            indices = selection.get("feature_indices")
            if (
                set(selection) != {"feature_indices"}
                or not isinstance(indices, list)
                or (not indices)
                or any((type(i) is not int or not 0 <= i < X.shape[1] for i in indices))
                or (len(set(indices)) != len(indices))
            ):
                raise ValueError(
                    "standalone selection must return valid feature_indices"
                )
            self.selected_indices_ = indices
            self.training_X_ = X[:, indices].copy()
            self.training_y_ = y.copy()
            self.selected_names_ = [names[i] for i in indices]
            self.diagnostics_ = {}
        self.selection_ = selection
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "n_features_in_")
        X = feature_matrix(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("prediction feature shape differs from training")
        if self.configuration["mode"] == "hybrid":
            return self.model_.predict(X)
        response = self._call(
            "predict",
            {
                "feature_names": self.selected_names_,
                "X_train": self.training_X_.tolist(),
                "y_train": self.training_y_.tolist(),
                "X_predict": X[:, self.selected_indices_].tolist(),
            },
            {"predictions": "one finite numeric prediction per X_predict row"},
        )
        prediction = np.asarray(response.get("predictions"), dtype=float)
        if (
            set(response) != {"predictions"}
            or prediction.shape != (len(X),)
            or (not np.isfinite(prediction).all())
        ):
            raise ValueError(
                "LLM predictions must be finite and aligned with prediction rows"
            )
        return prediction


def make_comparison(random_state, configuration):
    return LLMComparison(configuration=configuration, random_state=random_state)
