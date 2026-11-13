from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted


def feature_matrix(X):
    return (
        np.asarray(X, dtype=object) if isinstance(X, (list, tuple)) else np.asarray(X)
    )


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, impute: str = "median", scale: bool = True) -> None:
        self.impute = impute
        self.scale = scale

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "Preprocessor":
        X = feature_matrix(X)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2-dimensional array")
        if np.iscomplexobj(X):
            raise ValueError("complex-valued features are not supported")
        self.n_features_in_ = X.shape[1]
        numeric, categorical = ([], [])
        for j in range(X.shape[1]):
            values = X[:, j]
            try:
                np.asarray(values[~pd.isna(values)], dtype=np.float64)
                numeric.append(j)
            except (TypeError, ValueError):
                categorical.append(j)
        self.numeric_columns_ = np.asarray(numeric, dtype=int)
        self.categorical_columns_ = np.asarray(categorical, dtype=int)
        self.category_fill_values_ = []
        self.encoder_ = None
        if categorical:
            columns = []
            for j in categorical:
                values = X[:, j]
                missing = pd.isna(values)
                labels, counts = np.unique(
                    values[~missing].astype(str), return_counts=True
                )
                fill = labels[np.argmax(counts)]
                self.category_fill_values_.append(fill)
                columns.append(np.where(missing, fill, values).astype(str))
            self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.encoder_.fit(np.column_stack(columns))
        numeric_X = self._numeric(X)
        if self.impute == "median":
            fill = np.nanmedian(numeric_X, axis=0)
        elif self.impute == "mean":
            fill = np.nanmean(numeric_X, axis=0)
        else:
            raise ValueError(f"impute must be 'median' or 'mean', got {self.impute!r}")
        self.fill_values_ = np.where(np.isfinite(fill), fill, 0.0)
        Xf = self._encode(X)
        std = Xf.std(axis=0)
        self.keep_mask_ = std > 0.0
        if not self.keep_mask_.any():
            raise ValueError("X has no non-constant usable features")
        Xk = Xf[:, self.keep_mask_]
        self.mean_ = Xk.mean(axis=0)
        self.std_ = Xk.std(axis=0)
        self.std_[self.std_ == 0.0] = 1.0
        return self

    def _numeric(self, X):
        values = X[:, self.numeric_columns_]
        values = np.asarray(np.where(pd.isna(values), np.nan, values), dtype=np.float64)
        return np.where(np.isfinite(values), values, np.nan)

    def _encode(self, X):
        numeric = self._numeric(X)
        numeric = np.where(np.isnan(numeric), self.fill_values_, numeric)
        if self.encoder_ is None:
            return numeric
        columns = [
            np.where(pd.isna(X[:, j]), fill, X[:, j]).astype(str)
            for j, fill in zip(self.categorical_columns_, self.category_fill_values_)
        ]
        return np.column_stack(
            (numeric, self.encoder_.transform(np.column_stack(columns)))
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "fill_values_")
        X = feature_matrix(X)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features; Preprocessor was fitted on {self.n_features_in_}."
            )
        X = self._encode(X)
        X = X[:, self.keep_mask_]
        if self.scale:
            X = (X - self.mean_) / self.std_
        return X
