from __future__ import annotations
import logging
import math
from importlib.util import find_spec
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from .preprocessing import Preprocessor, feature_matrix

logger = logging.getLogger(__name__)
__all__ = [
    "STRATEGY_NAMES",
    "K_STRATEGIES",
    "FeatureFusion",
    "UCBBandit",
    "require_strategy_dependencies",
    "select_feature_strategy",
]


def _w_mutual_info(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    return mutual_info_regression(X, y, random_state=seed)


def _w_elasticnet(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    Xs = StandardScaler().fit_transform(X)
    model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=2000, random_state=seed)
    model.fit(Xs, y)
    return np.abs(model.coef_)


def _w_multiview(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    Xs = StandardScaler().fit_transform(X)
    hidden = min(64, max(8, X.shape[1] // 2))
    try:
        mlp = MLPRegressor(
            hidden_layer_sizes=(hidden,),
            max_iter=300,
            random_state=seed,
            early_stopping=True,
            n_iter_no_change=10,
        )
        mlp.fit(Xs, y)
        return np.abs(mlp.coefs_[0]).sum(axis=1)
    except Exception as exc:
        raise RuntimeError(
            "multiview weighting failed; no substitute strategy was used"
        ) from exc


def _w_clustering(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    Xs = StandardScaler().fit_transform(X)
    n_clusters = min(max(2, len(y) // 5), 10)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=5)
    labels = km.fit_predict(Xs)
    within = np.zeros(Xs.shape[1])
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() > 1:
            within += Xs[mask].var(axis=0)
    return 1.0 / (within / n_clusters + 1e-06)


def _w_permutation(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    rf = RandomForestRegressor(
        n_estimators=50, max_depth=5, random_state=seed, n_jobs=-1
    )
    rf.fit(X, y)
    result = permutation_importance(
        rf, X, y, n_repeats=5, random_state=seed, scoring="r2"
    )
    return np.maximum(result.importances_mean, 0.0)


def _w_attention(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    base = _w_multiview(X, y, seed)
    corr = np.zeros(X.shape[1])
    ys = (y - y.mean()) / (y.std() + 1e-12)
    for j in range(X.shape[1]):
        col = X[:, j]
        s = col.std()
        if s > 0:
            corr[j] = abs(np.mean((col - col.mean()) / s * ys))
    return base * (1.0 + corr)


def _w_shap(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=6, random_state=seed, n_jobs=-1
    )
    rf.fit(X, y)
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP feature weighting requires the 'shap' extra: pip install -e '.[shap]'"
        ) from exc
    explainer = shap.TreeExplainer(rf)
    values = explainer.shap_values(X)
    return np.abs(values).mean(axis=0)


def _w_meta(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:

    def norm(v: np.ndarray) -> np.ndarray:
        rng = v.max() - v.min()
        return (v - v.min()) / rng if rng > 0 else np.ones_like(v)

    parts = [
        _w_mutual_info(X, y, seed),
        _w_permutation(X, y, seed),
        _w_shap(X, y, seed),
    ]
    return np.mean([norm(p) for p in parts], axis=0)


_STRATEGIES: Dict[str, Callable[[np.ndarray, np.ndarray, int], np.ndarray]] = {
    "mutual_info": _w_mutual_info,
    "elasticnet": _w_elasticnet,
    "multiview": _w_multiview,
    "clustering": _w_clustering,
    "permutation": _w_permutation,
    "attention": _w_attention,
    "shap": _w_shap,
    "meta_ensemble": _w_meta,
}
STRATEGY_NAMES: List[str] = list(_STRATEGIES)
K_STRATEGIES: int = len(STRATEGY_NAMES)


def require_strategy_dependencies(strategies: List[str]) -> None:
    if {"shap", "meta_ensemble"}.intersection(strategies) and find_spec("shap") is None:
        raise ImportError(
            "Automatic, SHAP, and meta-ensemble feature weighting require the 'shap' extra: pip install -e '.[shap]'"
        )


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.maximum(w, 0.0)
    total = w.sum()
    if total <= 0:
        return np.ones_like(w)
    return w * (len(w) / total)


class UCBBandit:
    def __init__(self, n_arms: int, delta: float = 0.1) -> None:
        if n_arms < 1:
            raise ValueError("n_arms must be positive")
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        self.n_arms = n_arms
        self.delta = delta
        self._pulls = np.zeros(n_arms, dtype=int)
        self._raw_rewards: List[List[float]] = [[] for _ in range(n_arms)]
        self._t = 0

    @property
    def pulls(self) -> np.ndarray:
        return self._pulls.copy()

    def _all_raw(self) -> List[float]:
        return [r for arm in self._raw_rewards for r in arm]

    def _normalize(self, raw: float) -> float:
        values = [r for r in self._all_raw() if np.isfinite(r)]
        if not values:
            return 0.0
        lo = float(np.min(values))
        hi = float(np.max(values))
        if not np.isfinite(raw):
            return 0.0
        if hi <= lo:
            return 1.0
        return float(np.clip((raw - lo) / (hi - lo), 0.0, 1.0))

    @property
    def mean_rewards(self) -> np.ndarray:
        means = np.zeros(self.n_arms, dtype=np.float64)
        if self._t == 0:
            return means
        for k, raws in enumerate(self._raw_rewards):
            if not raws:
                continue
            norms = [self._normalize(r) for r in raws]
            means[k] = float(np.mean(norms))
        return means

    @property
    def raw_mean_rewards(self) -> np.ndarray:
        means = np.full(self.n_arms, np.nan, dtype=np.float64)
        for k, raws in enumerate(self._raw_rewards):
            if raws:
                means[k] = float(np.mean(raws))
        return means

    def select(self) -> int:
        unexplored = np.where(self._pulls == 0)[0]
        if unexplored.size:
            return int(unexplored[0])
        t = self._t + 1
        beta_t = 2.0 * math.log(self.n_arms * t**2 * math.pi**2 / (6.0 * self.delta))
        bonus = np.sqrt(beta_t * math.log(t) / (self._pulls + 1.0))
        return int(np.argmax(self.mean_rewards + bonus))

    def update(self, arm: int, reward: float) -> None:
        if arm < 0 or arm >= self.n_arms:
            raise IndexError(f"arm {arm} out of bounds for {self.n_arms} arms")
        if not np.isfinite(reward):
            reward = float("-inf")
        self._pulls[arm] += 1
        self._raw_rewards[arm].append(float(reward))
        self._t += 1


def _cv_reward(
    Xw: np.ndarray, y: np.ndarray, cv_splits: int, random_state: int
) -> float:
    cv = KFold(
        n_splits=min(cv_splits, len(y) - 1), shuffle=True, random_state=random_state
    )
    linear = cross_val_score(
        RidgeCV(alphas=[0.1, 1.0, 10.0]), Xw, y, cv=cv, scoring="r2"
    )
    tree = cross_val_score(
        RandomForestRegressor(
            n_estimators=60, max_depth=6, random_state=random_state, n_jobs=-1
        ),
        Xw,
        y,
        cv=cv,
        scoring="r2",
    )
    return float((np.mean(linear) + np.mean(tree)) / 2.0)


def _strategy_cv_reward(
    strategy: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: int,
    random_state: int,
) -> float:
    n = len(y)
    if n < 3:
        raise ValueError("feature-strategy scoring requires at least 3 samples")
    n_splits = min(int(cv_splits), n)
    if n_splits < 2:
        raise ValueError("cv_splits must be at least 2")
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    linear_oof = np.empty(n, dtype=np.float64)
    tree_oof = np.empty(n, dtype=np.float64)
    for fold, (train, test) in enumerate(cv.split(X)):
        preprocessor = Preprocessor().fit(X[train])
        X_train = preprocessor.transform(X[train])
        X_test = preprocessor.transform(X[test])
        weights = _normalize_weights(strategy(X_train, y[train], random_state + fold))
        X_train = X_train * weights
        X_test = X_test * weights
        linear = clone(RidgeCV(alphas=[0.1, 1.0, 10.0]))
        tree = RandomForestRegressor(
            n_estimators=60, max_depth=6, random_state=random_state + fold, n_jobs=-1
        )
        linear.fit(X_train, y[train])
        tree.fit(X_train, y[train])
        linear_oof[test] = linear.predict(X_test)
        tree_oof[test] = tree.predict(X_test)
    return float((r2_score(y, linear_oof) + r2_score(y, tree_oof)) / 2.0)


def select_feature_strategy(
    X: np.ndarray,
    y: np.ndarray,
    n_rounds: Optional[int] = None,
    cv_splits: int = 5,
    delta: float = 0.1,
    random_state: int = 42,
) -> Tuple[str, np.ndarray, np.ndarray]:
    X = feature_matrix(X)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X must be 2-dimensional and aligned with 1-dimensional y")
    if not np.isfinite(y).all():
        raise ValueError("y must contain only finite values")
    require_strategy_dependencies(STRATEGY_NAMES)
    bandit = UCBBandit(K_STRATEGIES, delta=delta)
    rounds = n_rounds if n_rounds is not None else K_STRATEGIES + 2
    if rounds < K_STRATEGIES:
        raise ValueError(
            f"n_rounds must be at least {K_STRATEGIES} so every strategy is evaluated"
        )
    for _ in range(rounds):
        arm = bandit.select()
        fn = _STRATEGIES[STRATEGY_NAMES[arm]]
        try:
            score = _strategy_cv_reward(fn, X, y, cv_splits, random_state)
        except Exception as exc:
            raise RuntimeError(
                f"feature strategy scoring failed for {STRATEGY_NAMES[arm]}"
            ) from exc
        if not np.isfinite(score):
            raise ValueError(f"nonfinite reward for {STRATEGY_NAMES[arm]}")
        bandit.update(arm, score)
    valid = np.isfinite(bandit.raw_mean_rewards)
    if not valid.any():
        raise RuntimeError("all feature strategies failed cross-validation")
    mean = np.where(valid, bandit.mean_rewards, -np.inf)
    chosen = STRATEGY_NAMES[int(np.argmax(mean))]
    return (chosen, bandit.pulls, bandit.mean_rewards)


class FeatureFusion(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        strategy: str = "auto",
        n_rounds: Optional[int] = None,
        cv_splits: int = 5,
        delta: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.strategy = strategy
        self.n_rounds = n_rounds
        self.cv_splits = cv_splits
        self.delta = delta
        self.random_state = random_state

    def _reward(self, Xw: np.ndarray, y: np.ndarray) -> float:
        return _cv_reward(Xw, y, self.cv_splits, self.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeatureFusion":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X must be 2-dimensional and aligned with 1-dimensional y")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError(
                "FeatureFusion requires finite inputs; place Preprocessor before it"
            )
        if self.strategy != "auto":
            if self.strategy not in _STRATEGIES:
                raise ValueError(
                    f"Unknown strategy {self.strategy!r}; choose from {STRATEGY_NAMES} or 'auto'."
                )
            chosen = self.strategy
        else:
            require_strategy_dependencies(STRATEGY_NAMES)
            chosen, pulls, rewards = select_feature_strategy(
                X,
                y,
                n_rounds=self.n_rounds,
                cv_splits=self.cv_splits,
                delta=self.delta,
                random_state=self.random_state,
            )
            self.bandit_pulls_ = pulls
            self.bandit_rewards_ = rewards
        require_strategy_dependencies([chosen])
        self.strategy_ = chosen
        self.weights_ = _normalize_weights(_STRATEGIES[chosen](X, y, self.random_state))
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "weights_")
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features; FeatureFusion was fitted on {self.n_features_in_}."
            )
        return X * self.weights_
