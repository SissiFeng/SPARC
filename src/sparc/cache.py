from __future__ import annotations
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
__all__ = ["EvidenceCache", "array_fingerprint", "config_fingerprint"]


def array_fingerprint(a: np.ndarray, *, rtol: float = 0.0, atol: float = 0.0) -> str:
    arr = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    if np.issubdtype(arr.dtype, np.floating):
        buf = np.array(arr, copy=True)
        buf[buf == 0] = 0.0
        h.update(buf.tobytes())
    elif arr.dtype.hasobject:
        values = []
        for value in arr.flat:
            if isinstance(value, np.generic):
                value = value.item()
            if (
                not isinstance(value, (str, int, float, bool, type(None)))
                and value is not pd.NA
            ):
                raise TypeError(
                    f"unsupported object feature for caching: {type(value)}"
                )
            if pd.isna(value):
                values.append(["missing", None])
            else:
                values.append([type(value).__name__, value])
        h.update(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode())
    else:
        h.update(arr.tobytes())
    if rtol or atol:
        h.update(f"rtol={rtol},atol={atol}".encode())
    return h.hexdigest()[:24]


def config_fingerprint(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


class EvidenceCache:
    def __init__(self, root: Union[str, Path], readonly: bool = False) -> None:
        self.root = Path(root)
        self.readonly = bool(readonly)
        if not self.readonly:
            self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, key: str) -> tuple[Path, Path]:
        safe = "".join((c if c.isalnum() or c in "-_" else "_" for c in key))[:80]
        base = self.root / safe
        return (base.with_suffix(".npz"), base.with_suffix(".meta.json"))

    def make_key(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        model_names: list,
        inner_splits,
        sample_weight: Optional[np.ndarray] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        cfg = {
            "model_names": list(model_names),
            "inner_splits": inner_splits,
            "extra": extra or {},
        }
        parts = [
            array_fingerprint(np.asarray(X)),
            array_fingerprint(np.asarray(y)),
            config_fingerprint(cfg),
        ]
        if sample_weight is not None:
            parts.append(array_fingerprint(np.asarray(sample_weight)))
        return "ev_" + hashlib.sha256("".join(parts).encode()).hexdigest()[:32]

    def load(self, key: str) -> Optional[np.ndarray]:
        npy_path, meta_path = self._paths(key)
        if not npy_path.is_file():
            return None
        try:
            P = np.load(npy_path)
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text())
                logger.info(
                    "evidence cache hit key=%s rows=%s cols=%s age_s=%.1f",
                    key[:12],
                    meta.get("n_rows"),
                    meta.get("n_cols"),
                    time.time() - float(meta.get("created_at", time.time())),
                )
            return P
        except Exception as exc:
            logger.warning("evidence cache load failed for %s: %s", key, exc)
            return None

    def store(
        self, key: str, P: np.ndarray, meta: Optional[Dict[str, Any]] = None
    ) -> None:
        if self.readonly:
            return
        npy_path, meta_path = self._paths(key)
        tmp = npy_path.with_suffix(".tmp.npy")
        np.save(tmp, np.asarray(P, dtype=np.float64))
        tmp.replace(npy_path)
        payload = {
            "key": key,
            "created_at": time.time(),
            "n_rows": int(P.shape[0]),
            "n_cols": int(P.shape[1]),
            **(meta or {}),
        }
        meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        logger.info("evidence cache store key=%s shape=%s", key[:12], P.shape)

    def load_metadata(self, key: str) -> Dict[str, Any]:
        _, path = self._paths(key)
        return json.loads(path.read_text()) if path.is_file() else {}

    def get_or_compute(self, key: str, compute_fn):
        hit = self.load(key)
        if hit is not None:
            return (hit, True)
        P = compute_fn()
        self.store(key, P)
        return (P, False)
