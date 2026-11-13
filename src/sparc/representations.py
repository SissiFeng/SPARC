from __future__ import annotations

import hashlib
import re
from importlib.metadata import version
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _molecule(smiles):
    from rdkit import Chem

    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a nonempty string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid molecular SMILES: {smiles!r}")
    return mol


def _finite_vector(values):
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError("Representation must be a finite, nonempty vector")
    return vector


class FrozenChemBERTa:
    def __init__(self, checkpoint, *, pooling, revision=None, max_length=512):
        from transformers import AutoModel, AutoTokenizer

        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must explicitly be 'cls' or 'mean'")
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        local = Path(checkpoint).is_dir()
        if not local and not re.fullmatch(r"[0-9a-fA-F]{40}", revision or ""):
            raise ValueError("Remote checkpoints require a full 40-character commit revision")
        options = {"local_files_only": local, "trust_remote_code": False}
        if not local:
            options["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), **options)
        self.model = AutoModel.from_pretrained(str(checkpoint), **options)
        self.model.cpu().eval().requires_grad_(False)
        self.pooling = pooling
        self.max_length = int(max_length)
        self._cache = {}
        hashes = ({str(p.relative_to(checkpoint)): file_sha256(p)
                   for p in sorted(Path(checkpoint).rglob("*")) if p.is_file()} if local else {})
        self._metadata = {"type": "frozen_chemberta", "checkpoint": str(checkpoint),
                          "revision": revision, "local_file_sha256": hashes,
                          "pooling": pooling, "max_length": self.max_length,
                          "truncation": False, "fine_tuned_on_target": False,
                          "versions": {p: version(p) for p in ("torch", "transformers", "rdkit")}}

    def encode(self, smiles):
        import torch

        if smiles not in self._cache:
            _molecule(smiles)
            inputs = self.tokenizer(smiles, return_tensors="pt", truncation=False)
            if inputs["input_ids"].shape[1] > self.max_length:
                raise ValueError("SMILES exceeds max_length; no silent truncation is allowed")
            with torch.no_grad():
                hidden = self.model(**inputs).last_hidden_state
                if self.pooling == "cls":
                    vector = hidden[0, 0]
                else:
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    vector = (hidden * mask).sum(dim=1)[0] / mask.sum(dim=1)[0]
            self._cache[smiles] = _finite_vector(vector.numpy())
        return self._cache[smiles].copy()

    def metadata(self):
        return dict(self._metadata)


class Mat2VecElementAverage:
    def __init__(self, checkpoint, *, checkpoint_format, include_hydrogens=True):
        from gensim.models import KeyedVectors, Word2Vec

        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if checkpoint_format == "word2vec_model":
            self.vectors = Word2Vec.load(str(checkpoint)).wv
        elif checkpoint_format == "keyed_vectors":
            self.vectors = KeyedVectors.load(str(checkpoint))
        elif checkpoint_format in {"word2vec_text", "word2vec_binary"}:
            self.vectors = KeyedVectors.load_word2vec_format(
                str(checkpoint), binary=checkpoint_format == "word2vec_binary")
        else:
            raise ValueError("Unknown checkpoint_format")
        self.include_hydrogens = bool(include_hydrogens)
        self._cache = {}
        files = [checkpoint, *sorted(checkpoint.parent.glob(checkpoint.name + ".*"))]
        self._metadata = {"type": "mat2vec_element_average", "checkpoint": str(checkpoint),
                          "checkpoint_format": checkpoint_format,
                          "file_sha256": {p.name: file_sha256(p) for p in files if p.is_file()},
                          "include_hydrogens": self.include_hydrogens,
                          "pooling": "atomic-fraction-weighted mean of element tokens",
                          "missing_elements": "error", "fine_tuned_on_target": False,
                          "versions": {p: version(p) for p in ("gensim", "rdkit")}}

    def encode(self, smiles):
        from rdkit import Chem

        if smiles not in self._cache:
            mol = _molecule(smiles)
            if self.include_hydrogens:
                mol = Chem.AddHs(mol)
            vectors = []
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                key = symbol if symbol in self.vectors else symbol.lower()
                if key not in self.vectors:
                    raise ValueError(f"Element {symbol!r} absent from the supplied checkpoint")
                vectors.append(self.vectors[key])
            self._cache[smiles] = _finite_vector(np.mean(vectors, axis=0))
        return self._cache[smiles].copy()

    def metadata(self):
        return dict(self._metadata)


def molecular_features(frame, smiles_columns, encoder, *, pair_mode):
    columns = tuple(smiles_columns)
    if len(columns) not in {1, 2} or len(set(columns)) != len(columns):
        raise ValueError("Specify one SMILES column or two distinct monomer SMILES columns")
    if frame.empty:
        raise ValueError("Input contains no rows")
    if pair_mode not in {"concat", "mean_absdiff", "mean"}:
        raise ValueError("pair_mode must be concat, mean_absdiff or mean")
    encoded = [np.vstack([encoder.encode(s) for s in frame[c]]) for c in columns]
    if len(encoded) == 1:
        result = encoded[0]
    elif pair_mode == "concat":
        result = np.hstack(encoded)
    elif pair_mode == "mean":
        result = (encoded[0] + encoded[1]) / 2
    else:
        result = np.hstack(((encoded[0] + encoded[1]) / 2, np.abs(encoded[0] - encoded[1])))
    if not np.isfinite(result).all():
        raise ValueError("Representation contains nonfinite values")
    return result
