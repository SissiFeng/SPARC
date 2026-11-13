from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GINConv, SAGEConv

from .representations import _finite_vector, _molecule, file_sha256


def resolve_checkpoint(source, *, sha256=None, cache_dir=".sparc_models", s3_version_id=None):
    if not source:
        raise ValueError("A local checkpoint path or s3:// URI is required")
    if sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
        raise ValueError("sha256 must contain 64 hexadecimal characters")
    uri = urlparse(str(source))
    if uri.scheme == "s3":
        import boto3

        if not uri.netloc or not uri.path.lstrip("/") or uri.query or uri.fragment:
            raise ValueError("Provide s3://bucket/key and a separate s3_version_id")
        if sha256 is None:
            raise ValueError("S3 checkpoints require sha256 to identify the exact weights")
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / (sha256.lower() + ".pt")
        if not path.exists() or file_sha256(path) != sha256.lower():
            with tempfile.NamedTemporaryFile(dir=cache, delete=False) as handle:
                temporary = Path(handle.name)
            try:
                options = {"VersionId": s3_version_id} if s3_version_id else {}
                boto3.client("s3").download_file(uri.netloc, uri.path.lstrip("/"),
                                                 str(temporary), ExtraArgs=options)
                if file_sha256(temporary) != sha256.lower():
                    raise ValueError("Downloaded checkpoint SHA256 does not match")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    elif uri.scheme:
        raise ValueError("Only local files and s3:// checkpoints are supported")
    else:
        path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256 is not None and file_sha256(path) != sha256.lower():
        raise ValueError("Checkpoint SHA256 does not match")
    return path


def smiles_to_graph(smiles):
    mol = _molecule(smiles)
    atoms = [[a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge(),
              int(a.GetHybridization()), int(a.GetIsAromatic()), a.GetTotalNumHs()]
             for a in mol.GetAtoms()]
    edges, attributes = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend(((i, j), (j, i)))
        values = [bond.GetBondTypeAsDouble(), int(bond.GetIsAromatic()),
                  int(bond.GetIsConjugated()), int(bond.IsInRing())]
        attributes.extend((values, values))
    return Data(x=torch.tensor(atoms, dtype=torch.float32),
                edge_index=torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous(),
                edge_attr=torch.tensor(attributes, dtype=torch.float32).reshape(-1, 4),
                batch=torch.zeros(len(atoms), dtype=torch.long))


class MolecularGraphNetwork(torch.nn.Module):
    def __init__(self, architecture="gcn", input_dim=6, hidden_dims=(32,),
                 embedding_dim=16, activation="relu", dropout=0.0,
                 gin_eps=0.0, gin_train_eps=False):
        super().__init__()
        if architecture not in {"gcn", "gin", "sage"}:
            raise ValueError("architecture must be gcn, gin or sage")
        dimensions = (input_dim, *hidden_dims, embedding_dim)
        if any(not isinstance(d, int) or d < 1 for d in dimensions):
            raise ValueError("All layer dimensions must be positive integers")
        activations = {"relu": torch.nn.ReLU, "gelu": torch.nn.GELU, "tanh": torch.nn.Tanh}
        if activation not in activations or not 0 <= dropout < 1:
            raise ValueError("Invalid activation or dropout")
        layers = []
        for source, target in zip(dimensions[:-1], dimensions[1:]):
            if architecture == "gcn":
                layer = GCNConv(source, target)
            elif architecture == "sage":
                layer = SAGEConv(source, target)
            else:
                network = torch.nn.Sequential(torch.nn.Linear(source, target),
                                              activations[activation](), torch.nn.Linear(target, target))
                layer = GINConv(network, eps=gin_eps, train_eps=gin_train_eps)
            layers.append(layer)
        self.layers = torch.nn.ModuleList(layers)
        self.activation = activations[activation]()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, graph):
        x = graph.x
        for index, layer in enumerate(self.layers):
            x = layer(x, graph.edge_index)
            if index < len(self.layers) - 1:
                x = self.dropout(self.activation(x))
        return x


class MolecularGNN:
    def __init__(self, *, weights, model_config, checkpoint=None, sha256=None,
                 model_factory=MolecularGraphNetwork, featurizer=smiles_to_graph,
                 output_level="node", pooling="mean", seed=42,
                 state_dict_key=None, cache_dir=".sparc_models", s3_version_id=None):
        if weights not in {"pretrained", "random"}:
            raise ValueError("weights must explicitly be pretrained or random")
        if output_level not in {"node", "graph"} or pooling not in {"mean", "sum", "max"}:
            raise ValueError("Invalid output_level or pooling")
        if weights == "random" and (checkpoint is not None or sha256 is not None or s3_version_id is not None):
            raise ValueError("Random initialization cannot receive a checkpoint")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            self.model = model_factory(**model_config).cpu()
        checkpoint_hash = None
        if weights == "pretrained":
            path = resolve_checkpoint(checkpoint, sha256=sha256, cache_dir=cache_dir,
                                      s3_version_id=s3_version_id)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            state = payload if state_dict_key is None else payload[state_dict_key]
            self.model.load_state_dict(state, strict=True)
            checkpoint_hash = file_sha256(path)
        self.model.eval().requires_grad_(False)
        self.featurizer = featurizer
        self.output_level = output_level
        self.pooling = pooling
        self._cache = {}
        digest = hashlib.sha256()
        for key, value in self.model.state_dict().items():
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        from importlib.metadata import version

        self._metadata = {"type": "molecular_gnn", "weights": weights,
                          "model_factory": f"{model_factory.__module__}.{model_factory.__qualname__}",
                          "featurizer": f"{featurizer.__module__}.{featurizer.__qualname__}",
                          "model_config": dict(model_config), "checkpoint": str(checkpoint) if checkpoint else None,
                          "checkpoint_sha256": checkpoint_hash, "state_dict_key": state_dict_key,
                          "s3_version_id": s3_version_id, "seed": int(seed),
                          "state_sha256": digest.hexdigest(), "output_level": output_level,
                          "pooling": pooling, "frozen_during_evaluation": True,
                          "versions": {p: version(p) for p in ("torch", "torch-geometric", "rdkit")}}

    def encode(self, smiles):
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError("SMILES must be a nonempty string")
        if smiles not in self._cache:
            graph = self.featurizer(smiles).cpu()
            with torch.no_grad():
                values = self.model(graph)
            if not isinstance(values, torch.Tensor):
                raise ValueError("The model adapter must return a tensor")
            if self.output_level == "node":
                if values.ndim != 2 or values.shape[0] != graph.num_nodes:
                    raise ValueError("Node output must have shape (num_atoms, embedding_dim)")
                vector = (values.max(dim=0).values if self.pooling == "max"
                          else getattr(values, self.pooling)(dim=0))
            else:
                if values.ndim == 2 and values.shape[0] == 1:
                    values = values[0]
                vector = values
            self._cache[smiles] = _finite_vector(vector.detach().cpu().numpy())
        return self._cache[smiles].copy()

    def metadata(self):
        return dict(self._metadata)
