import importlib.util
import json
import unittest
from pathlib import Path

for dependency in ("pytest", "torch", "torch_geometric", "rdkit", "transformers", "gensim", "boto3"):
    if importlib.util.find_spec(dependency) is None:
        raise unittest.SkipTest("Install sparc[representations,test] to run representation tests")

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from sparc.graph_models import MolecularGNN, MolecularGraphNetwork, resolve_checkpoint, smiles_to_graph
from sparc.representation_benchmark import evaluate_representations
from sparc.representations import FrozenChemBERTa, Mat2VecElementAverage, file_sha256, molecular_features


def random_gcn(seed=42):
    return MolecularGNN(weights="random", model_config={"architecture": "gcn"}, seed=seed)


def test_archived_gcn_parity_and_rng():
    from torch_geometric.nn import GCNConv

    state = torch.random.get_rng_state().clone()
    encoder = random_gcn(seed=42)
    assert torch.equal(state, torch.random.get_rng_state())
    graph = smiles_to_graph("CC(=O)O")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        first, second = GCNConv(6, 32), GCNConv(32, 16)
    with torch.no_grad():
        expected = second(torch.relu(first(graph.x, graph.edge_index)), graph.edge_index).mean(0)
    np.testing.assert_array_equal(encoder.encode("CC(=O)O"), expected.numpy())
    assert encoder.encode("[He]").shape == (16,)
    with pytest.raises(ValueError):
        encoder.encode("not a molecule")


@pytest.mark.parametrize("architecture", ["gcn", "gin", "sage"])
@pytest.mark.parametrize("pooling", ["mean", "sum", "max"])
def test_configurable_network_checkpoint_roundtrip(tmp_path, architecture, pooling):
    config = {"architecture": architecture, "hidden_dims": (12, 8), "embedding_dim": 5}
    random = MolecularGNN(weights="random", model_config=config, pooling=pooling, seed=7)
    path = tmp_path / "weights.pt"
    torch.save({"encoder": random.model.state_dict()}, path)
    restored = MolecularGNN(weights="pretrained", model_config=config, checkpoint=path,
                            sha256=file_sha256(path), state_dict_key="encoder", pooling=pooling, seed=99)
    np.testing.assert_array_equal(random.encode("CCO"), restored.encode("CCO"))
    assert restored.metadata()["checkpoint_sha256"] == file_sha256(path)
    assert not any(p.requires_grad for p in restored.model.parameters())
    assert restored.encode("[He]").shape == (5,)
    with pytest.raises(ValueError):
        MolecularGNN(weights="pretrained", model_config=config)
    with pytest.raises(RuntimeError):
        MolecularGNN(weights="pretrained", model_config={**config, "embedding_dim": 3},
                     checkpoint=path, state_dict_key="encoder")


def test_custom_aws_architecture_adapter(tmp_path):
    class OriginalModel(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.projection = torch.nn.Linear(6, width)

        def forward(self, graph):
            return self.projection(graph.x).sum(0).unsqueeze(0)

    model = OriginalModel(width=7)
    path = tmp_path / "aws.pt"
    torch.save(model.state_dict(), path)
    encoder = MolecularGNN(weights="pretrained", model_factory=OriginalModel,
                           model_config={"width": 7}, checkpoint=path, output_level="graph")
    with torch.no_grad():
        expected = model(smiles_to_graph("CCN"))[0].numpy()
    np.testing.assert_array_equal(encoder.encode("CCN"), expected)


def test_s3_version_and_hash(tmp_path, monkeypatch):
    import boto3
    import shutil

    source = tmp_path / "source.pt"
    torch.save({"x": torch.tensor([1.0])}, source)
    calls = []

    class Client:
        def download_file(self, bucket, key, path, ExtraArgs):
            calls.append((bucket, key, ExtraArgs))
            shutil.copyfile(source, path)

    monkeypatch.setattr(boto3, "client", lambda service: Client())
    sha = file_sha256(source)
    result = resolve_checkpoint("s3://example/gnn.pt", sha256=sha,
                                s3_version_id="v1", cache_dir=tmp_path / "cache")
    assert file_sha256(result) == sha
    assert calls == [("example", "gnn.pt", {"VersionId": "v1"})]
    with pytest.raises(ValueError, match="SHA256"):
        resolve_checkpoint("s3://example/gnn.pt", sha256="0" * 64, cache_dir=tmp_path / "bad")
    assert not list((tmp_path / "bad").iterdir())


def test_pair_invariance():
    frame = pd.DataFrame({"A": ["CCO", "CCN"], "B": ["CCN", "CCO"]})
    values = molecular_features(frame, ("A", "B"), random_gcn(), pair_mode="mean_absdiff")
    assert values.shape == (2, 32)
    np.testing.assert_array_equal(values[0], values[1])


def test_real_checkpoint_required_for_mat2vec(tmp_path):
    from gensim.models import KeyedVectors

    vectors = KeyedVectors(vector_size=2)
    vectors.add_vectors(["C", "H", "O"], np.array([[1., 0.], [0., 1.], [2., 2.]], dtype=np.float32))
    path = tmp_path / "elements.kv"
    vectors.save(str(path))
    encoder = Mat2VecElementAverage(path, checkpoint_format="keyed_vectors")
    np.testing.assert_allclose(encoder.encode("CO"), np.array([3., 6.]) / 6)
    with pytest.raises(ValueError, match="absent"):
        encoder.encode("N")
    with pytest.raises(FileNotFoundError):
        Mat2VecElementAverage(tmp_path / "missing", checkpoint_format="keyed_vectors")


@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_chemberta_frozen_local_checkpoint(tmp_path, pooling):
    from tokenizers import ByteLevelBPETokenizer
    from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(["CCO", "CCN", "CCC", "c1ccccc1"], vocab_size=280,
                                   special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    tokenizer.save_model(str(tmp_path))
    fast = RobertaTokenizerFast(vocab_file=str(tmp_path / "vocab.json"),
                                merges_file=str(tmp_path / "merges.txt"), model_max_length=32)
    fast.save_pretrained(tmp_path)
    model = RobertaModel(RobertaConfig(vocab_size=len(fast), hidden_size=12,
                                      num_hidden_layers=1, num_attention_heads=2,
                                      intermediate_size=16, max_position_embeddings=34))
    model.save_pretrained(tmp_path)
    encoder = FrozenChemBERTa(tmp_path, pooling=pooling, max_length=32)
    result = encoder.encode("CCO")
    assert result.shape == (12,)
    encoder._cache.clear()
    np.testing.assert_array_equal(result, encoder.encode("CCO"))
    assert not encoder.model.training
    with pytest.raises(ValueError, match="commit"):
        FrozenChemBERTa("seyonec/ChemBERTa-zinc-base-v1", pooling=pooling)
    with pytest.raises(ValueError, match="max_length"):
        encoder.encode("C" * 80)


def test_held_out_target_isolation_and_exported_metrics(tmp_path):
    frame = pd.DataFrame({"id": range(6), "A": ["C", "CC", "CCC", "CCCC", "CCO", "CCN"],
                          "feature": [1., 2., 3., 4., 5., 6.], "target": [2., 3., 6., 7., 11., 12.]})
    options = dict(feature_columns=("feature",), target_column="target", id_column="id",
                   smiles_columns=("A",), representations={"graph": (random_gcn(), "mean")},
                   estimator=Ridge(random_state=42))
    evaluate_representations(frame, output_dir=tmp_path / "first", **options)
    changed = frame.copy()
    changed.loc[0, "target"] = 10000.
    evaluate_representations(changed, output_dir=tmp_path / "second", **options)
    first = pd.read_csv(tmp_path / "first/predictions.csv")
    second = pd.read_csv(tmp_path / "second/predictions.csv")
    np.testing.assert_array_equal(first[first.row_index == 0].y_pred,
                                  second[second.row_index == 0].y_pred)
    metrics = pd.read_csv(tmp_path / "first/metrics.csv")
    for name, values in first.groupby("method"):
        assert metrics.loc[metrics.method == name, "r2"].iloc[0] == pytest.approx(r2_score(values.y_true, values.y_pred))
    assert json.loads((tmp_path / "first/run_manifest.json").read_text())["status"] == "complete"
    with pytest.raises(ValueError, match="distinct"):
        evaluate_representations(frame, output_dir=tmp_path / "bad",
                                 **{**options, "feature_columns": ("target",)})


def test_group_exclusion(tmp_path):
    frame = pd.DataFrame({"id": range(6), "A": ["C", "CC", "CCC", "CCCC", "CCO", "CCN"],
                          "x": [1., 2., 3., 4., 5., 6.], "y": [2., 3., 6., 7., 11., 12.],
                          "group": ["a", "a", "b", "b", "c", "c"]})
    evaluate_representations(frame, feature_columns=("x",), target_column="y", id_column="id",
                             smiles_columns=("A",), group_column="group", output_dir=tmp_path / "grouped",
                             representations={"graph": (random_gcn(), "mean")},
                             estimator=Ridge(random_state=42))
    folds = pd.read_csv(tmp_path / "grouped/folds.csv")
    for _, fold in folds.groupby("fold"):
        training = frame.loc[fold[fold.role == "train"].row_index, "group"]
        testing = frame.loc[fold[fold.role == "test"].row_index, "group"]
        assert not set(training) & set(testing)
