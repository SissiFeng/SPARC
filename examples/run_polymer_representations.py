from pathlib import Path

import pandas as pd

from sparc import SparseEvidenceEnsemble
from sparc.graph_models import MolecularGNN, MolecularGraphNetwork, smiles_to_graph
from sparc.representation_benchmark import evaluate_representations
from sparc.representations import FrozenChemBERTa, Mat2VecElementAverage


INPUT_CSV = Path("polymer.csv")
OUTPUT_DIR = Path("results/polymer_representations")
ID_COLUMN = "sample_id"
TARGET_COLUMN = "HER"
SMILES_COLUMNS = ("smiles_A", "smiles_B")
FEATURE_COLUMNS = ("feature_1", "feature_2")
GROUP_COLUMN = None
OUTER_FOLDS = None
SEEDS = (42,)
METHODS = ("pretrained_gnn",)
GNN_CHECKPOINT = None
GNN_SHA256 = None
GNN_S3_VERSION_ID = None
GNN_STATE_DICT_KEY = None
GNN_MODEL_FACTORY = MolecularGraphNetwork
GNN_FEATURIZER = smiles_to_graph
GNN_MODEL_CONFIG = {"architecture": "gcn", "input_dim": 6, "hidden_dims": (32,),
                    "embedding_dim": 16, "activation": "relu", "dropout": 0.0}
GNN_OUTPUT_LEVEL = "node"
GNN_POOLING = "mean"
GNN_PAIR_MODE = "mean_absdiff"
GNN_SEED = 42
CHEMBERTA_CHECKPOINT = "seyonec/ChemBERTa-zinc-base-v1"
CHEMBERTA_REVISION = None
CHEMBERTA_POOLING = "cls"
CHEMBERTA_PAIR_MODE = "concat"
MAT2VEC_CHECKPOINT = Path("models/mat2vec.model")
MAT2VEC_FORMAT = "word2vec_model"
MAT2VEC_INCLUDE_HYDROGENS = True
MAT2VEC_PAIR_MODE = "mean_absdiff"
FAMILY_PRIORS = None


def main():
    if len(set(METHODS)) != len(METHODS) or set(METHODS) - {"pretrained_gnn", "random_gnn", "chemberta", "mat2vec"}:
        raise ValueError("METHODS contains duplicate or unknown names")
    representations = {}
    for method in ("pretrained_gnn", "random_gnn"):
        if method in METHODS:
            pretrained = method == "pretrained_gnn"
            representations[method] = (
                MolecularGNN(weights="pretrained" if pretrained else "random",
                             model_config=GNN_MODEL_CONFIG, model_factory=GNN_MODEL_FACTORY,
                             featurizer=GNN_FEATURIZER, output_level=GNN_OUTPUT_LEVEL,
                             pooling=GNN_POOLING, seed=GNN_SEED,
                             checkpoint=GNN_CHECKPOINT if pretrained else None,
                             sha256=GNN_SHA256 if pretrained else None,
                             s3_version_id=GNN_S3_VERSION_ID if pretrained else None,
                             state_dict_key=GNN_STATE_DICT_KEY), GNN_PAIR_MODE)
    if "chemberta" in METHODS:
        representations["chemberta"] = (
            FrozenChemBERTa(CHEMBERTA_CHECKPOINT, revision=CHEMBERTA_REVISION,
                           pooling=CHEMBERTA_POOLING), CHEMBERTA_PAIR_MODE)
    if "mat2vec" in METHODS:
        representations["mat2vec"] = (
            Mat2VecElementAverage(MAT2VEC_CHECKPOINT, checkpoint_format=MAT2VEC_FORMAT,
                                  include_hydrogens=MAT2VEC_INCLUDE_HYDROGENS), MAT2VEC_PAIR_MODE)
    metrics = evaluate_representations(
        pd.read_csv(INPUT_CSV), feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN, id_column=ID_COLUMN, smiles_columns=SMILES_COLUMNS,
        group_column=GROUP_COLUMN, outer_folds=OUTER_FOLDS, seeds=SEEDS,
        representations=representations, output_dir=OUTPUT_DIR,
        estimator=SparseEvidenceEnsemble(family_priors=FAMILY_PRIORS))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
