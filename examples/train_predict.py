import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sparc import SparseEvidenceEnsemble, trust_score


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--feature", action="append", dest="features", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.target in args.features or len(set(args.features)) != len(args.features):
        raise ValueError("Features must be unique and exclude the target column")
    train = pd.read_csv(args.train)
    candidates = pd.read_csv(args.candidates)
    for frame in (train, candidates):
        missing = set(args.features) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
    model = SparseEvidenceEnsemble(random_state=args.seed)
    model.fit(train[args.features].to_numpy(), train[args.target].to_numpy(dtype=float))
    mean, epistemic_sd = model.predict_with_uncertainty(
        candidates[args.features].to_numpy()
    )
    predictions = pd.DataFrame(
        {
            "row_index": np.arange(len(candidates)),
            "prediction": mean,
            "epistemic_sd": epistemic_sd,
            "residual_scale": model.aleatoric_,
            "trust": trust_score(epistemic_sd, model.aleatoric_),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    if args.model_output:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "features": args.features}, args.model_output)
    print(f"Saved {len(predictions)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
