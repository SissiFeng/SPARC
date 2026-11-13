from pathlib import Path
import numpy as np
import pandas as pd


def main():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(16, 2))
    frame = pd.DataFrame(X, columns=["x1", "x2"])
    frame.insert(0, "sample_id", [f"demo_{i:02d}" for i in range(len(X))])
    frame["response"] = 2 * X[:, 0] - X[:, 1] + rng.normal(0, 0.1, len(X))
    frame.to_csv(Path(__file__).with_name("demo.csv"), index=False)


if __name__ == "__main__":
    main()
