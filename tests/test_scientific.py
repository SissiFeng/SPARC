import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "sparc"
DEPENDENCIES = ["numpy", "scipy", "pandas", "sklearn"]
AVAILABLE = all(importlib.util.find_spec(name) is not None for name in DEPENDENCIES)

if AVAILABLE:
    import sys
    import numpy as np
    from sklearn.linear_model import Ridge
    sys.path.insert(0, str(ROOT / "src"))
    from sparc.acquisition import BootstrapEnsemble, trust_weighted_confidence_bound
    from sparc.composition import _sampling_converged, evidence_matrix
    from sparc.finite_pool import RFReferenceModel
    from sparc.paper_workflows import rank_fig3_candidates
    from sparc.preprocessing import Preprocessor


@unittest.skipUnless(AVAILABLE, "NumPy/SciPy/pandas/scikit-learn are required")
class ScientificTests(unittest.TestCase):
    def test_noise_diagnostics_can_reject_otherwise_converged_chains(self):
        self.assertTrue(_sampling_converged([1.0], [1000], 0, 1.0, 1000))
        for rhat, ess in [(1.02, 1000), (1.0, 100), (np.nan, 1000), (1.0, np.nan)]:
            with self.subTest(rhat=rhat, ess=ess):
                self.assertFalse(_sampling_converged([1.0], [1000], 0, rhat, ess))
        self.assertFalse(_sampling_converged([1.0], [1000], 1, 1.0, 1000))

    def test_nonfinite_beta_is_rejected(self):
        for beta in [np.nan, np.inf, -np.inf]:
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                trust_weighted_confidence_bound([0.0], [1.0], [0.5], beta)

    def test_minimization_and_maximization_rank_expected_candidates(self):
        mean = np.array([0.0, 3.0])
        sd = np.zeros(2)
        trust = np.ones(2)
        self.assertEqual(np.argmax(trust_weighted_confidence_bound(mean, sd, trust)), 1)
        self.assertEqual(np.argmax(trust_weighted_confidence_bound(mean, sd, trust, maximize=False)), 0)

    def test_evidence_is_held_out(self):
        X = np.arange(16, dtype=float).reshape(8, 2)
        y = np.array([1, 4, 2, 9, 3, 8, 6, 5], dtype=float)
        P = evidence_matrix([("Ridge", Ridge())], X, y, transformer_factory=Preprocessor)
        expected = []
        for i in range(len(y)):
            keep = np.arange(len(y)) != i
            prep = Preprocessor().fit(X[keep])
            expected.append(Ridge().fit(prep.transform(X[keep]), y[keep]).predict(prep.transform(X[[i]]))[0])
        np.testing.assert_allclose(P[:, 0], expected)

    def test_rf_rejects_rows_without_oob_predictions(self):
        X = np.arange(24, dtype=float).reshape(12, 2)
        with self.assertRaisesRegex(RuntimeError, "out-of-bag"):
            RFReferenceModel(n_estimators=1).fit(X, X[:, 0])

    def test_prediction_shape_must_align_with_candidate_count(self):
        class InvalidModel:
            residual_scale_ = 1.0

            def predict_with_uncertainty(self, X):
                return np.zeros(2), np.ones(2)

        with self.assertRaisesRegex(ValueError, "aligned"):
            rank_fig3_candidates(InvalidModel(), np.ones((3, 2)), True, [1, 2, 3])

    def test_bootstrap_predictions_are_reproducible(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(16, 2))
        y = X[:, 0] - 2 * X[:, 1]
        models = [BootstrapEnsemble(60, [Ridge()], random_state=42).fit(X, y) for _ in range(2)]
        self.assertTrue(np.all(models[0].oob_counts_ > 0))
        for first, second in zip(models[0].predict(X), models[1].predict(X)):
            np.testing.assert_allclose(first, second)

    def test_cli_synthetic_ridge_workflow(self):
        if importlib.util.find_spec("jax") is None or importlib.util.find_spec("numpyro") is None:
            self.skipTest("CLI provenance currently requires JAX and NumPyro")
        import json
        import pandas as pd
        from sparc.figure_data import run
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            rng = np.random.default_rng(42)
            X = rng.normal(size=(12, 2))
            pd.DataFrame({"id": np.arange(12), "x1": X[:, 0], "x2": X[:, 1], "y": X[:, 0]}).to_csv(base / "data.csv", index=False)
            config = {"workflow": "benchmark", "seed": 42, "methods": ["Ridge"], "dataset": {"data": "data.csv", "features": ["x1", "x2"], "target": "y", "target_unit": "unit", "row_id": "id"}}
            run(config, base, base / "out")
            state = json.loads((base / "out" / "run_parameters.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertEqual(len(pd.read_csv(base / "out" / "predictions.csv")), 12)
            self.assertTrue(state["outputs_sha256"])

    def test_estimator_predictions_and_uncertainty_keep_unnormalized_scale(self):
        from unittest.mock import patch
        from sparc.ensemble import SparseEvidenceEnsemble
        X = np.arange(16, dtype=float).reshape(8, 2)
        y = X[:, 0]
        draws = np.array([[0.2, 0.1], [0.6, 0.3]])
        model = SparseEvidenceEnsemble(weight_strategy="global_local", global_scale=0.1)
        model.models_ = {}
        model.active_mask_ = np.array([True, True])
        model.weights_ = draws.mean(axis=0)
        model._weight_samples = draws
        surfaces = np.array([[2.0, 4.0], [1.0, 3.0]])
        with patch.object(model, "_surfaces", return_value=surfaces):
            mean, sd = model.predict_with_uncertainty(X[:2])
            np.testing.assert_allclose(mean, surfaces @ draws.mean(axis=0))
            np.testing.assert_allclose(sd, (draws @ surfaces.T).std(axis=0))
            sample = model.sample_joint(X[:2], random_state=42)
            self.assertTrue(any(np.allclose(sample, surfaces @ row) for row in draws))

    def test_strategy_parameters_survive_sklearn_clone(self):
        from sklearn.base import clone
        from sparc.ensemble import SparseEvidenceEnsemble
        cloned = clone(SparseEvidenceEnsemble(weight_strategy="global_local", global_scale=0.05))
        self.assertEqual(cloned.weight_strategy, "global_local")
        self.assertEqual(cloned.global_scale, 0.05)

    def test_solver_does_not_renormalize_global_local_posterior(self):
        if importlib.util.find_spec("numpyro") is None or importlib.util.find_spec("jax") is None:
            self.skipTest("JAX and NumPyro are required")
        from unittest.mock import patch
        from sparc.composition import solve_weights_bayes

        class Sampler:
            def __init__(self, *args, **kwargs):
                pass

            def run(self, *args, **kwargs):
                pass

            def get_samples(self, group_by_chain=False):
                return {
                    "weights": np.broadcast_to([0.4, 0.2], (4, 10, 2)).copy(),
                    "sigma_sq": np.ones((4, 10)),
                    "rho": np.ones((4, 10)),
                    "local_unit": np.ones((4, 10, 2)),
                }

            def get_extra_fields(self):
                return {"diverging": np.zeros(40, dtype=bool)}

        with patch("numpyro.infer.MCMC", Sampler), patch("numpyro.diagnostics.gelman_rubin", return_value=1.0), patch("numpyro.diagnostics.effective_sample_size", return_value=1000.0):
            for strategy, expected in [("global_local", [0.4, 0.2]), ("simplex", [2 / 3, 1 / 3])]:
                weights, _, diagnostics = solve_weights_bayes(np.ones((5, 2)), np.arange(5.0), weight_strategy=strategy)
                np.testing.assert_allclose(weights, expected)
                self.assertEqual(diagnostics["weight_strategy"], strategy)
