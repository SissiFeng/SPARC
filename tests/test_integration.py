import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sparc.ood import _fit_expert
from sparc.preprocessing import Preprocessor
from sparc.verification import matched_cohort, quality_report


class IntegrationTests(unittest.TestCase):
    def cohort_inputs(self):
        X = np.arange(10, dtype=float).reshape(-1, 1)
        ids = [f"sample_{i}" for i in range(10)]
        report = pd.DataFrame({
            "idx": np.arange(10), "n_flags": [2, 3, 2, 0, 0, 0, 0, 0, 0, 0],
            "y": np.arange(10, dtype=float), "residual": np.zeros(10),
            "conformal_pvalue": np.linspace(0.01, 0.9, 10),
            "influence": np.arange(10, dtype=float), "trust": np.full(10, 0.8),
        })
        return X, ids, report

    def test_additional_control_requires_explicit_id(self):
        X, ids, report = self.cohort_inputs()
        with self.assertRaisesRegex(ValueError, "additional_control_ids"):
            matched_cohort(report, X, ids, 3, 42, priority_ids=ids[:2])

    def test_fixed_priority_subset_and_extra_control_are_preserved(self):
        X, ids, report = self.cohort_inputs()
        cohort = matched_cohort(
            report, X, ids, 3, 42, priority_ids=ids[:2],
            additional_control_ids=[ids[-1]],
        )
        self.assertEqual(cohort.groupby("group").size().to_dict(), {"control": 3, "priority": 2})
        self.assertEqual(cohort[cohort.group == "priority"].row_id.tolist(), ids[:2])
        self.assertTrue(cohort.row_id.is_unique)
        extra = cohort[cohort.row_id == ids[-1]].iloc[0]
        self.assertTrue(pd.isna(extra.matched_priority_id))
        self.assertEqual(extra.selection_rule, "explicit_additional_control")
        self.assertTrue((cohort[cohort.group == "priority"].eligible_priority_records == 3).all())

    def test_requested_priority_count_cannot_silently_shrink(self):
        X, ids, report = self.cohort_inputs()
        with self.assertRaisesRegex(ValueError, "eligible count"):
            matched_cohort(report, X, ids, 4, 42, n_priority=4)

    def test_influence_uses_supplied_ensemble_oof_predictions(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(12, 2))
        y = X[:, 0] + 0.3 * X[:, 1]
        oof = y + np.linspace(-0.2, 0.3, len(y))
        report = quality_report(X, y, oof, trust_uncertainty=np.full(len(y), 0.1))
        Z = Preprocessor().fit_transform(X)
        expected = np.abs(Ridge(alpha=1.0).fit(Z, y).predict(Z) - oof)
        np.testing.assert_allclose(report.set_index("idx").loc[np.arange(len(y)), "influence"], expected)

    def test_gp_joint_draw_preserves_identical_candidate_values(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(12, 2))
        y = np.sin(X[:, 0]) + rng.normal(0, 0.1, len(X))
        expert = _fit_expert(2, X, y, 0, 42)
        query = np.repeat(X[[0]], 2, axis=0)
        latent_mean, latent_sd = expert.predict(query)
        observed_mean, observed_sd = expert.predict(query, latent=False)
        np.testing.assert_allclose(latent_mean, observed_mean)
        expected_noise = expert.model.kernel_.k2.noise_level * expert.model._y_train_std**2
        np.testing.assert_allclose(observed_sd**2 - latent_sd**2, expected_noise, atol=1e-10)
        draw = expert.sample(query, np.random.RandomState(3))
        self.assertAlmostEqual(float(draw[0]), float(draw[1]), places=7)

    def test_manual_ablation_runner_writes_aligned_metrics(self):
        runner_path = Path(__file__).resolve().parents[1] / "examples" / "run_sparc_ablation.py"
        spec = importlib.util.spec_from_file_location("manual_ablation_runner", runner_path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            frame = pd.DataFrame({"id": [f"r{i}" for i in range(8)], "x": np.arange(8), "y": np.arange(8) * 2.0 + 1.0})
            frame.to_csv(base / "data.csv", index=False)
            runner.DATA_PATH = base / "data.csv"
            runner.DATASET_NAME = "integration"
            runner.FEATURE_COLUMNS = ["x"]
            runner.TARGET_COLUMN = "y"
            runner.TARGET_UNIT = "unit"
            runner.ROW_ID_COLUMN = "id"
            runner.OUTPUT_ROOT = base / "results"
            with patch("sparc.ablation.build_library", return_value=[("Ridge", Ridge())]):
                output = runner.run()
            metrics = pd.read_csv(output / "ablation_metrics.csv").set_index("method")
            self.assertEqual(len(metrics), 7)
            self.assertTrue(pd.isna(metrics.loc["No Family Priors", "r2"]))
            self.assertEqual(metrics.loc["No Family Priors", "status"], "not_defined_uniform_family_scales")
            predictions = pd.read_csv(output / "predictions.csv")
            self.assertEqual(len(predictions), 8 * 6)
            self.assertTrue((predictions.groupby("method").row_id.nunique() == 8).all())
            np.testing.assert_allclose(metrics.loc["Full SPARC", "delta_r2_percent"], 0.0)
