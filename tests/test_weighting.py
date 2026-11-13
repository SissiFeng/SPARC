import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "sparc"
spec = importlib.util.spec_from_file_location("sparc_weighting", SOURCE / "weighting.py")
weighting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weighting)


class WeightStrategyTests(unittest.TestCase):
    def test_simplex_is_invariant_to_global_multiplier(self):
        args = (np.array([1.0, 2.0]), np.array([0.2, 4.0]), np.array([0.5, 1.0]))
        low = weighting.compose_weights(0.01, *args, "simplex")
        high = weighting.compose_weights(100.0, *args, "simplex")
        np.testing.assert_allclose(low, high)
        self.assertAlmostEqual(low.sum(), 1.0)

    def test_global_local_retains_global_scale_in_predictions(self):
        args = (np.ones(2), np.array([0.2, 2.0]), np.ones(2))
        low = weighting.compose_weights(0.1, *args, "global_local")
        high = weighting.compose_weights(1.0, *args, "global_local")
        P = np.array([[1.0, 3.0], [4.0, 2.0]])
        np.testing.assert_allclose(P @ high, 10 * (P @ low))
        self.assertFalse(np.isclose(low.sum(), 1.0))

    def test_local_scale_changes_only_its_coefficient(self):
        first = weighting.compose_weights(0.2, np.ones(2), np.ones(2), np.ones(2), "global_local")
        second = weighting.compose_weights(0.2, np.ones(2), np.array([1.0, 8.0]), np.ones(2), "global_local")
        self.assertEqual(first[0], second[0])
        self.assertEqual(second[1], 8 * first[1])

    def test_single_global_local_model_is_not_forced_to_unit_weight(self):
        actual = weighting.compose_weights(0.2, np.ones(1), np.array([0.5]), np.ones(1), "global_local")
        np.testing.assert_allclose(actual, [0.1])

    def test_relative_diagnostics_do_not_mutate_prediction_coefficients(self):
        draws = np.array([[0.01, 0.09], [0.1, 0.9]])
        before = draws.copy()
        summary = weighting.summarize_coefficients(draws)
        np.testing.assert_array_equal(draws, before)
        np.testing.assert_allclose(summary["relative_weight_mean"], [0.1, 0.9])
        self.assertAlmostEqual(summary["coefficient_sum_mean"], 0.55)

    def test_effective_model_count_distinguishes_concentrated_and_uniform_draws(self):
        uniform = weighting.summarize_coefficients(np.array([[1.0, 1.0, 1.0]]))
        concentrated = weighting.summarize_coefficients(np.array([[1.0, 0.0, 0.0]]))
        self.assertAlmostEqual(uniform["effective_models_mean"], 3.0)
        self.assertAlmostEqual(concentrated["effective_models_mean"], 1.0)

    def test_near_zero_probability_is_separate_from_relative_concentration(self):
        small = weighting.summarize_coefficients(np.array([[0.001, 0.001]]))
        large = weighting.summarize_coefficients(np.array([[1.0, 1.0]]))
        self.assertEqual(small["coefficient_below_threshold_probability"], [1.0, 1.0])
        self.assertEqual(large["coefficient_below_threshold_probability"], [0.0, 0.0])
        self.assertEqual(small["effective_models_mean"], large["effective_models_mean"])

    def test_invalid_strategy_and_global_scale_are_rejected(self):
        for strategy, scale in [("unknown", 1.0), ("global_local", 0.0), ("simplex", np.inf), ("global_local", np.nan)]:
            with self.subTest(strategy=strategy, scale=scale), self.assertRaises(ValueError):
                weighting.validate_weight_strategy(strategy, scale)

    def test_invalid_draws_are_rejected(self):
        for draws in [[], [[0.0, 0.0]], [[np.inf, 1.0]], [[-1.0, 2.0]]]:
            with self.subTest(draws=draws), self.assertRaises(ValueError):
                weighting.summarize_coefficients(draws)
