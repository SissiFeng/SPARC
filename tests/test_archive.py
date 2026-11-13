import ast
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "sparc"


class ArchiveTests(unittest.TestCase):
    def test_all_python_sources_parse(self):
        paths = list(SOURCE.glob("*.py")) + list((ROOT / "examples").glob("*.py"))
        for path in paths:
            with self.subTest(path=path.name):
                ast.parse(path.read_text(), filename=str(path))

    def test_demo_has_explicit_units_and_no_target_leakage(self):
        config = json.loads((ROOT / "examples" / "benchmark.json").read_text())
        spec = config["dataset"]
        self.assertTrue(spec["target_unit"])
        self.assertNotIn(spec["target"], spec["features"])
        self.assertNotIn(spec["row_id"], spec["features"])


try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy is not installed")
class UncertaintyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("sparc_uncertainty", SOURCE / "uncertainty.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_trust_is_invariant_to_response_unit_scaling(self):
        y = np.array([1.0, 2.0, 5.0, 8.0])
        sd = np.array([0.1, 1.0, 10.0])
        q = self.module.response_scale(y)
        base = self.module.trust_score(sd, 0.3, q)
        scaled = self.module.trust_score(sd * 1000, 300, q * 1000)
        np.testing.assert_allclose(base, scaled)

    def test_trust_decreases_with_uncertainty(self):
        values = self.module.trust_score(np.array([0.0, 1.0, 100.0]), 0.2, 1.0)
        self.assertTrue(np.all(np.diff(values) < 0))
        self.assertTrue(np.all((values >= 0) & (values <= 1)))

    def test_invalid_uncertainty_is_rejected(self):
        for value in [-1, np.nan, np.inf]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.module.trust_score(np.array([value]), 0.2, 1.0)

    def test_constant_targets_need_an_explicit_reference_scale(self):
        with self.assertRaises(ValueError):
            self.module.response_scale(np.ones(5))
        self.assertEqual(self.module.resolve_reference_scale(2.0, np.ones(5)), 2.0)
