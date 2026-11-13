import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Dict, Optional, Tuple
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "sparc"
spec = importlib.util.spec_from_file_location("contract_weighting", SOURCE / "weighting.py")
weighting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weighting)


class SolverContractTests(unittest.TestCase):
    def setUp(self):
        namespace = {"np": np, "Dict": Dict, "Optional": Optional, "Tuple": Tuple}
        for name in ["compose_weights", "summarize_coefficients", "validate_weight_strategy"]:
            namespace[name] = getattr(weighting, name)
        tree = ast.parse((SOURCE / "composition.py").read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"solve_weights_bayes", "_sampling_converged"}]
        exec(compile(ast.Module(body=functions, type_ignores=[]), "composition.py", "exec"), namespace)
        self.solve = namespace["solve_weights_bayes"]
        self.traces = []
        self.attempts = []
        self.bad_rho = False

    def runtime(self):
        owner = self
        names = ["jax", "numpyro", "numpyro.distributions", "numpyro.diagnostics", "numpyro.infer"]
        modules = {name: ModuleType(name) for name in names}
        modules["jax.numpy"] = np
        modules["jax"].numpy = np
        modules["jax"].random = type("Random", (), {"PRNGKey": staticmethod(lambda seed: seed)})
        modules["numpyro"].distributions = modules["numpyro.distributions"]
        trace = {}

        def sample(name, distribution, obs=None):
            if obs is not None:
                trace["likelihood_mean"] = np.asarray(distribution[0])
                return obs
            value = {"rho": 0.2, "sigma_sq": 1.0}.get(name, np.ones_like(distribution))
            trace[name] = value
            return value

        def deterministic(name, value):
            trace[name] = np.asarray(value)
            return value

        modules["numpyro"].sample = sample
        modules["numpyro"].deterministic = deterministic
        modules["numpyro.distributions"].HalfCauchy = lambda scale: scale
        modules["numpyro.distributions"].HalfNormal = lambda scale: scale
        modules["numpyro.distributions"].InverseGamma = lambda **kwargs: 1.0
        modules["numpyro.distributions"].Normal = lambda mean, sd: (mean, sd)
        modules["numpyro.diagnostics"].gelman_rubin = lambda values: 1.1 if owner.bad_rho and np.asarray(values).ndim == 2 and np.allclose(values, 0.2) else np.ones(np.asarray(values).shape[2:])
        modules["numpyro.diagnostics"].effective_sample_size = lambda values: np.full(np.asarray(values).shape[2:], 1000.0)
        modules["numpyro.infer"].NUTS = lambda model, **kwargs: model

        class Sampler:
            def __init__(self, model, **kwargs):
                self.model = model
                owner.attempts.append(kwargs["num_warmup"])

            def run(self, key, P_, y_, scales_, **kwargs):
                trace.clear()
                self.model(P_, y_, scales_)
                owner.traces.append(dict(trace))

            def get_samples(self, group_by_chain=False):
                weights = trace["weights"]
                return {
                    "weights": np.broadcast_to(weights, (4, 20, len(weights))).copy(),
                    "sigma_sq": np.ones((4, 20)),
                    "rho": np.full((4, 20), 0.2),
                    "local_unit": np.ones((4, 20, len(weights))),
                }

            def get_extra_fields(self):
                return {"diverging": np.zeros(80, dtype=bool)}

        modules["numpyro.infer"].MCMC = Sampler
        return patch.dict(sys.modules, modules)

    def test_global_local_likelihood_and_return_values_are_not_normalized(self):
        P = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        with self.runtime():
            weights, _, diagnostic = self.solve(P, np.array([1.0, 2.0, 3.0]), weight_strategy="global_local")
        np.testing.assert_allclose(weights, [0.2, 0.2])
        np.testing.assert_allclose(self.traces[0]["likelihood_mean"], P @ weights)
        self.assertTrue(diagnostic["global_shrinkage_effective"])

    def test_simplex_remains_normalized_and_marks_rho_as_ineffective(self):
        with self.runtime():
            weights, _, diagnostic = self.solve(np.ones((3, 2)), np.arange(3.0))
        np.testing.assert_allclose(weights, [0.5, 0.5])
        self.assertFalse(diagnostic["global_shrinkage_effective"])

    def test_single_global_local_model_uses_sampler(self):
        with self.runtime():
            weights, _, _ = self.solve(np.ones((3, 1)), np.arange(3.0), weight_strategy="global_local")
        np.testing.assert_allclose(weights, [0.2])
        self.assertEqual(len(self.attempts), 1)

    def test_failed_global_diagnostics_trigger_retry_and_rejection(self):
        self.bad_rho = True
        with self.runtime(), self.assertRaisesRegex(RuntimeError, "rho_rhat"):
            self.solve(np.ones((3, 2)), np.arange(3.0), weight_strategy="global_local")
        self.assertEqual(self.attempts, [1000, 2000])
