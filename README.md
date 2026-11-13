# SPARC

Sparse Predictive Aggregation of Regression Components. Version 2.5.3rc5.

## Install

Use Python 3.12 in an isolated environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install .
```

Install `'.[shap]'` for all eight feature strategies and `'.[excel]'` for Excel input to the ablation runner. The import name and distribution name are both `sparc`.

## Train and predict

```python
from sparc import SparseEvidenceEnsemble, trust_score
from sparc.uncertainty import response_scale

model = SparseEvidenceEnsemble(random_state=42, weight_strategy="simplex")
model.fit(X_train, y_train)
mean, epistemic_sd = model.predict_with_uncertainty(X_candidates)
trust = trust_score(epistemic_sd, model.residual_scale_, response_scale(y_train))
```

The manuscript model uses nonnegative simplex weights. The default `simplex` strategy normalizes coefficient magnitudes to sum to one. Its shared global multiplier cancels under normalization. The supplied optional `global_local` strategy uses unnormalized positive coefficients and changes the predictive model; it is not the manuscript default or a substitute for reproducing its results.

The twelve-model library is tuned by training-only five-fold grid search, except Bayesian ridge and the GP, which estimate their own parameters. Internal leave-one-out evidence supports stability screening and posterior composition. Screening uses 50 subsamples of 80% of training rows and a 60% selection threshold. It is skipped below 20 rows; if no hypothesis passes, the five highest selection frequencies are retained. Hyperparameters, active models and posterior diagnostics are saved with each fitted training state.

NUTS uses four chains, 1,000 warmup iterations and 2,000 retained draws per chain. Acceptance requires weight and noise R-hat below 1.01, ESS above 400 and no divergences. Failed sampling is retried once with 2,000 warmup iterations and otherwise raises an error. The optional global-local strategy also checks its sampled global and local scales. Posterior weight draws propagate composition uncertainty. Truncated importance-LOO residuals are conditional on the evidence matrix; complete outer-held-out evaluation refits the predictive pipeline.

Automatic feature-strategy selection is enabled separately with `feature_strategy="auto"`. Family priors default to one and accept positive scales for `linear`, `tree_based`, `locally_smooth` and `monotonic`. Preprocessing is refitted within each training partition. Features must exclude measured responses, IDs and unavailable future information. The training-MAD trust reference scale has the response's units.

```bash
python examples/train_predict.py --train train.csv --candidates candidates.csv \
  --target response --feature temperature --feature precursor \
  --output predictions.csv --model-output model.joblib
```

## Run ablations by editing one Python file

Edit the input path, feature list, target, units and IDs at the top of `examples/run_sparc_ablation.py`, then run:

```bash
python examples/run_sparc_ablation.py
```

This entry point calls the ablation implementation directly. It writes `predictions.csv`, `ablation_metrics.csv` and run parameters. Each dataset requires a new output directory. The runner checks sample alignment and records the chosen features, input and code hashes, seed and environment.

The component variants fix local scales, allow signed affine weights, remove family preferences, remove screening, or select the best single hypothesis from training evidence. Uniform family scales do not define a distinct prior-removal experiment and are marked not applicable in the runner's metric table. Relative changes use `100*(R2_variant-R2_full)/R2_full` from the same run, preserving signs. The percentage is undefined at a zero reference score and reverses the usual improvement interpretation for a negative reference score.

`Non-nested k-fold` intentionally fits composition on all responses using k-fold evidence, then applies that composition to the same outer-held-out base-model predictions as the other variants. Its `evaluation` field distinguishes this diagnostic from fully nested prediction. It cannot be interpreted as an independent test estimate of the complete composition procedure.

## Numerical workflows

The package contains numerical-data generation functions and no plotting functions. `sparc.figure_data` also provides a configurable entry point:

```bash
python -m sparc.figure_data examples/benchmark.json --out results/demo
```

The supplied 16-row synthetic Ridge example demonstrates execution only. It is not a manuscript dataset or result. Configurations in `examples/figure_data/` are templates: replace paths, columns, units and IDs with actual inputs. Paths are relative to the configuration file; output directories must be empty.

| Workflow | Calculation |
| --- | --- |
| `benchmark` | Outer LOO predictions, pooled metrics per seed, repeated-seed summaries and fold records. |
| `feature_strategies` | Fresh five-fold scoring per policy pull, counts, rewards and explicit failure records. |
| `ablation` | Nested component comparisons and the labelled non-nested diagnostic. |
| `learning_curve` | Repeated learning curves, saturation fits and bootstrap intervals. |
| `finite_pool` | Sequential acquisition with shared, explicitly supplied initial IDs. |
| `extrapolation` | Three GP experts and batches from a joint latent Thompson draw. |
| `campaign` | Candidate ranking and a fixed recommendation list before measurements. |
| `verification` | Diagnostic flags, specified priority records and feature-matched controls. |
| `repeatability` | Fixed cohort IDs joined to subsequent repeat measurements. |
| `campaign_outcomes` | Fixed recommendation IDs joined to measured outcomes. |

Finite-pool search requires 20 initial IDs for `ru_dissolution` or 10 for `polymer_her`; counts include initialization. Candidate responses enter training only after selection. Controlled extrapolation uses latent GP uncertainty, summed cross-validated log predictive densities for expert weights, and one joint latent function draw per batch. Each batch is fixed before its labels are revealed.

Verification retains the manuscript's Ridge-proxy influence definition: the absolute difference between a full-fit Ridge prediction and the supplied outer-OOF ensemble prediction. Priority records require at least two flags; controls have none. Supply `priority_ids` to reproduce a fixed subset. Matching selects one distinct nearest control per priority record. Any extra controls require explicit `additional_control_ids`; the program does not invent an experimental assignment. A residual-only comparator is optional and disabled by default. Select `oof_seed` explicitly when prediction files contain multiple seeds. Save the cohort before joining repeat measurements.

## LLM interface

`sparc.llm:make_comparison` accepts an external provider returning parsed JSON. Configuration requires `mode`, `provider`, `model_snapshot`, `feature_names`, `prompts` and `decoding`. Standalone mode returns selected features and numerical predictions. Hybrid mode returns a feature strategy and model hypotheses for SPARC to fit on the training partition. Requests, responses and selected configurations are saved with each fold. The interface supplies no assumed historical snapshots or prompts.

## Validation and reuse

```bash
python -m unittest discover -s tests -v
```

Tests check numerical invariants, held-out evidence, acquisition behavior, solver contracts and workflow outputs. Mocked solver tests do not replace real posterior sampling or dataset-level reproduction. Historical paper inputs and figure results are not included in this source package.

Repository: https://github.com/SissiFeng/SPARC. Licensed under Apache-2.0; see `LICENSE`. Author: Sissi Feng.
