# SPARC

Sparse Predictive Aggregation of Regression Components.

## Install

Use Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install .
```

Install `pip install '.[shap]'` for the eight feature strategies.

## Train and predict

```python
from sparc import SparseEvidenceEnsemble, trust_score

model = SparseEvidenceEnsemble(random_state=42)
model.fit(X_train, y_train)
mean, epistemic_sd = model.predict_with_uncertainty(X_new)
trust = trust_score(epistemic_sd, model.aleatoric_, reference_scale=1.0)
```

The model uses internal leave-one-out evidence, the twelve-model library,
50 subsamples containing 80% of training rows, a 60% selection threshold,
and Bayesian simplex composition. Screening is skipped below 20 training
observations; if no model passes, the five highest frequencies are retained.
NUTS uses four chains, 1,000 warmup iterations and 2,000 draws per chain;
failed weight convergence triggers one retry with 2,000 warmup iterations.
Every retained model is refitted, regardless of its posterior-mean weight.

The primary benchmark has no feature policy or training-time observation
weighting. Automatic feature-strategy selection is enabled separately with
`feature_strategy="auto"`. Family scales default to one and can be supplied
using `family_priors` with keys `linear`, `tree_based`, `locally_smooth` and
`monotonic`. All specified values must be positive.

Imputation, encoding, zero-variance filtering and standardization are fitted
inside each training fold. Input features must be available before the
response is measured. Input scaling does not standardize the response.
Trust's reference scale uses the same units as the response.

```bash
python examples/train_predict.py \
  --train train.csv --candidates candidates.csv --target response \
  --feature temperature --feature precursor \
  --output predictions.csv --model-output model.joblib
```

`joblib.load("model.joblib")` loads the fitted estimator and feature order.

## Generate numerical figure data

```bash
sparc-figure-data examples/figure_data/benchmark.json --out results/benchmark
```

Each example JSON specifies a workflow. Replace its input paths, column names,
units and sample IDs with the dataset fields. Paths are relative to the JSON.
Use unique sample IDs and finite targets. Specify other response columns in
`excluded_outcomes`. Outputs are CSV tables and JSON run parameters. The output
directory must be empty. The package contains no plotting functions.

| Configuration | Calculation |
| --- | --- |
| `benchmark` | Matched outer LOO, inner LOO for SPARC, assembled predictions and pooled R²/MAE/RMSE. |
| `feature_strategies` | Five-fold RidgeCV/RF rewards for all eight strategies within each outer training set; UCB with δ=0.1; raw R², normalized rewards and selection counts. |
| `ablation` | Paired nested-LOO component comparisons and a separately exported non-nested k-fold alternative. |
| `learning_curve` | ET/RF/GBR mean, 20 random 80:20 splits, nested subsets, saturation fitting and 2,000 bootstrap resamples with 90% intervals. |
| `finite_pool` | Six learner configurations over 150 bootstrap resamples; trust-modulated acquisition and GP/RF comparisons with identical initialization. |
| `extrapolation` | Central 20–80% split, three GP experts, three-fold likelihood weights and four acquisition batches. |
| `campaign` | 125 initial observations, 150 bootstraps and 11 recommendations separated by at least `0.15*sqrt(d)` in standardized feature space. |
| `verification` | Four diagnostics from initial measurements and OOF predictions, followed by feature-matched controls. |
| `repeatability` | Fixed cohort IDs joined to subsequent measurements; individual differences, group means and medians. |
| `campaign_outcomes` | Fixed recommendation IDs joined to subsequent measurements; observed threshold count and minimum. |

The ablation configuration specifies fixed local scales for sparse-prior
removal, signed coefficients normalized by their signed sum for removal of
non-negativity, unit family scales, removal of stability screening, and best
single-model selection using training-only OOF error. Each nested variant has
the same outer observations. With uniform family scales, their removal leaves
the model unchanged. Relative change is `100*(R²_variant-R²_full)/R²_full`.
The non-nested alternative fits and scores composition on the same base-model
OOF matrix; its output is separate from the outer-held-out metrics.

Finite-pool initialization requires 20 supplied IDs for `ru_dissolution` or
10 for `polymer_her`. Counts include initialization; discovery means reaching
within 0.1% of the pool optimum. Supply the fixed candidate feature pool.
Each response enters training only after that candidate is selected.

Extrapolation requires 214/92 central/pool observations for formate oxidation
or 406/56 for OER activity. Batch sizes are 23 and 14. The Thompson score is
`s*mean + sd*trust*z`, with an independent standard-normal draw per candidate.
Each complete batch is selected before any of its responses enter training.

Verification takes `benchmark/predictions.csv` and explicit priority IDs.
Choose `absolute_residual` or `residual_over_sd` as the nonconformity measure.
The latter requires positive OOF standard deviations. Priority observations
must have at least two flags; controls must have none. Save the cohort before
passing repeat measurements to `repeatability`. Repeat input columns are
`row_id,repeated`; recommendation measurements use `row_id,measured`.

## Representation and LLM interfaces

An external estimator can enter the same outer LOO benchmark through an
adapter factory taking `random_state` and `configuration`, and returning an
estimator with `fit(X_train, y_train)` and `predict(X_test)`. Any trainable
representation is fitted inside its `fit` method. Frozen descriptor columns
may be supplied directly when their construction is independent of task labels.

For LLM comparisons use the adapter implementation `sparc.llm:make_comparison`.
Its configuration requires `mode` (`standalone` or `hybrid`), `provider`
(an importable `module:function`), `model_snapshot`, `feature_names`, `prompts`
and `decoding`. These values are explicit inputs; there are no default model
snapshots, prompts or decoding settings. `family_priors` is optional.

The provider accepts one request dictionary and returns a parsed JSON object.
Requests contain the model identifier, complete prompt, decoding settings,
output schema and inputs. `prompts` contains `select` and, for standalone mode,
`predict`. Standalone selection returns `feature_indices`, then prediction
returns `predictions`. It does not invoke SPARC's decision modules. Hybrid
selection receives all eight feature strategies and twelve model hypotheses
and returns `feature_strategy` and `model_names`. SPARC then fits those choices
using the same preprocessing, evidence, screening and posterior procedures.
Selection receives only the outer training observations. Requests, responses
and choices are saved with each benchmark fold.
