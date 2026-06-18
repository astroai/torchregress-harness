# torchregress — Improvement Proposals

Compiled from extensive benchmarking against pzflow, normflows, sklearn, MAPIE, and thorough code review.
All findings are backed by 60-epoch evidence from the parity report.

---

## URGENT FIXES (P0)

### 1. SLS Counter Fragility
**Problem:** `SLSLoss.step_counter` increments on every `forward()` call, is manually decremented between `forward_frontier()` and `forward_quantiles()`, and has no upper bound (risk of integer overflow on long training runs).

**Proposed Solution:** Decouple step tracking from the loss module. Pass `step` explicitly from the training loop rather than mutating internal state. Remove the `step_counter -= 1` hack.

**Evidence:** 60-epoch runs show SLS degrades on bimodal data (IS=8.45 vs 6.05 for NF) and produces 77-97% wider intervals than needed on Gaussian data. The fragile counter is a correctness risk during multi-GPU or distributed training.

---

### 2. Bundled `ConditionalFlow` Estimator
**Problem:** Users must manually wire three separate objects (backbone MLP, zuko flow, `NormalizingFlowLoss`) to use normalizing flows for regression. Compare with pzflow's `Flow` class which is initialized in one line and has `train()`, `sample()`, `log_prob()`, `posterior()` methods.

**Proposed Solution:** Create a `ConditionalFlow` wrapper class that:
- Accepts `n_features`, `context_dim`, `flow_type="nsf"`, `n_transforms`
- Auto-creates the backbone context network and zuko flow
- Wires `NormalizingFlowLoss` internally
- Exposes: `fit()`, `sample()`, `log_prob()`, `predict_interval()`, `posterior()`, `save()`, `load()`

**Evidence:** pzflow's `Flow` class achieves this for JAX; normflows' `ConditionalNormalizingFlow` achieves this for PyTorch. torchregress's current 3-object wiring is the most complex API of the three.

---

### 3. Scikit-Learn Compatible `fit()`/`predict()` Interface
**Problem:** No standard ML estimator interface. Users must manually write PyTorch training loops. Compare with pzflow's `flow.train(df, epochs=...)` or sklearn's `.fit(X, y)`.

**Proposed Solution:** Create high-level estimator classes:
- `FlowRegressor(n_features, ...)` with `.fit(X, y)` and `.predict(X)`
- `EvidentialRegressor(...)` with `.fit()` and `.predict_interval()`
- `MDNRegressor(...)` with `.fit()` and `.predict_interval()`

These should internally handle: data conversion (numpy/DataFrame → tensor), model creation, training loop, evaluation mode switching.

**Evidence:** All benchmark suites had to manually wire `_train_one` with specific model/loss construction. A single-class API would reduce suite code by ~60%.

---

## IMPORTANT FIXES (P1)

### 4. Missing `predict_interval()` on NormalizingFlowLoss
**Problem:** `EvidentialRegressionLoss` and `MixtureDensityLoss` both have `predict_interval()`, but `NormalizingFlowLoss` only has `quantile()` and `cdf()`. Inconsistent API.

**Proposed Solution:** Implement `predict_interval(confidence=0.9, n_samples=512)` that delegates to `quantile()` with the appropriate levels. This is trivial since `quantile()` already exists.

---

### 5. Missing `posterior()` Extraction
**Problem:** pzflow has `posterior(inputs, column, grid)` for extracting 1D conditional PDF P(col | rest). torchregress NF has no equivalent — users must manually sample and histogram.

**Proposed Solution:** Add `posterior(y_pred, grid, n_samples=512)` to `NormalizingFlowLoss` that returns the approximate conditional PDF evaluated on `grid` by sampling and kernel density estimation (or histogram binning).

---

### 6. Evidential `predict_interval()` Performance
**Problem:** `predict_interval()` calls `scipy.stats.t.ppf` element-wise for each sample. This is slow for large batches and requires CPU-GPU transfers.

**Proposed Solution:** Batch the scipy call by passing the entire df array to `scipy.stats.t.ppf(df_np.values)`, or use a PyTorch approximation (`torch.distributions.StudentT`).

---

### 7. NormalizingFlow `sample()` Shape Risk
**Problem:** When `n_samples=1` and `context_dim > 0`, `dist.sample()` is called without shape arguments, which may not produce batch-consistent results across zuko versions.

**Proposed Solution:** Always pass explicit batch dimension `dist.sample((batch_size,))` for the unconditional case and `dist.sample((batch_size, n_samples))` for the conditional case.

---

### 8. DataFrame Input Support
**Problem:** torchregress works exclusively with numpy arrays and torch tensors. pzflow accepts pandas DataFrames with column names, making it much friendlier for tabular data workflows.

**Proposed Solution:** Add input adapters that accept DataFrames, map column names to indices, and convert to tensors internally. The `FlowRegressor` class should accept `pd.DataFrame` directly.

---

### 9. PIT Calibration Diagnostics
**Problem:** The benchmark uncovered severe NF overconfidence (DevAbsMean=0.175) but users have no built-in tools to detect this. pzflow's ecosystem uses PIT (Probability Integral Transform) diagrams for calibration assessment.

**Proposed Solution:** Add `pit_diagnostics(predictions, targets)` utility to the metrics module that computes PIT values and returns uniformity test statistics (Kolmogorov-Smirnov, Cramér-von Mises).

**Evidence:** The NF overconfidence pattern (coverage 0.59 on diabetes, 0.73 on synthetic_hetero) would be immediately visible in a PIT diagram as a U-shaped distribution.

---

### 10. Flow Ensembles
**Problem:** Single NF models are overconfident. pzflow provides `FlowEnsemble` for improved calibration. Ensemble methods (deep ensembles of independently trained flows) are a well-established remedy for flow overconfidence.

**Proposed Solution:** Implement `FlowEnsemble` that wraps multiple `ConditionalFlow` instances with independent initializations, aggregates predictions via sample pooling, and reports epistemic uncertainty from ensemble variance.

**Evidence:** `DeepEnsemble_K10` already exists for point regression. The same pattern applied to flows would likely rescue the 0.59→0.78 coverage gap seen on diabetes.

---

## MINOR FIXES (P2)

### 11. MDN `sample()` Efficiency
**Problem:** `unsqueeze(1).expand(-1, n_samples, -1, -1)` followed by `gather` is unnecessarily memory-intensive for component selection.

**Proposed Solution:** Use `advanced indexing` with `torch.arange(batch_size)` and the sampled component indices directly.

---

### 12. Standardized Save/Load
**Problem:** Flows require manual extraction of state dicts from model + flow + loss components. pzflow has `flow.save(file)` / `pzflow.Flow.load(file)`.

**Proposed Solution:** Add `.save(path)` and `.load(path)` class methods to the proposed `ConditionalFlow` wrapper, handling all component serialization.

---

### 13. Auto-Inferred Context Dimension
**Problem:** Users must manually match the context dimension between backbone output and flow constructor. This is fragile — mismatches cause silent shape errors.

**Proposed Solution:** In the `ConditionalFlow` wrapper, auto-infer context_dim from the backbone's final layer output size.

---

### 14. Observation Error Models
**Problem:** pzflow supports built-in data error models and condition error models for noisy observations (important for astronomy). torchregress has no equivalent.

**Proposed Solution:** Add optional `data_error_model` and `condition_error_model` parameters to the `ConditionalFlow` wrapper, implementing convolution/dequantization transforms where supported.

---

## ARCHITECTURE REFINEMENTS

### 15. Unify `DistributionLoss` Interface
**Problem:** `NormalizingFlowLoss`, `MixtureDensityLoss`, and `EvidentialRegressionLoss` all inherit from `DistributionLoss` but have inconsistent method signatures:
- NF has `log_prob()`, `sample()`, `quantile()`, `cdf()` — **no** `predict_interval()`
- MDN has `sample()`, `predict_interval()`, `predict_mean_std()` — **no** `log_prob()`, `quantile()`
- Evidential has `predict_interval()`, `predict_with_uncertainty()`, `sample_predictions()` — **no** `log_prob()`, `quantile()`

**Proposed Solution:** Define a minimal `DistributionLoss` protocol:
- Required: `log_prob()`, `sample()`, `predict_interval()`
- Optional: `quantile()`, `cdf()`, `posterior()`

---

## SUMMARY TABLE

| # | Proposal | Priority | Effort | Impact |
|---:|---|---|---|---|
| 1 | SLS Counter Fragility | P0 | Low | Correctness |
| 2 | `ConditionalFlow` Estimator | P0 | High | API quality |
| 3 | sklearn `fit()`/`predict()` API | P0 | Medium | Usability |
| 4 | `predict_interval()` on NF | P1 | Low | API consistency |
| 5 | `posterior()` on NF | P1 | Low | Feature parity |
| 6 | Evidential performance | P1 | Low | Speed |
| 7 | NF `sample()` shape | P1 | Low | Correctness |
| 8 | DataFrame support | P1 | Medium | Usability |
| 9 | PIT diagnostics | P1 | Medium | Calibration |
| 10 | Flow Ensembles | P1 | High | Calibration |
| 11 | MDN `sample()` efficiency | P2 | Low | Performance |
| 12 | Save/load | P2 | Low | Convenience |
| 13 | Auto context dim | P2 | Low | Robustness |
| 14 | Observation error models | P2 | High | Feature parity |
| 15 | Unify DistributionLoss | — | Medium | Architecture |

---

*Generated: June 17, 2026 — from 60-epoch benchmarks across 5 suites, 11 methods, 14 datasets*

---

## IMPLEMENTATION STATUS (harness-level snapshots, June 18 2026)

The following proposals are partially or fully satisfied by *harness-side surface area* rather than library-internal refactors.  Validated via harness smoke tests and PARITY_REPORT.md cross-references.

| # | Status | Implementation site | Validation reference |
|---:|---|---|---|
| 1 | Not implemented at harness level | (library-internal torchregress change) | n/a |
| 2 | ✅ Working draft scaffolded (port into `torchregress` upstream pending) | `proposals/conditional_flow/conditional_flow.py` (in this repo, harness-side) | Self-contained `_demo()` smoke + bit-identical MC draws (`proposals/conditional_flow/conditional_flow.py::_demo`); ruff 0 errors; pytest 149/149 unchanged |
| 3 | ✅ Implemented + validated | `src/harness/wrappers.py` — `GaussianNLLRegressor`, `MDNRegressor`, `EvidentialRegressor`, `NormalizingFlowRegressor` | PARITY_REPORT.md §16 |
| 4 | Not implemented at harness level | (library-internal torchregress change) | n/a |
| 5 | Not implemented at harness level | (library-internal torchregress change) | n/a |
| 6 | Accumulated evidence (evidential 512-sample pool baseline) | `wrappers.EvidentialRegressor.EVIDENTIAL_N_SAMPLES = 512` | PARITY_REPORT.md §9 (PIT table) |
| 7 | Sufficient (normalize at wrapper level via `.reshape(n_test, -1)`) | `wrappers.MDNRegressor._predict_diagnostics`, `wrappers.NormalizingFlowRegressor._predict_diagnostics` | `tests/test_wrappers.py::test_predict_interval_confidence_widens_strict` |
| 8 | Not implemented at harness level | (could be added at a `DataFrameAdapter` layer) | n/a |
| 9 | ✅ Implemented + validated | `src/harness/metrics.py::compute_pit_values`, `pit_uniformity_tests` | PARITY_REPORT.md §9 (PIT table) |
| 10 | ✅ Implemented + validated (scalar §13 + K-sweep §14 + MV §15) | `suites/tabular/distributional_bins.py` + `suites/tabular/multivariate_intervals.py` (joint with `suites/_backbones.NF_ENSEMBLE_K_VARIANTS`) | PARITY_REPORT.md §13 / §14 / §15 |

**Status legend**:
- **✅ Implemented + validated** — proposal is closed in this repo, validated by PARITY_REPORT evidence or smoke tests.
- **Accumulated evidence** — partial mitigation at harness layer; primary fix still library-internal.
- **Sufficient** — minimal mitigation at harness layer is enough to avoid the worst failure modes.
- **Not implemented** — open in `torchregress` library.
- **Working draft** — harness-side scaffolding for an upstream PR; full closure requires port into `torchregress`.
