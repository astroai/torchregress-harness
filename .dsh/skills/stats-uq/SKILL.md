---
name: stats-uq
description: Statistics-principled uncertainty quantification — conformal coverage checks, proper scoring rules, calibration plots, PIT diagnostics, post-hoc calibration
metadata:
  domains: statistics, uncertainty-quantification, astronomy
---

# Stats-principled UQ

The ecosystem's UQ standard (extends science-core): **calibration over
confidence**, proper scoring, honest coverage, reproducibility. torchregress
is the shared implementation (losses, metrics, calibration, viz); cosmodist
reuses it through its own wrappers (`src/cosmodist/validation/`). API names
below are the current public surface — verify signatures with `help()` /
docstrings against the installed versions before writing code.

## 1. Report calibrated intervals with conformal prediction

Split conformal (CQR-style), calibration on a held-out split ONLY:

```python
from torchregress.losses.conformal import SplitConformal

cal = SplitConformal(alpha=0.1)                # 90% intervals
cal.calibrate(y_pred_cal, y_cal)               # calibration split — never train
lower, upper = cal.predict_interval(y_pred_test)   # (Tensor, Tensor)
```

In cosmodist, use the existing wrappers (they wrap `CQR`, not
`SplitConformal` — `CQR` calibrates on a (lower, upper) **quantile pair**):

```python
from cosmodist.validation.calibration import (
    conformalize_quantiles,
    validate_conformal_coverage,
)

# fit on a calibration split of labels, for the outer quantile pair
cqr = conformalize_quantiles(
    cal_predictions,          # [N_cal, n_quantiles] field predictions at labels
    cal_targets,              # [N_cal]
    quantile_indices=(0, -1), # outermost pair of the z/μ quantile field
    target_coverage=0.9,
)
# verify on held-out TEST tiles (never the calibration split)
metrics = validate_conformal_coverage(
    predictions, targets, cqr,
    quantile_indices=(0, -1), target_coverage=0.9,
)
# metrics["coverage_valid"] is True iff |empirical - target| < 0.05
```

Rules:

- The calibration split is held out; tuning `alpha` (or `target_coverage`)
  on the test set is leakage and must never happen.
- Report `empirical_coverage_*` AND `mean_width_*`/`median_width_*` —
  coverage alone is meaningless (a trivially wide interval hits any
  coverage). The wrappers return all of these; use them.
- For multimodal predictive densities, `LevelSetConformalPredictor` beats
  interval CQR; read its docstring before defaulting to interval form.

## 2. Coverage checks (non-negotiable)

Verify on the TEST set, never the calibration set:

```python
from torchregress.metrics.interval import (
    interval_score,                              # Winkler score (sharpness + coverage penalty)
    prediction_interval_coverage_probability,
    interval_metrics_report,
)
from torchregress.metrics.distribution import (
    probability_integral_transform,
    kolmogorov_smirnov_uniform_statistic,
)
```

- Empirical coverage must match the stated alpha within sampling error.
- PIT check: for a calibrated predictive distribution, PIT values are
  ~ Uniform(0,1); test with the KS statistic. U-shaped PIT = overconfidence,
  ∩-shaped = underconfidence.
- **cosmodist is a field model** — marginal coverage over pixels can hide
  spatial miscalibration. Also check coverage per spatial bin and report
  `compute_spatial_coherence` (`cosmodist.validation.distance_field`) so a
  science claim is not propped up by well-calibrated regions masking
  miscalibrated ones.

## 3. Proper scoring rules

- Full predictive distribution: CRPS (`ContinuousRankedProbabilityScore`
  metric, or closed-form `crps_gaussian(mean, y_true, std)`) and NLL
  (`gaussian_nll(mean, y_true, var)`).
- Quantile heads: `MultiQuantileLoss` for training (cosmodist supervises
  every quantile of the z/μ fields where sparse labels exist), paired with
  `QuantileCrossoverLoss` to prevent quantile crossing.
- A point metric (MSE/MAE) is never the headline for a science claim — the
  field metrics in `compute_field_redshift_metrics` /
  `compute_distance_field_metrics` must be paired with interval coverage
  and PIT, not reported alone.

## 4. Calibration diagnostics and plots

- Metrics: `ExpectedCalibrationError(n_bins=10)` (update with
  `y_pred_quantiles: Dict[float, Tensor]`, `y_true`) and
  `MarginalCalibrationError(n_bins=20)` (update with samples).
- Plots (`torchregress.viz.diagnostic`): `plot_reliability_diagram`,
  `plot_qq_plot`, `plot_residuals`, `plot_prediction_intervals`,
  `plot_residual_histogram` — check docstrings for expected argument shapes.
- Post-hoc fixes (fit on the CALIBRATION split, apply to test — never refit
  on test): `VarianceTemperatureScaler`, `IsotonicMeanCalibrator`,
  `PITCalibrator`. In cosmodist this belongs in a calibrate step
  (`scripts/calibrate_scene`), fit once and frozen before evaluation.

## 5. Prediction-powered inference (PPI) for sparse-label readouts

cosmodist's cosmology readout is the canonical PPI case: distance-indicator
labels are few, model predictions are dense. Use torchregress PPI so the
dense predictions carry power and the sparse labels debias — do NOT report
a naive mean over labeled points as the science result.

```python
from cosmodist.validation.cosmology_ppi import ppi_om_scan

res = ppi_om_scan(
    mu_labeled=..., mu_model_labeled=...,
    mu_model_unlabeled=..., z_labeled=..., z_unlabeled=...,
    om_grid=np.linspace(0.15, 0.5, 36),   # default
)
# res["om_best"]: PPI-preferred Om (smallest |mean Hubble residual|)
# res["om_interval"]: Om range whose residual CI covers zero (or None)
# res["residual_estimate"/"residual_lower"/"residual_upper"]: CI per Om
```

Under the hood it calls `torchregress.inference.ppi.ppi_mean_ci`; more
advanced variants exist (`ppi_calibrated_mean_ci`, `ppi_quantile_ci`).
Report `n_labeled`/`n_unlabeled` and CI widths with every PPI number, and
never present the point estimate without its interval.

## Application per repo

- **cosmodist** — the field-level UQ discipline: every science claim about
  the z/μ/SED fields must carry conformalized intervals validated on
  held-out TEST tiles (via `conformalize_quantiles` +
  `validate_conformal_coverage`), spatial coverage checks
  (`compute_spatial_coherence`), and PPI-based cosmology readouts
  (`ppi_om_scan`). Sparse supervision means calibration splits are precious:
  state the split scheme exactly and reuse it across runs. Do not tune
  `target_coverage` on test; the `coverage_valid` gate is a floor, not a
  target to game.
- **torchregress** — the canonical implementations; every new loss/metric
  ships with a test that verifies calibration empirically (simulated ground
  truth where possible). Generic UQ machinery belongs here (upstream-first:
  add it as a PR to torchregress, not as a cosmodist-local copy).
- **torchz** — photo-z heads (gaussian / quantile / binned_pdf) must report
  sigma_nmad and outlier fraction AND interval coverage + PIT calibration;
  n(z) work uses proper density scoring, not histogram eyeballing.
- **uspm** — latent representations are not point estimates: report
  predictive spread and marginal calibration on held-out spectra/photometry
  before any science claim (rest-frame SEDs, population studies).

## Guardrails

- Never report bare point estimates for science claims (fields or
  cosmology readouts).
- Never tune on the test set; state the split scheme exactly.
- Report all trials and seeds; no p-hacking or cherry-picked bins.
- Every number in a report must be reproducible from the session log +
  provenance manifests.
