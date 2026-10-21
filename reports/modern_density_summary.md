# Modern conditional-density and multi-target harness results

Runs: `release_20261020_1653` (density_parity), `release_20261020_2032` (noise_aware_density), `release_20261020_1939` (flow_eiv + likelihood_ratio).  All on CANFAR GPU with shared encoders/training budgets and 3 seeds unless noted.

## 1. Multi-target conditional densities (`density_parity`)

7 datasets × 3 seeds, 60-epoch budget with early stopping.  New methods: cINN-style conditional RealNVP, BPF (Bernstein polynomial flow), CFM (OT-CFM, sampling-only).

| method | n | median NLL | median CRPS | median ES | median Cov90 | median PIT KS | median train_s |
|---|--:|--:|--:|--:|--:|--:|--:|
| torchregress_NSF | 21 | -1.951 | 0.1487 | 0.2241 | 0.890 | 0.091 | 90 |
| torchregress_MAF | 21 | -1.393 | 0.1233 | 0.2388 | 0.889 | 0.095 | 23 |
| torchregress_cINN | 21 | -0.902 | 0.1380 | 0.2388 | 0.893 | 0.080 | 25 |
| torchregress_BPF | 21 | -1.419 | 0.0902 | 0.1586 | 0.981 | 0.162 | 81 |
| torchregress_CFM | 21 | — | 0.1502 | 0.2395 | 0.907 | 0.121 | 11 |
| torchregress_MDN | 21 | -0.614 | 0.1609 | 0.2410 | 0.898 | 0.101 | 25 |

**Reading:** BPF gives the best median CRPS and energy score on these multi-target tables, with conservative coverage (median 0.98) but weaker NLL.  NSF remains the likelihood default.  cINN/MAF have rare extreme-transform failures on `mt_rf1`/`mt_scm20d` (means are unusable; medians are shown).  CFM is ~8× cheaper than NSF but has no exact NLL and is not a calibration replacement.

## 2. Known-label-noise densities (`noise_aware_density`)

2 synthetic datasets × 3 seeds, 60 epochs; evaluation on **clean test labels**.  `_noise` methods use the known Gaussian measurement model.
SMOKE: binned results use the scalar dataset only (multi-target binned head is not defined).

| method | n | mean RMSE | mean NLL | mean CRPS | mean Cov90 | mean PIT-CvM | mean train_s |
|---|--:|--:|--:|--:|--:|--:|--:|
| NSF | 6 | 0.7044 | 1.125 | 0.4041 | 0.934 | 0.00059 | 176 |
| NSF_noise | 6 | 0.7052 | 1.100 | 0.4039 | 0.888 | 0.00067 | 191 |
| MDN | 6 | 0.7025 | 1.102 | 0.4024 | 0.932 | 0.00044 | 29 |
| MDN_noise | 6 | 0.7022 | 1.063 | 0.4013 | 0.888 | 0.00024 | 34 |
| Binned | 3 | 0.6999 | 2.636 | 0.4042 | 0.950 | 0.00100 | 16 |
| Binned_noise | 3 | 0.6980 | 2.642 | 0.4036 | 0.964 | 0.00125 | 17 |
| CFM | 6 | 0.7068 | — | 0.4101 | 0.923 | 0.00301 | 28 |

**Reading:** the analytic MDN convolution gives the largest NLL improvement (1.102 → 1.063) and a small CRPS gain; NSF noise-aware also improves NLL (1.125 → 1.100).  Both noise-aware variants become slightly overconfident (coverage 0.93→0.89), so calibration must be checked separately.  CFM remains sample-competitive but has no likelihood.

## 3. Forward inversion and contrastive normalizing flows

Existing `flow_eiv` and `likelihood_ratio` suites, 3 seeds, 60-epoch budget.  Metrics are averaged over runs.

| method | RMSE / W1 | Coverage / Cov95 | NLL / RefMeanRMSE |
|---|--:|--:|--:|
| ForwardInversion | 0.6494 RMSE | 0.913 | 0.941 NLL |
| FlowEB | 0.5437 RMSE | 0.873 | 0.775 NLL |
| OLS | 0.5400 RMSE | 0.897 | 0.803 NLL |
| ContrastiveFlow | 0.0784 W1 | 0.972 Cov95 | 0.0638 RefMeanRMSE |
| NRE multiclass | 0.0833 W1 | 1.000 Cov95 | 0.0720 RefMeanRMSE |
| NPE | 0.0876 W1 | 0.917 Cov95 | 0.0745 RefMeanRMSE |
| sklearn NRE | 0.914 W1 | 1.000 Cov95 | 0.717 RefMeanRMSE |

**Reading:** `ContrastiveFlow` is the best posterior engine by W1/RefMeanRMSE in this SBI comparison, but it is a likelihood-ratio/ranking method, not a calibrated conditional regression density.  `ForwardInversion` is stable and has good coverage but worse point RMSE than OLS/FlowEB on this linear synthetic design; it is an EIV tool, not a general density win.

## Caveats

- cINN/MAF means are polluted by extreme NLL/CRPS on hard multi-target datasets; use medians/robust summaries.  This is a real instability, not a rounding artefact.
- Binned NLL is a probability mass, not a continuous density; its NLL is not comparable to flow/MDN NLL.  Use CRPS/coverage/PIT for binned heads.
- CFM has no exact likelihood; only sample-based metrics are reported.
- The noise-aware suite is synthetic and scalar/multi-target by design; the real-data PZ benchmarks are separate.
- Existing `density_parity` baselines (sbi/pzflow) were not rerun in this targeted run; only torchregress methods were selected.

Artifacts: `reports/modern_density_results.json`, `reports/noise_aware_results.json`, `reports/forward_inversion_contrastive_results.json`. Raw harness JSONs are in the CANFAR run archives.

## 4. Normalizing Flow Map (`NFM`) — noise-aware suite

Run `release_20261021_1003`: 2 synthetic datasets × 3 seeds, 60 epochs, shared
encoder.  NFM is trained on noisy observed labels; evaluation is on clean test
labels.  The real-data `density_parity` NFM run was skipped by the dataset
loader (network timeout for `mt_enb`), so only the self-contained synthetic
suite is reported here.

| method | n | mean RMSE | mean CRPS | mean Cov90 | mean NLL |
|---|--:|--:|--:|--:|--:|
| NSF | 6 | 0.7044 | 0.4041 | 0.934 | 1.125 |
| MDN | 6 | 0.7025 | 0.4024 | 0.932 | 1.102 |
| **NFM** | **6** | **0.8818** | **0.5064** | **0.857** | **1.300** |

The tabular NFM is underfit at this budget and is clearly worse than NSF/MDN on
these small synthetic problems.  It is a research implementation for exact
likelihood + flow-map few-step transport, not a recommended regression default
yet.  A larger architecture/training budget is required before comparing it
fairly with the mature flows.

## 5. WALDO and conformalized-density status

- `WaldoConformal` and `WaldoPosteriorCalibrator` are implemented and unit
  tested on synthetic Gaussian/posterior data (coverage, scalar intervals,
  multi-target Wald regions).
- `ConformalizedDensity` / `ContrastiveFlowConformal` are implemented and unit
  tested against scalar Gaussian level sets and a contrastive-flow-shaped mock.
- These were not added as harness rows in this round: a proper WALDO harness
  needs a simulator/parameter-grid protocol, and conformal coverage must be
  evaluated on a held-out calibration split.  The library classes are the
  validated building blocks; do not claim benchmark coverage numbers for them
  yet.

## 6. Conformalized ContrastiveFlow benchmark (`tools/conformal_density_bench.py`)

Synthetic scalar and two-target conditional densities, shared MLP encoder,
60 epochs, one seed.  `CTI` = marginal density level-set conformal;
`CTI_Mondrian` = separate thresholds by calibration-target terciles; `Waldo` =
WALDO score with full conditional covariance from samples.  Coverage is measured
on a clean test set with nominal 90% regions.

| dataset | model | test NLL | sample cov90 | CTI cov | CTI width | Mondrian cov | Waldo cov |
|:--|:--|--:|--:|--:|--:|--:|--:|
| scalar | NSF | 1.190 | 0.887 | 0.890 | 2.696 | 0.898 | 0.905 |
| scalar | ContrastiveFlow | 1.161 | 0.888 | 0.905 | 2.740 | 0.910 | 0.903 |
| multi | NSF | 1.810 | 0.880 | 0.863 | — | 0.885 | 0.903 |
| multi | ContrastiveFlow | 1.870 | 0.870 | 0.908 | — | 0.893 | 0.908 |

**ContrastiveFlowConformal works:** its marginal CTI coverage is 0.905
(scalar) and 0.908 (multi), and the scalar density/width is comparable to the
NSF conformal band.  Its test NLL/CRPS are slightly worse on the multi-target
task.

**Improvements found:**
- **Mondrian (group) thresholds** lift the multi-target NSF coverage from
  **0.863 → 0.885** and scalar ContrastiveFlow from **0.905 → 0.910** at a small
  width cost.  This is the cheapest practical improvement.
- **WALDO** reaches 0.903/0.908 coverage with similar width, but it is a
  different Wald region and requires conditional moments; it is a useful
  alternative when the density is overconfident.
- The remaining gap is **conditional** coverage: target-tercile Mondrian is
  coarse.  A proper test needs context/difficulty groups and repeated seeds.
