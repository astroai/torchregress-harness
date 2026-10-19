# CANFAR GPU benchmark results (2026-10-18)

Headless CANFAR sessions on `images.canfar.net/astroai/base:latest`, one GPU
each, `--device cuda`. Every artifact below reports
`environment.cuda_available: true` and `config.device: "cuda"`; real-data jobs
report `SyntheticFallback: false`.

- Harness: `ee41b02` + `0dc4029` (FrEIA suite, calibration CUDA fix)
- torchregress: `66d52fe`
- Runs: `release_20261018_0805` (image + tabular), `release_20261018_0911`
  (calibration rerun), `release_20261018_0944` (Gaussian-mixture SBI),
  `release_20261018_1034` (Bernoulli-GLM SBI), `release_20261018_1222`
  (CNF reproduction), `release_20261018_1354` (VIR + new conformal comparators),
  `release_20261018_1503` (Engression, UQ-shift, flow-based EIV);
  two Gaussian-linear SBI attempts
  (`release_20261018_1033`, reduced-method `..._1137`) were stopped after
  >1 h without an artifact.
- Single seed per job (`seed 0`), 60 epochs where applicable; flexible or
  4-core GPU slices were used. These are comparative snapshots, not
  statistically converged rankings.

See [BENCHMARK_RESEARCH.md](BENCHMARK_RESEARCH.md) for what the original
papers used and why a given comparator was selected.

## Established-package SBI parity (`sbi_parity`, sbibm tasks)

Lower C2ST is better (0.5 = indistinguishable from the reference posterior).
`SBC95` is the marginal 95% coverage; `SBI_minp` is sbi's own SBC diagnostic.

**sbibm Two Moons**

| method | library | C2ST | MeanRMSE | SBC95 | SBI minp | train s |
|---|---|--:|--:|--:|--:|--:|
| torchregress_NSF | torchregress | 0.504 | 0.0127 | 0.980 | 0.114 | 356 |
| torchregress_MAF | torchregress | 0.691 | 0.0464 | 0.930 | 0.000 | 146 |
| torchregress_MDN | torchregress | 0.565 | 0.0132 | 0.995 | 0.022 | 80 |
| pzflow_NSF | pzflow | 0.536 | 0.0602 | 0.960 | 0.005 | 25 |
| sbi_NPE | sbi | 0.494 | 0.0207 | 0.940 | 0.050 | 109 |
| sbi_NPE_MAF | sbi | 0.652 | 0.0355 | 0.955 | 0.076 | 41 |
| sbi_NPE_MDN | sbi | 0.492 | 0.0298 | 0.945 | 0.007 | 28 |
| sbi_FMPE | sbi | 0.670 | 0.0308 | 0.920 | 0.018 | 29 |
| sbi_NPSE | sbi | 0.525 | 0.0148 | 0.965 | 0.008 | 111 |

**sbibm SLCP**

| method | C2ST | MeanRMSE | SBC95 | SBI minp | train s |
|---|--:|--:|--:|--:|--:|
| torchregress_NSF | 0.872 | 0.972 | 0.916 | 0.005 | 356 |
| torchregress_MAF | 0.930 | 1.349 | 0.964 | 0.000 | 148 |
| torchregress_MDN | 0.936 | 1.497 | 0.960 | 0.014 | 81 |
| pzflow_NSF | 0.855 | 1.199 | 0.930 | 0.012 | 96 |
| sbi_NPE | 0.855 | 1.133 | 0.942 | 0.032 | 118 |
| sbi_NPE_MAF | 0.906 | 1.037 | 0.906 | 0.000 | 51 |
| sbi_NPE_MDN | 0.867 | 0.967 | 0.900 | 0.094 | 94 |
| sbi_FMPE | 0.825 | 0.973 | 0.904 | 0.009 | 57 |
| sbi_NPSE | 0.797 | 0.719 | 0.930 | 0.050 | 179 |

**sbibm Gaussian Mixture**

| method | C2ST | MeanRMSE | SBC95 | SBI minp | train s |
|---|--:|--:|--:|--:|--:|
| torchregress_NSF | 0.644 | 0.226 | 0.910 | 0.094 | 369 |
| torchregress_MAF | 0.691 | 0.173 | 0.900 | 0.012 | 140 |
| torchregress_MDN | 0.514 | 0.122 | 0.935 | 0.551 | 81 |
| pzflow_NSF | 0.579 | 0.135 | 0.920 | 0.253 | 26 |
| sbi_NPE | 0.502 | 0.077 | 0.940 | 0.103 | 58 |
| sbi_NPE_MAF | 0.685 | 0.215 | 0.905 | 0.014 | 42 |
| sbi_NPE_MDN | 0.561 | 0.184 | 0.950 | 0.197 | 24 |
| sbi_FMPE | 0.564 | 0.106 | 0.910 | 0.487 | 78 |
| sbi_NPSE | 0.545 | 0.156 | 0.925 | 0.180 | 37 |

**sbibm Bernoulli GLM**

| method | C2ST | MeanRMSE | SBC95 | SBI minp | train s |
|---|--:|--:|--:|--:|--:|
| torchregress_NSF | 0.744 | 0.735 | 0.937 | 0.001 | 366 |
| torchregress_MAF | 0.732 | 0.637 | 0.956 | 0.001 | 148 |
| torchregress_MDN | 0.760 | 0.914 | 0.944 | 0.011 | 79 |
| pzflow_NSF | 0.802 | 0.512 | 0.908 | 0.002 | 267 |
| sbi_NPE | 0.675 | 0.465 | 0.962 | 0.002 | 111 |
| sbi_NPE_MAF | 0.678 | 0.486 | 0.941 | 0.076 | 45 |
| sbi_NPE_MDN | 0.673 | 0.646 | 0.960 | 0.040 | 59 |
| sbi_FMPE | 0.654 | 0.328 | 0.942 | 0.000 | 34 |
| sbi_NPSE | 0.664 | 0.629 | 0.955 | 0.009 | 62 |

**sbibm Gaussian Linear**: not completed in this campaign. Two GPU runs
(the full 9-method suite and a reduced 6-method suite) exceeded one hour
without producing an artifact, while the four lighter sbibm tasks finished
in 25–50 min each. The likely cause is one of the 10-D-target comparator
trainings stalling rather than a GPU failure; this task remains open for a
dedicated run with per-method timeouts. The result is reported as missing,
not as a parity claim.

## Real image data (`cell_counting`, BBBC005 fluorescence images)

GPU, real counts (no synthetic fallback), `bbbc_cellcount`:

| method | RMSE | MAE | R² | train s |
|---|--:|--:|--:|--:|
| CNN+MSE | 4.777 | 3.469 | 0.975 | 20.1 |
| CNN+Poisson | 7.598 | 6.115 | 0.937 | 22.0 |
| CNN+PoissonLR | 5.875 | 4.521 | 0.962 | 21.9 |
| CNN+ZIP | 4.994 | 3.955 | 0.973 | 23.2 |
| CNN+NegBin | 13.421 | 9.437 | 0.802 | 23.4 |
| CNN+GaussianNLL | 5.785 | 4.216 | 0.963 | 19.6 |
| CNN+PoissonGaussian | 7.401 | 5.284 | 0.940 | 22.9 |
| CNN+EnhancedPG_fixed | 5.360 | 4.211 | 0.968 | 23.3 |
| CNN+EnhancedPG_learned | 7.597 | 5.908 | 0.937 | 23.6 |
| CNN+PoissonGaussianLR | 5.764 | 4.736 | 0.963 | 22.7 |
| sklearn_RF | 12.046 | 8.622 | 0.841 | 29.3 |

## Real tabular data

### EIV/LatentNN methods on HSC photo-z EIV (`photoz_eiv`)

| method | RMSE | MAE | R² | Attenuation | train s |
|---|--:|--:|--:|--:|--:|
| NoiseAware | **0.063** | **0.035** | **0.988** | 0.986 | 92 |
| MSE | 0.124 | 0.073 | 0.953 | 0.950 | 33 |
| EnsembleEIV | 0.150 | 0.096 | 0.932 | 1.009 | 56 |
| InputNoiseBinnedPDF | 0.150 | 0.087 | 0.932 | 0.905 | 57 |
| RC | 0.152 | 0.096 | 0.930 | 0.933 | 49 |
| InputNoiseMC | 0.160 | 0.098 | 0.922 | 0.898 | 54 |
| MSE_WarmupMC | 0.167 | 0.090 | 0.915 | 0.905 | 106 |
| FunctionalEIV | 0.175 | 0.108 | 0.907 | 0.858 | 151 |
| SIMEX | 0.267 | 0.184 | 0.783 | 1.085 | 1182 |
| ODR | 0.372 | 0.335 | 0.577 | 0.594 | 557 |
| SciPyODR (external) | 3.949 | 3.184 | −46.5 | 0.805 | 6 |
| TLS / TLS_Init+FunctionalEIV | skipped (ill-conditioned design) | | | | |

### FrEIA cINN vs conditional density estimators on real tabular data

| dataset | method | RMSE | NLL | CRPS | train s |
|---|---|--:|--:|--:|--:|
| diabetes | GaussianNLL | 53.12 | 5.545 | 30.40 | 2.1 |
| diabetes | NSF | 55.95 | 5.720 | 31.73 | 11.3 |
| diabetes | FrEIA_cINN | 74.17 | 5.728 | 43.05 | 4.8 |
| california_housing | GaussianNLL | 0.642 | 40.00 | 12.29 | 16.5 |
| california_housing | NSF | **0.573** | **0.822** | **0.269** | 118 |
| california_housing | FrEIA_cINN | 1.156 | 1.565 | 0.644 | 49 |
| openml_year | GaussianNLL | **0.159** | **−0.465** | **0.083** | 32 |
| openml_year | NSF | 0.162 | 6.560 | 0.083 | 243 |
| openml_year | FrEIA_cINN | 0.353 | 0.379 | 0.202 | 98 |

The FrEIA cINN is a genuine external framework (MIT, GPU) and trains
successfully, but under the shared budget the torchregress NSF is stronger on
the two smaller tabular problems; Gaussian NLL wins the large OpenML-year set.

### Imbalanced regression (`stress_imbalanced` synthetic + real California housing)

| dataset | method | RMSE | TailRMSE | WorstGroupMAE | train s |
|---|---|--:|--:|--:|--:|
| stress_imbalanced | MSE | 0.596 | 1.033 | 0.616 | 11 |
| stress_imbalanced | GroupDRO | 0.671 | 1.014 | 0.592 | 4 |
| stress_imbalanced | PRIME | 0.617 | 1.036 | 0.592 | 34 |
| stress_imbalanced | ConR | 0.615 | 1.028 | **0.578** | 29 |
| california_housing | MSE | **0.600** | 1.226 | 0.850 | 12 |
| california_housing | GroupDRO | 0.718 | **1.144** | **0.611** | 4 |
| california_housing | PRIME | 0.612 | 1.242 | 0.898 | 38 |
| california_housing | ConR | 0.620 | 1.211 | 0.873 | 31 |

### Likelihood-ratio/SBI suite on GPU (exact-reference synthetic tasks)

**Gaussian linear**

| method | PostMeanRMSE | MLE RMSE | W1 | Cov95 | SBC min p | TARP err | ESS | train s |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| torchregress_NRE | 0.225 | 0.421 | **0.046** | 1.00 | 0.129 | 0.060 | 157 | 24 |
| torchregress_NRE_multiclass | 0.233 | 0.401 | 0.056 | 1.00 | 0.101 | 0.055 | 154 | 23 |
| torchregress_ContrastiveFlow | **0.219** | 0.475 | 0.060 | 1.00 | 0.083 | 0.080 | 153 | 345 |
| torchregress_NPE | 0.254 | — | 0.066 | 1.00 | 0.026 | 0.065 | — | 145 |
| sklearn_NRE | 1.312 | 1.622 | 1.392 | 1.00 | 0.060 | 0.060 | 1680 | 0.04 |

**Bimodal**

| method | PostMeanRMSE | MLE RMSE | W1 | Cov95 | SBC min p | TARP err | ESS | train s |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| torchregress_NRE | 0.999 | 1.655 | 0.122 | 0.83 | 0.364 | 0.060 | 216 | 24 |
| torchregress_NRE_multiclass | 0.979 | 1.396 | 0.108 | 1.00 | 0.121 | 0.055 | 226 | 23 |
| torchregress_ContrastiveFlow | 0.901 | 1.299 | **0.071** | 1.00 | 0.667 | **0.030** | 277 | 371 |
| torchregress_NPE | **0.882** | — | 0.081 | 0.67 | 0.598 | 0.035 | — | 148 |
| sklearn_NRE | 0.951 | 1.321 | 0.480 | 1.00 | 0.084 | 0.050 | 1680 | 0.03 |

### Calibration (GPU, including CUDA isotonic fix)

Selected `california_housing` and `synthetic_hetero` rows (`MACE` lower is
better):

| dataset | method | RMSE | MACE | NLL |
|---|---|--:|--:|--:|
| california_housing | Uncalibrated | 0.612 | 0.0135 | 22.13 |
| california_housing | VTS_Floor_Clip | 0.612 | 0.0189 | **0.942** |
| california_housing | IsotonicMeanCalibrator | **0.610** | 0.0149 | 20.74 |
| california_housing | VarianceRestoring | 0.630 | **0.0079** | 23.90 |
| california_housing | sklearn_IsotonicRegression | 0.610 | 0.0149 | 20.74 |
| synthetic_hetero | Uncalibrated | 0.381 | 0.0207 | 0.527 |
| synthetic_hetero | VarianceTemperatureScaler | 0.381 | 0.0130 | 0.457 |
| synthetic_hetero | VarianceRestoring | 0.404 | 0.0223 | 0.601 |

The GPU run also exposed and fixed a real defect: the isotonic calibrator's
transform path indexed a CUDA tensor with CPU indices; it now runs on CPU in
the harness and is covered by the CUDA rerun.

## Contrastive Normalizing Flow reproduction (paper method, GPU)

`cnf_parity`, run `release_20261018_1222`, `config.device=cuda`.  The suite
implements the paper's contrastive loss with clamping at `k=-50`, `c=2` for
the signal flow and `c=0.5` for the background flow, and the
`Phi = p/(1+p)` DNN classifier.  The paper's 30-layer gated-affine/quadratic
coupling architecture is approximated with zuko NSF couplings, so this is a
method reproduction, not a numerical replication of the paper's tables.

**Two-Gaussian toy (2-D, 4200 train / 1800 test), rotation nuisance**

| method | clean acc | clean AUC | rot15 acc | rot30 acc | rot45 acc | rot30 AUC | train s |
|---|--:|--:|--:|--:|--:|--:|--:|
| DNN | 0.811 | 0.896 | 0.733 | 0.613 | 0.541 | 0.716 | 4.5 |
| NF_NLL | 0.802 | 0.887 | 0.743 | 0.613 | 0.545 | 0.691 | 84.7 |
| CNF | 0.809 | 0.895 | 0.725 | 0.613 | 0.536 | 0.706 | 122.2 |

At this architecture/budget the CNF's rotation robustness is **not** better
than the plain DNN — an honest negative result relative to the paper's toy
claim. The paper's custom gated/quadratic coupling and its full training recipe
remain candidates for closing that gap.

**FAIR-Universe HiggsML sample (1000 real simulated events, 28 features; 700 train / 300 test)**

| method | accuracy | AUC | log-loss | train s |
|---|--:|--:|--:|--:|
| DNN | 0.793 | 0.843 | 0.502 | 0.9 |
| NF_NLL | 0.480 | 0.659 | 3.580 | 10.2 |
| CNF | **0.807** | 0.842 | **0.493** | 14.5 |

The CNF is competitive with the DNN on the real sample and clearly better than
the two independent NLL flows, but it does not establish a parity or
uncertainty-coverage claim. The full 15.1 GB challenge data and the paper's
Neyman confidence-belt pipeline are a documented follow-up.

## VIR and new conformal comparators (GPU)

Run `release_20261018_1354`, `config.device=cuda`. torchregress `f9f71f2`,
harness `04bb08e`.

**VIR on imbalanced regression** (`imbalanced_representation`)

| dataset | method | RMSE | WorstGroupMAE | TailRMSE | NLL | CRPS |
|---|---|--:|--:|--:|--:|--:|
| stress_imbalanced | MSE | 0.596 | 0.616 | 1.033 | — | — |
| stress_imbalanced | GroupDRO | 0.671 | 0.592 | 1.014 | — | — |
| stress_imbalanced | PRIME | 0.617 | 0.592 | 1.036 | — | — |
| stress_imbalanced | ConR | 0.615 | **0.578** | 1.028 | — | — |
| stress_imbalanced | VIR | 0.613 | 0.607 | 1.071 | 0.939 | 0.329 |
| california_housing | MSE | **0.600** | 0.850 | 1.226 | — | — |
| california_housing | GroupDRO | 0.718 | **0.611** | **1.144** | — | — |
| california_housing | PRIME | 0.612 | 0.898 | 1.242 | — | — |
| california_housing | ConR | 0.620 | 0.873 | 1.211 | — | — |
| california_housing | VIR | 0.602 | 0.800 | 1.152 | 0.783 | 0.297 |

VIR gives the best real-data worst-region MAE/tail tradeoff among the
loss/representation methods while also producing NLL/CRPS/coverage; on the
synthetic stress target, representation losses remain slightly better in the
extra tail. These are single-seed observations.

**Conformal parity — coverage at nominal 90% (GPU host, sklearn rows are CPU work)**

| dataset | puncc split | puncc CQR | native CQR reference | conformal-tights CQR | torchregress Split+MLP |
|---|--:|--:|--:|--:|--:|
| california_housing | 0.901 | 0.921 | 0.921 | **0.847** | 0.913 |
| diabetes | 0.925 | 0.940 | 0.940 | **0.774** | 0.940 |
| synthetic_hetero | 0.925 | 0.912 | 0.912 | **0.853** | 0.920 |

puncc (`deel.puncc` 0.9.3) and the independent CQR reference agree exactly on
matched GBR bases, which validates the CQR implementation. concrete-tights
coherent CQR **under-covers** on these harness splits at nominal 90% (0.77–0.85)
and is reported raw rather than tuned; its own calibration carve is documented
in the row. puncc is installed `--no-deps` because it pins `numpy<2` while
torchregress requires numpy>=2.5; its regression wrappers smoke-test fine on the
numpy 2 stack but that combination is upstream-unsupported.

## New regression benchmarks (GPU)

Run `release_20261018_1503`, `config.device=cuda`, harness `e3f1b04`,
torchregress `f9f71f2`.

### Engression vs torchregress (`engression_parity`)

Official `engression` 1.1.0, shared MLP capacity/budget (Engression's official
lr/epochs/BN defaults deliberately overridden and documented):

| dataset | method | RMSE | NLL | CRPS | Cov90 | train s |
|---|---|--:|--:|--:|--:|--:|
| california_housing | Engression | 0.641 | 1.110 | 0.323 | 0.921 | 34 |
| california_housing | MSE_MLP | 0.582 | — | — | — | 6 |
| california_housing | GaussianNLL | 0.627 | 2.034 | 0.372 | 0.915 | 16 |
| california_housing | Quantile_MLP | 0.600 | — | — | 0.877 | 15 |
| california_housing | NormalizingFlow_NLL | **0.570** | **0.609** | **0.273** | 0.843 | 123 |
| synthetic_hetero | Engression | 0.384 | **0.486** | 0.216 | 0.933 | 14 |
| synthetic_hetero | GaussianNLL | 0.383 | 0.572 | **0.212** | 0.800 | 7 |
| synthetic_hetero | Quantile_MLP | **0.369** | — | — | 0.875 | 7 |
| stress_noisy_inputs | Engression | **0.455** | **0.737** | **0.268** | 0.991 | 31 |
| stress_noisy_inputs | GaussianNLL | 0.483 | 0.741 | 0.278 | 0.926 | 16 |
| stress_noisy_inputs | Quantile_MLP | 0.476 | — | — | 0.949 | 15 |

Engression is competitive/strong on the noisy-feature stress set and the
synthetic heteroscedastic set, but on California housing the torchregress flow
is clearly better on NLL/CRPS. These are proxy datasets for the paper's
pre-additive-noise extrapolation setup, not a reproduction of its artificial
extrapolation benchmark.

### Regression UQ under shift (`uq_shift`)

Gustafsson et al. (TMLR 2023) motivation; synthetic_hetero with mean shift
(`x0 += 1.5 std_train`) and test-time heteroscedastic label noise:

| shift | method | RMSE | NLL | Cov95 | NLL gap | Cov95 gap |
|---|---|--:|--:|--:|--:|--:|
| id | GaussianNLL | 0.383 | 0.572 | 0.877 | 0 | 0 |
| id | MDN_K3 | 0.393 | 0.546 | 0.877 | 0 | 0 |
| mean shift | GaussianNLL | 1.118 | 5.114 | 0.375 | +4.54 | −0.50 |
| mean shift | MDN_K3 | 1.127 | 5.144 | 0.282 | +4.60 | −0.60 |
| noise shift | MSE | 0.600 | 1.378 | 0.782 | +0.90 | −0.14 |
| noise shift | GaussianNLL | 0.597 | 1.541 | 0.733 | +0.97 | −0.14 |

The expected result is reproduced: all probabilistic heads become badly
overconfident under covariate/mean shift, while test-time noise degrades
coverage less sharply. OOD-gap columns are in the artifact for every dataset
(California housing, stress_ood, synthetic_eiv, synthetic_hetero).

### Astronomical flow-EIV (`flow_eiv`)

Synthetic bimodal latent, correlated heteroscedastic errors, linear relation
y = 1.25 x + 0.40; ID = 500, OOD = 300 beyond the training support.

| method | ID RMSE | ID slope | ID Cov90 | OOD RMSE | OOD slope | OOD Cov90 |
|---|--:|--:|--:|--:|--:|--:|
| OLS | 0.544 | 0.905 | 0.884 | 1.257 | 0.893 | 0.413 |
| WLS | 0.562 | 0.870 | 0.338 | 1.241 | 0.860 | 0.050 |
| ForwardInversion | 0.656 | 1.265 | 0.906 | **0.906** | **1.250** | **0.893** |
| FlowEB | **0.541** | **1.208** | 0.868 | 1.695 | 0.714 | 0.210 |

FlowEB removes most in-distribution slope attenuation (−3.3% vs −27.6% OLS)
and has the best ID RMSE/NLL/CRPS; its prior shrinkage hurts OOD
extrapolation, where forward inversion is best. This is a documented simplified
ML-EB approximation of Jing & Li (AJ 2025), not the paper's `raddest` code.

## BMC and RankSim (GPU)

Run `release_20261018_1626`, harness `544abcd`, torchregress `7707e0b`;
`config.device=cuda`, shared MLP budget, single seed. BMC uses a fixed
normalized target-noise scale `sigma=0.15*std(y)` for comparability (the
library also supports a fitted prior and optional sigma); RankSim uses
`gamma=1` (mean reduction).

| dataset | method | RMSE | WorstGroupMAE | TailRMSE |
|---|---|--:|--:|--:|
| california_housing | MSE | 0.600 | 0.850 | 1.226 |
| california_housing | GroupDRO | 0.718 | **0.611** | 1.144 |
| california_housing | PRIME | 0.612 | 0.898 | 1.242 |
| california_housing | ConR | 0.620 | 0.873 | 1.211 |
| california_housing | **BMC** | **0.598** | 0.745 | **1.113** |
| california_housing | RankSim | 0.603 | 0.862 | 1.212 |
| california_housing | VIR | 0.602 | 0.799 | 1.152 |
| stress_imbalanced | MSE | 0.596 | 0.616 | 1.033 |
| stress_imbalanced | GroupDRO | 0.671 | **0.592** | 1.014 |
| stress_imbalanced | ConR | 0.615 | 0.578 | 1.028 |
| stress_imbalanced | BMC | 0.598 | 0.615 | 1.044 |
| stress_imbalanced | RankSim | 0.616 | 0.629 | 1.087 |
| stress_imbalanced | VIR | 0.613 | 0.607 | 1.071 |

BMC is the best non-DRO method on California housing worst-region error and tail
RMSE, while GroupDRO remains best on worst-group MAE. RankSim is neutral under
this shared-budget setting. These are single-seed results, not a ranking.

## LSST DESC PZ Data Challenge (GPU)

Run `release_20261019_0003`, harness `022e73b`, torchregress `2a99f4c`.
Official task-set-1 training files, 20% held-out split (the official test
labels are not public), `device=cuda`, `EPOCHS=60`.

**Cardinal 10yr, N=100000 (held-out 20%)**

| method | RMSE | NLL | CRPS | Cov90 | sigma_NMAD | outlier>0.15 | train s |
|---|--:|--:|--:|--:|--:|--:|--:|
| MSE_MLP | 0.0255 | — | — | — | 0.0119 | 0.00075 | 18 |
| GaussianNLL_MLP | 0.0369 | −2.583 | 0.0118 | 0.900 | 0.0107 | 0.00160 | 25 |
| MDN_K3 | 0.0311 | −2.431 | 0.0108 | 0.892 | 0.0101 | 0.00115 | 30 |
| Quantile_MLP | 0.0302 | — | — | 0.908 | 0.0106 | 0.00095 | 24 |
| NormalizingFlow | **0.0199** | **−2.704** | **0.00974** | 0.860 | **0.00960** | **0.00030** | 2735 |
| TabM_MSE | 0.0236 | — | — | — | 0.0106 | 0.00060 | 31 |
| TabM_GaussianNLL | 0.0363 | −2.668 | 0.0109 | 0.912 | 0.00985 | 0.00160 | 33 |
| TabM_SplitCP | 0.0236 | — | — | 0.890 | 0.0106 | 0.00060 | 56 |

**Flagship 10yr, N=100000**

| method | RMSE | NLL | CRPS | Cov90 | sigma_NMAD | outlier>0.15 | train s |
|---|--:|--:|--:|--:|--:|--:|--:|
| MSE_MLP | 0.0271 | — | — | — | 0.0118 | 0.0002 | 18 |
| GaussianNLL_MLP | 0.0345 | −2.430 | 0.0122 | 0.889 | 0.0113 | 0.00065 | 24 |
| MDN_K3 | 0.0322 | −2.566 | 0.0114 | 0.886 | 0.0109 | 0.00045 | 29 |
| Quantile_MLP | 0.0282 | — | — | 0.865 | 0.0114 | 0.0002 | 23 |
| NormalizingFlow | **0.0209** | **−2.635** | **0.0103** | 0.872 | **0.0101** | **0.0002** | 2706 |
| TabM_MSE | 0.0261 | — | — | — | 0.0110 | 0.0002 | 31 |
| TabM_GaussianNLL | 0.0445 | −2.539 | 0.0124 | 0.907 | 0.0111 | 0.00095 | 33 |
| TabM_SplitCP | 0.0261 | — | — | 0.896 | 0.0110 | 0.0002 | 56 |

**1yr simulations (N=100000 each)**

| dataset | method | RMSE | NLL | CRPS | Cov90 | sigma_NMAD | outlier |
|---|---|--:|--:|--:|--:|--:|--:|
| cardinal 1yr | MSE_MLP | 0.0415 | — | — | — | 0.0177 | 0.0031 |
| cardinal 1yr | NormalizingFlow | 0.0399 | **−2.190** | **0.0166** | 0.855 | **0.0147** | 0.0021 |
| cardinal 1yr | TabM_MSE | 0.0394 | — | — | — | 0.0159 | 0.0029 |
| cardinal 1yr | TabM_GaussianNLL | 0.0504 | −2.142 | 0.0193 | 0.924 | 0.0159 | 0.0057 |
| flagship 1yr | MSE_MLP | 0.0408 | — | — | — | 0.0196 | 0.0021 |
| flagship 1yr | NormalizingFlow | 0.0401 | **−2.082** | **0.0187** | 0.867 | 0.0167 | 0.0021 |
| flagship 1yr | TabM_MSE | **0.0400** | — | — | — | 0.0177 | 0.0023 |

**Task sets 2–4 (held-out 20% of official training; N=100000)**

| dataset | method | RMSE | NLL | CRPS | Cov90 | sigma_NMAD | outlier |
|---|---|--:|--:|--:|--:|--:|--:|
| task2 cardinal | NormalizingFlow | **0.0330** | **−2.448** | **0.0133** | 0.872 | **0.0119** | 0.0013 |
| task2 cardinal | TabM_MSE | 0.0381 | — | — | — | 0.0132 | 0.0027 |
| task2 flagship | NormalizingFlow | **0.0287** | **−2.381** | **0.0141** | 0.877 | **0.0122** | 0.0006 |
| task2 flagship | TabM_MSE | 0.0297 | — | — | — | 0.0128 | 0.0006 |
| task3 cardinal | NormalizingFlow | **0.0341** | **−2.655** | **0.0113** | 0.863 | **0.0084** | 0.0017 |
| task3 cardinal | TabM_MSE | 0.0371 | — | — | — | 0.0099 | 0.0037 |
| task3 flagship | NormalizingFlow | **0.0217** | **−2.623** | **0.0106** | 0.844 | **0.0084** | 0.0002 |
| task3 flagship | TabM_MSE | 0.0295 | — | — | — | 0.0096 | 0.0007 |
| task4 flagship | NormalizingFlow | **0.0394** | **−2.575** | **0.0125** | 0.900 | **0.0081** | 0.0022 |
| task4 flagship | TabM_MSE | 0.0440 | — | — | — | 0.0111 | 0.0034 |

Task set 4 cardinal is a documented blend-heavy stress case; the quantile head
crossed its 5/95% quantiles on some rows in the first pass (the suite now sorts
quantiles before forming intervals) and that row is being rerun. The general
pattern holds: NSF is the best p(z)/sigma_NMAD model, TabM the fast point model.

The TabM Gaussian-NLL row on cardinal 1yr initially hit numerical overflow
(`log_variance` → NaN); the suite now clamps the log-variance to [−20, 20], and
the rerun reported the valid row above. No NaN row is reported as a result.

The NSF gives the best point RMSE, NLL, CRPS and sigma_NMAD on both 10yr
simulations, but costs ~45 min on 100k; TabM is ~30 s and nearly as accurate
(RMSE 0.024/0.026), so it is the practical default when training time matters.
TabM Gaussian NLL is the best fast calibrated interval row. These are
strong internal numbers; they are **not** an official leaderboard result
because the challenge test labels are withheld.

## Robust relaunch of failed jobs

The hung fleet jobs were relaunched with a bounded `JOB_TIMEOUT` (45–90 min),
`JOB_RETRIES=1` and a retry delay. The release script now retries each job once
and continues after a timeout instead of blocking its shard.

- **EIV run `release_20261018_1746`** — all six synthetic `eiv` designs, all
  three `eiv_estimators` designs and `flow_eiv` completed. `photoz_eiv`
  completed in the relaunch (`release_20261018_2203`): `NoiseAware` wins
  (RMSE 0.063) over MSE 0.124 and the classic EIV/ODR/SIMEX variants, a useful
  caution that real photometric noise is not a clean linear-EIV problem.
  `galaxiesml_eiv` did not complete (3 GB HDF5 download/training stalled past
  the 90 min budget) and is not claimed. Full recommendations:
  [`EIV_IMBALANCED_WORTHINESS.md`](EIV_IMBALANCED_WORTHINESS.md).
- **Image run `release_20261018_1747`** — 27/31 artifacts. The heavy NYU/KITTI
  and Shanghai crowd jobs now complete or return cleanly; all rows report
  device=cuda.
- **SBI run `release_20261018_1747`** — 4/5; `sbibm_gaussian_linear` timed out
  on both attempts (the same no-convergence case seen earlier), while
  two-moons, SLCP, Gaussian-mixture and Bernoulli-GLM completed.
- **Large-tabular run `release_20261018_1747`** — `folktables_all` and
  `tlc_fare` each timed out twice at 90 min; the session was stopped after
  repeated no-output attempts, and `folktables_eiv` was not reached. No
  big-data rows are claimed. The large-tabular group needs a dedicated
  data-staging/sharding pass before it is scientifically comparable.

The robust timeout/retry behavior is implemented in `scripts/skaha_release.sh`
and passed through by `scripts/canfar_launch_release.sh`.

## Massive mixed sweep (partial snapshot)

Run `release_20261018_1533` launched the full harness fleet on CANFAR headless
GPUs: **160 tabular jobs (4 replicas), 31 image jobs (2 replicas), 3
large-tabular jobs (8c/128G) and 5 SBI jobs**, all with `SEEDS=0`, `EPOCHS=60`.
Artifacts are written per job under
`/arc/projects/ots/torchregress-harness/runs/release_20261018_1533/<group>/`
and are pulled with `PULL_GROUPS=... scripts/canfar_launch_release.sh pull`;
a snapshot tarball is in `.sessions/release_20261018_1533.tgz`.

At the final snapshot (2026-10-18 23:55 UTC) 69 artifacts were complete
(61 JSON suite files),
including all 16 image rows from replica 1 (age regression, cell counting,
censored imaging, depth, crowd counting, keypoint, image conformal/UQ/robust),
60 tabular suite-dataset artifacts (conformal, calibration, ensembles,
stress matrix, freia/CNF parity, robust, sklearn baselines, Tweedie, censored,
Bayesian linear, SBI parity) and the two SBI tasks that had finished. Selected
GPU examples:

| run | metric | value |
|---|---|---|
| image / age_regression / ixi_brainage | CNN+BetaNLL MAE | 15.38 |
| image / age_regression / udacity_steering | CNN+BetaNLL MAE | 0.184 |
| image / cell_counting / bbbc_cellcount | best MAE | see artifact |
| tabular / stress_matrix / stress_ood | DeepEnsemble_K10 NLL | 0.680 |
| tabular / stress_matrix / stress_ood | GaussianNLL_MLP NLL | 6.809 |
| tabular / stress_matrix / stress_imbalanced | MSE_MLP RMSE | 0.473 |
| tabular / conformal / california_housing | 22 rows | see artifact |

The long-running jobs (conformal/robust on the largest tabular sets,
NYU-depth image download/training, folktables large-tabular, and the
Gaussian-linear SBI task) did not complete within the session window and were
stopped after >1 h to free the cluster; artifacts written before that remain
on /arc and in `.sessions/release_20261018_1533.tgz`. Re-run those jobs with
longer timeouts or smaller shards. Do not interpret the partial snapshot as a
completed fleet result.

## Caveats

- Single seed, one GPU slice type; C2ST/calibration metrics are stochastic.
- `sbi`/`sbibm`/`pzflow` versions are the release pins in
  `scripts/skaha_release.sh`; `FrEIA` is an optional harness extra.
- The exact-reference SBI tasks are synthetic by construction; sbibm is the
  faithful benchmark for NRE/NPE, while real-data coverage is provided by the
  image/tabular suites above.
- The DIR datasets used by PRIME/ConR (AgeDB, IMDB-WIKI, NYUD2, STS-B) are
  research-use and multi-GB; they were not staged for this campaign.
