# Conditional density and multi-target experiments: canonical summary

This is the exhaustive index of the density-estimation experiments run for the
2026 photo-z / NF work.  It distinguishes **CANFAR GPU production runs** from
**local CPU screening runs**; see the provenance column.  Code locations are
listed at the end.

## 1. Experiment matrix

| # | Experiment | Data | Seeds × epochs | Compute / run | Artifact |
|:-:|:--|:--|:--|:--|:--|
| E1 | Density-head factorial (clean/noisy × imbalance × budget) | synthetic scalar, 1 seed | 1 × 40 | **local CPU screening** | [`density_robustness.md`](density_robustness.md), `.json` |
| E2 | Multi-target density parity: NSF/MAF/cINN/BPF/CFM/MDN | 7 real/synthetic tabular datasets | 3 × 60 | **CANFAR GPU** `release_20261020_1653` | [`modern_density_results.json`](modern_density_results.json) |
| E3 | Known-label-noise densities (clean vs noise-aware NSF/MDN/binned, CFM) | 2 synthetic (scalar, 2-target) | 3 × 60 | **CANFAR GPU** `release_20261020_2032` | [`noise_aware_results.json`](noise_aware_results.json) |
| E3b | CTI-calibrated noise-aware NSF on clean test labels | 2 synthetic | 3 × 60 | **CANFAR GPU** `release_20261022_1224` | [`noise_cti_results.json`](noise_cti_results.json) |
| E4 | Forward inversion + contrastive normalizing flows | synthetic EIV + SBI tasks | 3 × 60 | **CANFAR GPU** `release_20261020_1939` | [`forward_inversion_contrastive_results.json`](forward_inversion_contrastive_results.json) |
| E5 | NFM (coupling) on known-noise targets | 2 synthetic | 3 × 60 | **CANFAR GPU** `release_20261021_1003` | [`nfm_noise_results.json`](nfm_noise_results.json) |
| E5b | NFM on real multi-target data, cached OpenML/UCI | 5 real datasets | 3 × 60 | **CANFAR GPU** `release_20261022_0749` | [`nfm_real_results.json`](nfm_real_results.json) |
| E5c | Joint multi-target binned CDF on known-noise synthetic data | synthetic 2-target | 3 × 60 | **CANFAR GPU** `release_20261022_0853` | [`binned_joint_results.json`](binned_joint_results.json) |
| E5d | Ambitious tokenized TARFlow on real multi-target data | mt_jura (3), mt_rf1 (8), mt_scm20d (16) | 3 seeds | **CANFAR GPU** `release_20261022_1205` (running) | artifacts under `/arc/.../runs/release_20261022_1205/tabular` |
| E6 | NFM TARFlow-style architecture comparison | 2 synthetic | 3 × 4×60 | **CANFAR GPU** `release_20261021_1433` | [`nfm_tarflow_results.json`](nfm_tarflow_results.json) |
| E7 | Conformalized ContrastiveFlow / CTI / Mondrian / WALDO | synthetic scalar + 2-target | 3 × 80 | **CANFAR GPU** `release_20261022_0750` | [`conformal_density_canfar_results.json`](conformal_density_canfar_results.json); local screen [`conformal_density_bench.json`](conformal_density_bench.json) |
| E7b | WALDO + local critical-value regressor | synthetic scalar + 2-target | 3 × 80 | **CANFAR GPU** `release_20261022_0852` | [`conformal_qr_results.json`](conformal_qr_results.json) |
| E8 | PZRecipe stage 1 on the LSST DESC PZ challenge | 16 PZ datasets | 1 × 15 | **CANFAR GPU** `release_20261020_1005` | [`pz_recipe_all_results.json`](pz_recipe_all_results.json) |
| E9 | PZRecipe + TabM stage 2 | 16 PZ datasets | 3 × 60 | **CANFAR GPU** `release_20261020_1127` | [`pz_recipe_summary.md`](pz_recipe_summary.md), `pz_recipe_all_results_long.json` |
| E9b | **Matched-budget** PZRecipe vs NormalizingFlow | 16 PZ datasets | 1 × 60 | **CANFAR GPU** `release_20261022_0940` | [`pz_matched_results.json`](pz_matched_results.json) |
| E10 | Existing PZ flow comparison (cNSF vs cINN) | 8 PZ flow datasets | existing run | historical harness | [`pz_flow_summary.md`](pz_flow_summary.md) |
| E11 | Existing PZ core heads (16 datasets) | 16 PZ datasets | existing run | historical harness | [`pz_core_summary.md`](pz_core_summary.md) |

“Screening” means one seed, small synthetic data and/or local CPU; it is useful
for code/statistics checks but not a final ranking.  All production-style tables
use CANFAR GPU artifacts and three seeds.

### E1 screening takeaways

The full 48-row factorial table is in [`density_robustness.md`](density_robustness.md).
Key patterns: at budget 1000 and zero noise, MDN has the best NLL
(1.002 vs NSF 1.123); at noise 0.3, `MDN_noise` reaches 1.043 while
`NSF_noise` can degrade (1.277).  Binned heads are the most stable under severe
imbalance but their NLL is a probability mass, not a continuous likelihood.
This is one seed / 40 epochs and is a **screening experiment**, not a ranking.

## 2. Multi-target conditional densities (E2)

Median over 7 datasets × 3 seeds:

| Method | NLL | CRPS | Energy | Cov90 | PIT KS | train_s |
|:--|--:|--:|--:|--:|--:|--:|
| NSF | **−1.951** | 0.1487 | 0.2241 | 0.890 | 0.091 | 90 |
| MAF | −1.393 | 0.1233 | 0.2388 | 0.889 | 0.095 | 23 |
| cINN | −0.902 | 0.1380 | 0.2388 | 0.893 | 0.080 | 25 |
| BPF | −1.419 | **0.0902** | **0.1586** | 0.981 | 0.162 | 81 |
| CFM | — | 0.1502 | 0.2395 | 0.907 | 0.121 | **11** |
| MDN | −0.614 | 0.1609 | 0.2410 | 0.898 | 0.101 | 25 |

**What holds:** BPF is the best sample/coverage head but not the NLL head; NSF is
the likelihood default; CFM is much cheaper but has no exact likelihood;
cINN/MAF have rare extreme-transform failures on `mt_rf1`/`mt_scm20d` (means
unusable — use medians/robust aggregation).

## 3. Known-label-noise densities (E3)

Clean-test evaluation, mean over 2 datasets × 3 seeds:

| Method | RMSE | NLL | CRPS | Cov90 |
|:--|--:|--:|--:|--:|
| NSF | 0.7044 | 1.125 | 0.4041 | 0.934 |
| NSF_nn | 0.7052 | **1.100** | 0.4039 | 0.888 |
| MDN | 0.7025 | 1.102 | 0.4024 | 0.932 |
| MDN_nn | 0.7022 | **1.063** | **0.4013** | 0.888 |
| Binned | 0.6999 | —* | 0.4042 | 0.950 |
| Binned_nn | **0.6980** | —* | 0.4036 | 0.964 |
| CFM | 0.7068 | — | 0.4101 | 0.923 |

`_nn` = noise-aware.  `*` Binned NLL is a probability mass, not a continuous
density.  The analytic MDN convolution gives the largest NLL gain; both
noise-aware variants trade about 4–5 coverage points for likelihood fit.

### E3b: CTI-calibrated noise-aware density

`NSF_noise_CTI` trains the same Gaussian-Hermite noise-aware NSF on an 80% fit
split and calibrates a split-conformal CTI threshold on the held-out 20%.
CANFAR `release_20261022_1224`, 2 synthetic datasets × 3 seeds:

| Dataset | Method | NLL | Raw Cov90 | CTI Cov90 | CTI width | Energy | Variogram |
|:--|:--|--:|--:|--:|--:|--:|--:|
| scalar | NSF_noise | 0.854 | 0.885 | — | — | 0.399 | — |
| scalar | NSF_noise_CTI | 0.873 | 0.884 | **0.952** | 2.276 | 0.401 | — |
| multi | NSF_noise | 1.346 | 0.891 | — | — | 0.608 | 0.057 |
| multi | NSF_noise_CTI | 1.389 | 0.893 | **0.963** | — | 0.611 | 0.059 |

The calibrated variant fixes the under-coverage direction but is now
**over-conservative** (0.95–0.96 vs nominal 0.90), and the split reduces training
data, so NLL/CRPS are slightly worse.  This is a proof-of-concept, not a tuned
final answer: the next step is to choose the calibration split/level and compare
interval width and conditional coverage by target region.  Energy/variogram
metrics are now reported in the noise-aware suite as well.

## 4. EIV and contrastive SBI (E4)

| Method | RMSE / W1 | Coverage | NLL / RefMeanRMSE |
|:--|--:|--:|--:|
| ForwardInversion | 0.6494 RMSE | 0.913 | 0.941 NLL |
| FlowEB | 0.5437 RMSE | 0.873 | 0.775 NLL |
| OLS | 0.5400 RMSE | 0.897 | 0.803 NLL |
| ContrastiveFlow | **0.0784 W1** | 0.972 Cov95 | **0.0638 RefMeanRMSE** |
| NRE multiclass | 0.0833 W1 | 1.000 Cov95 | 0.0720 RefMeanRMSE |
| NPE | 0.0876 W1 | 0.917 Cov95 | 0.0745 RefMeanRMSE |

Contrastive flow is the strongest posterior engine in this synthetic SBI
comparison, but it is a ranking/likelihood-ratio method, not a calibrated
regression density.  ForwardInversion is an EIV tool, not a general density win.

## 5. Normalizing Flow Map (E5–E6)

### E5b: real multi-target data with a cached dataset

The earlier failed real-data run was a cache/network problem, not a model
result.  I staged the OpenML/UCI files on `/arc` and reran NFM on **5 real
datasets × 3 seeds, 60 epochs** (`release_20261022_0749`):

| Dataset | NLL | CRPS | Cov90 | NFM RMSE-equivalent |
|:--|--:|--:|--:|:--|
| california_housing | 0.656 | 0.254 | 0.915 | — |
| mt_enb | 0.618 | 0.172 | 0.986 | — |
| mt_jura | 2.537 | 0.319 | 0.927 | — |
| mt_rf1 | 1.745 | 0.131 | 0.999 | — |
| mt_scm20d | 5.936 | 0.217 | 0.977 | — |

NFM is still clearly worse than the matched NSF/BPF rows (E2) on NLL/CRPS, so
this is **not** evidence for replacing NSF/MDN.  It is now a real-data result
rather than a synthetic-only claim.  The cached data path is
`/arc/projects/ots/torchregress-harness/datasets/cache/tr_cache`; no laptop
training was used.

### E5/E6 synthetic architecture comparison

| NFM variant | RMSE | CRPS | NLL | Notes |
|:--|--:|--:|--:|:--|
| coupling (3×64, 60 ep) | 0.8818 | 0.5064 | 1.300 | earlier noise-aware run |
| TARFlow (3×128, 4×60 ep) | 0.9356 | 0.5533 | 1.326 | `release_20261021_1433`; not better |

TARFlow is implemented (`arch="tarflow"`) and promising at larger budgets, but
the current tabular harness does not justify replacing NSF/MDN.

## 6. Conformal (E7, CANFAR 3 seeds)

Canonical CANFAR run `release_20261022_0750`; 2 synthetic datasets × 3 seeds,
80 epochs.  Coverage is on a clean test split; nominal **0.90**.

| Dataset | Model | NLL | Raw cov | CTI cov | Mondrian cov | WALDO cov |
|:--|:--|--:|--:|--:|--:|--:|
| scalar | NSF | 1.183 | 0.885 | 0.909 | 0.910 | 0.896 |
| scalar | ContrastiveFlow | 1.168 | 0.901 | 0.906 | 0.908 | 0.907 |
| multi | NSF | 1.813 | 0.881 | 0.892 | 0.894 | 0.895 |
| multi | ContrastiveFlow | 1.804 | 0.874 | 0.893 | 0.895 | 0.891 |

ContrastiveFlowConformal works: canonical multi-target CTI coverage is 0.893
and scalar 0.906, comparable to NSF.  Mondrian target-tercile groups consistently
nudge coverage toward nominal; WALDO density regions are a comparable
alternative.  The earlier one-seed/local screen (0.863 multi NSF) is superseded
by this CANFAR run.  Full 48-row artifact:
[`conformal_density_canfar_results.json`](conformal_density_canfar_results.json).

### E5d: ambitious tokenized TARFlow

The earlier TARFlow comparison used the lean per-layer causal transformer.  The
ambitious run uses the new `TokenizedTARFlowLayer`: explicit context/time tokens,
one token per target, a causal **3-layer** Transformer per flow layer,
`128` hidden units, 2 flow layers, and NLL weight 0.5.  It runs on the 3-, 8- and
16-target real datasets (`mt_jura`, `mt_rf1`, `mt_scm20d`), 3 seeds, on CANFAR
GPU `release_20261022_1205`.

Initial partial artifact (mt_jura seed 0): NLL 2.85, CRPS 0.381 — worse than the
matched NSF/BPF rows, so this remains an architecture experiment, not a default
recommendation.  Full artifacts and comparison will be added when the 3-seed job
completes.

### E5c: joint multi-target binned CDF

`JointBinnedDensityLoss` is a true joint histogram for low-dimensional targets
(here 2 targets, 25×25 bins), with Gaussian-smoothed soft targets.  CANFAR
`release_20261022_0853`, multi-target synthetic set, 3 seeds:

| Method | RMSE | CRPS | Cov90 | NLL (mass) |
|:--|--:|--:|--:|--:|
| Binned_joint | 0.7121 | 0.4108 | 0.958 | 1.640 |
| Binned_joint_noise | 0.7121 | 0.4119 | 0.971 | 1.654 |

For reference, the matched NSF row in E3 is RMSE 0.7044 / CRPS 0.4041.  The
joint binned head is competitive on point/CRPS but **under-covers less / over-covers**
relative to nominal 90%, and its NLL is a probability mass, not a continuous
density.  Scalar datasets are skipped by design.  This closes the structural
gap: the noise-robust binned branch now has a joint multi-target head.

### E7b: WALDO critical-value regression

`WaldoCriticalValueRegressor` was added and evaluated on the same CANFAR
conformal suite (`release_20261022_0852`).  Mean coverage over 3 seeds:

| Dataset | Method | Default WALDO | WALDO + local QR |
|:--|:--|--:|--:|
| scalar | NSF | 0.896 | 0.895 |
| scalar | ContrastiveFlow | 0.907 | 0.897 |
| multi | NSF | 0.895 | 0.888 |
| multi | ContrastiveFlow | 0.891 | 0.879 |

The local quantile regressor did **not** improve coverage on this synthetic
problem; it slightly under-covered and did not shrink widths consistently.
This is a negative but useful result: keep the simpler split-conformal/global
threshold for small synthetic calibration sets, and test QR again only with a
larger, genuinely heteroscedastic calibration set.  The API and tests remain
available for that experiment.

## 7. PZRecipe and PZ conditional distributions (E8–E11)

PZRecipe is a **binned conditional CDF** model, not a flow, but belongs in the
conditional-density comparison.  Full table: [`pz_recipe_summary.md`](pz_recipe_summary.md).

**Matched-budget run** (E9b, `release_20261022_0940`): 16 datasets, 1 seed,
60 epochs, same split/budget, `PZRecipe` vs `NormalizingFlow` (cNSF):

| Method | RMSE | median NLL | CRPS | Cov90 | σ_NMAD | train_s |
|:--|--:|--:|--:|--:|--:|--:|
| PZRecipe | 0.0578 | −2.0212 | 0.0195 | **0.8794** | 0.0131 | **694** |
| NormalizingFlow | **0.0480** | **−2.2209** | **0.0174** | 0.8670 | **0.0128** | 2107 |

The matched run confirms the historical picture: **NSF wins point accuracy,
median NLL and CRPS; PZRecipe is ~3× faster and has slightly better coverage**
(0.879 vs 0.867, still below nominal 0.90).  This is a 1-seed comparison; the
3-seed PZRecipe/TabM table below is the separately-run E9 result.

Three-seed PZRecipe/TabM (E9):

| Method | RMSE | median NLL | CRPS | Cov90 | σ_NMAD | train_s |
|:--|--:|--:|--:|--:|--:|--:|
| PZRecipe | 0.0581 | −2.002 | 0.0196 | 0.8762 | 0.0131 | 693 |
| PZRecipe_TabM | **0.0520** | **−2.114** | **0.0174** | 0.8702 | **0.0119** | 733 |

TabM stage 2 improves PZRecipe in **48/48** dataset×seed cells on RMSE, CRPS and
σ_NMAD.  It is still not a replacement for the mature NSF on point accuracy.
Historical PZ flow/core summaries are E10/E11.

### E10 historical PZ flow comparison (8 datasets)

From [`pz_flow_summary.md`](pz_flow_summary.md), same features/budget:

| Method | RMSE | NLL (winsorized) | CRPS | Cov90 | train_s |
|:--|--:|--:|--:|--:|--:|
| NormalizingFlow (cNSF) | 0.0482 | −1.431 | 0.0166 | 0.8691 | 2115 |
| NormalizingFlow_cINN | 0.1932 | −2.043 | 0.0193 | 0.8806 | 896 |
| MDN_K3 | 0.0595 | −2.221 | 0.0190 | 0.8915 | 22 |

The 16-dataset PZ core run (E11) is the source of the NormalizingFlow/MDN
numbers used in the PZRecipe comparison above; the 8-dataset table is retained
because it isolates the cNSF vs cINN flow-family comparison.

## 8. What the experiments support

1. **No single density head wins every metric.** NSF for likelihood; BPF for
   CRPS/coverage; CFM for cheap sampling; MDN/binned for low-dimensional,
   small-data, noise-aware cases.
2. **Noise-aware likelihood is a bias/variance trade.** It improves clean NLL
   but under-covers; calibrate after training.
3. **Binned likelihood is not continuous likelihood.** Never compare its NLL
   directly with flow/MDN NLL.
4. **cINN/MAF can fail catastrophically on hard multi-target data.** Report
   medians and tail diagnostics, not just means.
5. **Conformal fixes marginal coverage cheaply.** CTI and Mondrian are practical;
   WALDO is an alternative Wald region; conditional guarantee still needs a
   simulator or a validated conditional protocol.
6. **NF recipes need calibration and enough budget.** PZRecipe/TabM gains need
   long schedules; NFM/TARFlow need more work before default use.
7. **PZRecipe is a useful binned reference, not an automatic SOTA win.**

## 9. Code map

**torchregress** (sibling repository, `../../torchregress/`)
- Flows: [`losses/nflows.py`](../../torchregress/src/torchregress/losses/nflows.py) (`NSF`, `MAF`, `RealNVP/cINN`, `BPF`, `NormalizingFlowLoss`, `ContrastiveFlowLoss`)
- Noise-aware / binned: [`losses/noise_aware_density.py`](../../torchregress/src/torchregress/losses/noise_aware_density.py)
- PZRecipe: [`losses/recipe.py`](../../torchregress/src/torchregress/losses/recipe.py), [`models/ft_transformer.py`](../../torchregress/src/torchregress/models/ft_transformer.py), [`models/pz_recipe.py`](../../torchregress/src/torchregress/models/pz_recipe.py)
- CFM: [`models/flow_matching.py`](../../torchregress/src/torchregress/models/flow_matching.py)
- NFM: [`models/normalizing_flow_map.py`](../../torchregress/src/torchregress/models/normalizing_flow_map.py)
- Conformal: [`losses/conformal.py`](../../torchregress/src/torchregress/losses/conformal.py) (`CTI`, `ConformalizedDensity`, `ContrastiveFlowConformal`)
- WALDO: [`losses/waldo.py`](../../torchregress/src/torchregress/losses/waldo.py)
- Metrics: [`metrics/photo_z.py`](../../torchregress/src/torchregress/metrics/photo_z.py), [`metrics/representation.py`](../../torchregress/src/torchregress/metrics/representation.py)
- Guides: [`docs/methods/flow_matching.md`](../../torchregress/docs/methods/flow_matching.md), [`docs/methods/normalizing_flow_maps.md`](../../torchregress/docs/methods/normalizing_flow_maps.md), [`docs/methods/waldo.md`](../../torchregress/docs/methods/waldo.md), [`docs/losses/conformal.md`](../../torchregress/docs/losses/conformal.md), [`docs/losses/photo_z_recipe.md`](../../torchregress/docs/losses/photo_z_recipe.md), [`docs/guide/density_estimation_roadmap.md`](../../torchregress/docs/guide/density_estimation_roadmap.md)

**torchregress-harness** (this repository)
- Suites: [`suites/tabular/density_parity.py`](../suites/tabular/density_parity.py), [`noise_aware_density.py`](../suites/tabular/noise_aware_density.py), [`pz_challenge.py`](../suites/tabular/pz_challenge.py), [`flow_eiv.py`](../suites/tabular/flow_eiv.py), [`likelihood_ratio.py`](../suites/tabular/likelihood_ratio.py)
- Benchmarks: [`tools/density_robustness_bench.py`](../tools/density_robustness_bench.py), [`tools/conformal_density_bench.py`](../tools/conformal_density_bench.py)
- Reports: this file, [`modern_density_summary.md`](modern_density_summary.md), [`pz_recipe_summary.md`](pz_recipe_summary.md), [`density_robustness.md`](density_robustness.md), and the JSON artifacts above.

## 10. Recommendations

- **Default scalar regression:** start with MDN or binned CDF; use NSF if NLL and
  multimodal tails matter.  Do not start with a flow by default on tiny 1-D data.
- **Multi-target:** NSF for NLL; BPF for CRPS/coverage; CFM only when sampling
  cost dominates and no NLL is required.
- **Known label noise:** try analytic MDN convolution first; noise-aware NSF
  second; always re-check coverage.
- **Calibration:** always calibrate on a held-out split; report coverage/width
  and PIT, not just NLL; use Mondrian groups when conditional coverage matters.
- **Imbalanced targets:** reweighting changes the population optimum; report the
  reweighted objective separately (see `docs/losses/imbalanced.md`).
- **PZ:** PZRecipe + TabM is a strong binned reference; compare against NSF with
  matched seeds/budgets before claiming a win.
- **Reproducibility:** shared encoder, fixed budget, multiple seeds, clean test
  set, robust medians for unstable flows, and explicit compute provenance.

## 11. Roadmap

See the torchregress roadmap in
[`docs/guide/density_estimation_roadmap.md`](../../torchregress/docs/guide/density_estimation_roadmap.md).
