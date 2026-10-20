# PZRecipe on the LSST DESC PZ Data Challenge

Full task-set 1–4 p(z) challenge, all 16 datasets, seed 0, 15 epochs, one CANFAR GPU session sharded 4 ways (`release_20261020_1017`).  PZRecipe is the two-stage FT-Transformer with CE+CDF+RnC-N first-stage objective; no TabM stage 2 and no hyperparameter search.

| metric | mean | median |
|---|--:|--:|
| RMSE | 0.06363 | 0.063051 |
| NLL | -1.8209 | -1.7591 |
| CRPS | 0.023541 | 0.021909 |
| Coverage | 0.91647 | 0.91339 |
| SigmaNMAD | 0.014882 | 0.014635 |
| PITCVM | 0.0077125 | 0.0060873 |
| LocalW1 | 0.085038 | 0.079022 |
| W68 | 0.033793 | 0.032371 |
| train_s | 173.65 | 185.23 |

## Per-dataset

| dataset | RMSE | NLL | CRPS | coverage90 | sigma_NMAD | train_s |
|---|--:|--:|--:|--:|--:|--:|
| pz_challenge_cardinal_10yr | 0.0338 | -2.359 | 0.0130 | 0.906 | 0.01043 | 223 |
| pz_challenge_cardinal_1yr | 0.0444 | -2.008 | 0.0189 | 0.904 | 0.01587 | 224 |
| pz_challenge_flagship_10yr | 0.0470 | -2.206 | 0.0158 | 0.913 | 0.01185 | 223 |
| pz_challenge_flagship_1yr | 0.0570 | -1.788 | 0.0233 | 0.903 | 0.01893 | 224 |
| pz_challenge_taskset2_cardinal_10yr | 0.0514 | -2.163 | 0.0173 | 0.911 | 0.01285 | 223 |
| pz_challenge_taskset2_cardinal_1yr | 0.0785 | -1.685 | 0.0295 | 0.895 | 0.02085 | 224 |
| pz_challenge_taskset2_flagship_10yr | 0.0480 | -1.957 | 0.0196 | 0.901 | 0.01431 | 223 |
| pz_challenge_taskset2_flagship_1yr | 0.0773 | -1.337 | 0.0337 | 0.884 | 0.02518 | 224 |
| pz_challenge_taskset3_cardinal_10yr | 0.0428 | -2.033 | 0.0179 | 0.920 | 0.01008 | 141 |
| pz_challenge_taskset3_cardinal_1yr | 0.0871 | -1.677 | 0.0298 | 0.922 | 0.01496 | 144 |
| pz_challenge_taskset3_flagship_10yr | 0.0676 | -1.730 | 0.0214 | 0.962 | 0.01134 | 102 |
| pz_challenge_taskset3_flagship_1yr | 0.0722 | -1.466 | 0.0290 | 0.929 | 0.01752 | 106 |
| pz_challenge_taskset4_cardinal_10yr | 0.0685 | -1.915 | 0.0216 | 0.913 | 0.01041 | 144 |
| pz_challenge_taskset4_cardinal_1yr | 0.1026 | -1.632 | 0.0333 | 0.918 | 0.01532 | 148 |
| pz_challenge_taskset4_flagship_10yr | 0.0585 | -1.703 | 0.0222 | 0.951 | 0.01102 | 102 |
| pz_challenge_taskset4_flagship_1yr | 0.0814 | -1.476 | 0.0304 | 0.931 | 0.01719 | 105 |

## Context

Existing PZ core means (16 datasets, default budgets) from `reports/pz_core_summary.md`:

| method | mean RMSE | median NLL | mean CRPS | mean coverage90 | mean sigma_NMAD | mean train_s |
|---|--:|--:|--:|--:|--:|--:|
| MSE_MLP | 0.0535 | — | — | — | 0.0166 | 13 |
| GaussianNLL_MLP | 0.0663 | -2.1445 | 0.0218 | 0.8997 | 0.0147 | 19 |
| MDN_K3 | 0.0594 | -2.2065 | 0.0199 | 0.8897 | 0.0138 | 22 |
| Quantile_MLP | 0.0548 | — | — | 0.8968 | 0.0145 | 18 |
| NormalizingFlow | 0.0480 | -2.2209 | 0.0174 | 0.8670 | 0.0128 | 2129 |
| TabM_MSE | 0.0500 | — | — | — | 0.0147 | 24 |
| **PZRecipe (this run)** | **0.0636** | **-1.821** | **0.0235** | **0.9165** | **0.0149** | **174** |

**Reading:** In this unsearched implementation, PZRecipe is cheaper than the NSF flow by ~12× but has worse point accuracy and NLL than MDN/NSF/TabM.  It is best treated as the paper-structure reference, not as a new state-of-the-art result.  The paper’s full recipe uses a TabM stage-2 classifier and a longer schedule; both are absent here.  Coverage is slightly conservative; PIT-CvM/local-W1 are reported for the paper diagnostics.
