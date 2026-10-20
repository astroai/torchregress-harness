# PZRecipe on the LSST DESC PZ Data Challenge

Full task sets 1–4, **all 16 p(z) datasets**, **3 seeds**, **60 epochs**, CANFAR GPU run `release_20261020_1127` (8 shards).  `PZRecipe` = CE+CDF+RnC-N FT-Transformer stage 1; `PZRecipe_TabM` adds the paper-style TabM stage-2 classifier on frozen features (200 bins).

## Aggregate (16 datasets × 3 seeds)

| method | n | mean RMSE | median NLL | mean CRPS | mean coverage90 | mean σ_NMAD | mean PIT-CvM | mean local-W1 | mean train_s |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| PZRecipe | 48 | 0.0581 | -2.002 | 0.0196 | 0.8762 | 0.01309 | 0.01047 | 0.0855 | 693 |
| PZRecipe_TabM | 48 | 0.0520 | -2.114 | 0.0174 | 0.8702 | 0.01193 | 0.00468 | 0.0571 | 733 |

## Per-dataset means (3 seeds)

| dataset | method | RMSE | CRPS | coverage90 | σ_NMAD |
|---|:--|--:|--:|--:|--:|
| pz_challenge_cardinal_10yr | PZRecipe | 0.0285 | 0.0113 | 0.8676 | 0.00981 |
| pz_challenge_cardinal_10yr | PZRecipe_TabM | 0.0242 | 0.0097 | 0.8785 | 0.00883 |
| pz_challenge_cardinal_1yr | PZRecipe | 0.0414 | 0.0173 | 0.8796 | 0.01507 |
| pz_challenge_cardinal_1yr | PZRecipe_TabM | 0.0388 | 0.0161 | 0.8731 | 0.01429 |
| pz_challenge_flagship_10yr | PZRecipe | 0.0433 | 0.0140 | 0.8758 | 0.01089 |
| pz_challenge_flagship_10yr | PZRecipe_TabM | 0.0323 | 0.0116 | 0.8721 | 0.00980 |
| pz_challenge_flagship_1yr | PZRecipe | 0.0534 | 0.0213 | 0.8708 | 0.01733 |
| pz_challenge_flagship_1yr | PZRecipe_TabM | 0.0451 | 0.0190 | 0.8725 | 0.01614 |
| pz_challenge_taskset2_cardinal_10yr | PZRecipe | 0.0426 | 0.0153 | 0.8659 | 0.01179 |
| pz_challenge_taskset2_cardinal_10yr | PZRecipe_TabM | 0.0363 | 0.0134 | 0.8717 | 0.01093 |
| pz_challenge_taskset2_cardinal_1yr | PZRecipe | 0.0709 | 0.0263 | 0.8724 | 0.01919 |
| pz_challenge_taskset2_cardinal_1yr | PZRecipe_TabM | 0.0677 | 0.0247 | 0.8739 | 0.01795 |
| pz_challenge_taskset2_flagship_10yr | PZRecipe | 0.0466 | 0.0175 | 0.8697 | 0.01302 |
| pz_challenge_taskset2_flagship_10yr | PZRecipe_TabM | 0.0365 | 0.0149 | 0.8713 | 0.01160 |
| pz_challenge_taskset2_flagship_1yr | PZRecipe | 0.0765 | 0.0314 | 0.8667 | 0.02265 |
| pz_challenge_taskset2_flagship_1yr | PZRecipe_TabM | 0.0684 | 0.0286 | 0.8668 | 0.02082 |
| pz_challenge_taskset3_cardinal_10yr | PZRecipe | 0.0348 | 0.0130 | 0.8627 | 0.00851 |
| pz_challenge_taskset3_cardinal_10yr | PZRecipe_TabM | 0.0307 | 0.0108 | 0.8668 | 0.00742 |
| pz_challenge_taskset3_cardinal_1yr | PZRecipe | 0.0797 | 0.0239 | 0.8775 | 0.01298 |
| pz_challenge_taskset3_cardinal_1yr | PZRecipe_TabM | 0.0751 | 0.0220 | 0.8673 | 0.01204 |
| pz_challenge_taskset3_flagship_10yr | PZRecipe | 0.0650 | 0.0148 | 0.8925 | 0.00855 |
| pz_challenge_taskset3_flagship_10yr | PZRecipe_TabM | 0.0570 | 0.0122 | 0.8690 | 0.00756 |
| pz_challenge_taskset3_flagship_1yr | PZRecipe | 0.0599 | 0.0224 | 0.8877 | 0.01356 |
| pz_challenge_taskset3_flagship_1yr | PZRecipe_TabM | 0.0511 | 0.0197 | 0.8692 | 0.01242 |
| pz_challenge_taskset4_cardinal_10yr | PZRecipe | 0.0606 | 0.0162 | 0.8667 | 0.00914 |
| pz_challenge_taskset4_cardinal_10yr | PZRecipe_TabM | 0.0561 | 0.0142 | 0.8653 | 0.00788 |
| pz_challenge_taskset4_cardinal_1yr | PZRecipe | 0.0950 | 0.0278 | 0.8797 | 0.01347 |
| pz_challenge_taskset4_cardinal_1yr | PZRecipe_TabM | 0.0913 | 0.0258 | 0.8688 | 0.01247 |
| pz_challenge_taskset4_flagship_10yr | PZRecipe | 0.0555 | 0.0162 | 0.8863 | 0.00936 |
| pz_challenge_taskset4_flagship_10yr | PZRecipe_TabM | 0.0497 | 0.0135 | 0.8688 | 0.00804 |
| pz_challenge_taskset4_flagship_1yr | PZRecipe | 0.0759 | 0.0240 | 0.8975 | 0.01411 |
| pz_challenge_taskset4_flagship_1yr | PZRecipe_TabM | 0.0724 | 0.0221 | 0.8684 | 0.01275 |

## Comparison with existing PZ core means

| method | mean RMSE | median NLL | mean CRPS | mean coverage90 | mean σ_NMAD | mean train_s |
|---|--:|--:|--:|--:|--:|--:|
| MSE_MLP | 0.0535 | — | — | — | 0.0166 | 13 |
| GaussianNLL_MLP | 0.0663 | -2.1445 | 0.0218 | 0.8997 | 0.0147 | 19 |
| MDN_K3 | 0.0594 | -2.2065 | 0.0199 | 0.8897 | 0.0138 | 22 |
| Quantile_MLP | 0.0548 | — | — | 0.8968 | 0.0145 | 18 |
| NormalizingFlow | 0.0480 | -2.2209 | 0.0174 | 0.8670 | 0.0128 | 2129 |
| TabM_MSE | 0.0500 | — | — | — | 0.0147 | 24 |
| **PZRecipe (60 ep)** | **0.0581** | **-1.833** | **0.0196** | **0.8762** | **0.0131** | **693** |
| **PZRecipe_TabM (60 ep)** | **0.0520** | **-1.926** | **0.0174** | **0.8702** | **0.0119** | **733** |

Paired across the 48 dataset×seed cells, TabM stage 2 improves RMSE in **48/48**, CRPS in **48/48**, and σ_NMAD in **48/48** relative to PZRecipe stage 1.

## Caveats

- The paper TabM stage 2 is now included, but this remains a lean reference: no exhaustive hyperparameter search, no 600-epoch cosine schedule, no early stopping, and one architecture size.
- **Binned NLL is not a continuous-density NLL** and can be positive when the test target falls in a low-mass edge bin.  One task3 flagship seed is an outlier (PZRecipe 1.00; PZRecipe_TabM 5.01); these rows are retained, not hidden.  Compare CRPS/coverage/σ_NMAD for the main ranking.
- The existing PZ core table uses different budgets/seeds; treat the cross-table comparison as indicative rather than paired.
- Coverage90 is ~0.87 for both variants, slightly under the nominal 0.90; this is a real calibration gap, not rounded away.
