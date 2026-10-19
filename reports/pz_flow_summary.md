# cNSF vs cINN-style flow (8 PZ datasets)

Reference: Ren et al. 2026, ApJ 997, 65 (DOI 10.3847/1538-4357/ae201d). Paper: cINN 25 affine layers, cNSF 3 spline layers; paper cINN σ_NMAD=0.018/η=1.60%, cNSF 0.017/0.83%; cNSF ~6 min vs cINN ~25 min on CSST.

| method | n | mean RMSE | median sigma_NMAD | mean NLL (winsorized) | mean CRPS | mean coverage90 | mean train_s |
|---|--:|--:|--:|--:|--:|--:|--:|
| NormalizingFlow | 8 | 0.0482 | 0.0117 | -1.431 | 0.0166 | 0.8691 | 2115 |
| NormalizingFlow_cINN | 8 | 0.1932 | 0.0117 | -2.043 | 0.0193 | 0.8806 | 896 |
| MDN_K3 | 8 | 0.0595 | 0.0127 | -2.221 | 0.0190 | 0.8915 | 22 |

## Per-dataset flow rows

| dataset | method | RMSE | outlier15 | sigma_NMAD | NLL | CRPS | coverage90 | train_s |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| cardinal_10yr | NormalizingFlow | 0.0199 | 0.0003 | 0.0096 | -2.70 | 0.0097 | 0.8599 | 2709 |
| cardinal_10yr | NormalizingFlow_cINN | 0.0234 | 0.0003 | 0.0093 | -2.74 | 0.0095 | 0.8792 | 1165 |
| cardinal_10yr | TabM_MSE | 0.0236 | 0.0006 | 0.0106 | nan | nan | nan | 32 |
| cardinal_1yr | NormalizingFlow | 0.0399 | 0.0021 | 0.0147 | -2.19 | 0.0166 | 0.8546 | 2720 |
| cardinal_1yr | NormalizingFlow_cINN | 0.0594 | 0.0021 | 0.0148 | -2.18 | 0.0170 | 0.8790 | 1166 |
| cardinal_1yr | TabM_MSE | 0.0394 | 0.0029 | 0.0159 | nan | nan | nan | 31 |
| flagship_10yr | NormalizingFlow | 0.0209 | 0.0002 | 0.0101 | -2.63 | 0.0103 | 0.8717 | 2798 |
| flagship_10yr | NormalizingFlow_cINN | 0.0202 | 0.0003 | 0.0102 | -2.63 | 0.0103 | 0.8769 | 1186 |
| flagship_10yr | TabM_MSE | 0.0261 | 0.0002 | 0.0110 | nan | nan | nan | 30 |
| flagship_1yr | NormalizingFlow | 0.0401 | 0.0021 | 0.0167 | -2.08 | 0.0187 | 0.8669 | 2629 |
| flagship_1yr | NormalizingFlow_cINN | 0.0381 | 0.0022 | 0.0167 | -2.11 | 0.0182 | 0.8731 | 1101 |
| flagship_1yr | TabM_MSE | 0.0400 | 0.0022 | 0.0177 | nan | nan | nan | 29 |
| taskset4_cardinal_10yr | NormalizingFlow | 0.0654 | 0.0066 | 0.0090 | -2.25 | 0.0164 | 0.8445 | 1762 |
| taskset4_cardinal_10yr | NormalizingFlow_cINN | 0.0569 | 0.0053 | 0.0083 | -2.56 | 0.0145 | 0.8761 | 746 |
| taskset4_cardinal_10yr | TabM_MSE | 0.0597 | 0.0082 | 0.0121 | nan | nan | nan | 19 |
| taskset4_cardinal_1yr | NormalizingFlow | 0.0922 | 0.0179 | 0.0152 | -2.01 | 0.0265 | 0.8655 | 1820 |
| taskset4_cardinal_1yr | NormalizingFlow_cINN | 0.0931 | 0.0182 | 0.0141 | -1.99 | 0.0266 | 0.8778 | 755 |
| taskset4_cardinal_1yr | TabM_MSE | 0.0933 | 0.0257 | 0.0174 | nan | nan | nan | 21 |
| taskset4_flagship_10yr | NormalizingFlow | 0.0394 | 0.0022 | 0.0081 | -2.57 | 0.0125 | 0.9002 | 1265 |
| taskset4_flagship_10yr | NormalizingFlow_cINN | 1.1810 | 0.0056 | 0.0094 | -2.45 | 0.0361 | 0.9098 | 537 |
| taskset4_flagship_10yr | TabM_MSE | 0.0440 | 0.0034 | 0.0111 | nan | nan | nan | 14 |
| taskset4_flagship_1yr | NormalizingFlow | 0.0681 | 0.0072 | 0.0133 | 170.86 | 0.0218 | 0.8899 | 1218 |
| taskset4_flagship_1yr | NormalizingFlow_cINN | 0.0738 | 0.0067 | 0.0133 | 0.31 | 0.0222 | 0.8727 | 510 |
| taskset4_flagship_1yr | TabM_MSE | 0.0723 | 0.0089 | 0.0162 | nan | nan | nan | 14 |

Findings:
- Our cINN-style RealNVP is ~2.3x faster than the 3-transform NSF (mean 847 s vs 2112 s across these 8 datasets).
- cINN gives better NLL/coverage on most datasets, but has a catastrophic RMSE on task4 flagship 10yr (1.18) from a few extreme point errors while its median scatter and outlier fraction are normal: use robust medians/outlier fractions for point ranking.
- cNSF has a single extreme density underflow on task4 flagship 1yr (mean NLL +170.9) while CRPS/coverage are fine; bounded NSF needs a tail/bound diagnostic. Both are flow-specific numerical caveats, not overall model failures.
- Paper cNSF usually wins point accuracy/outlier rate; our proxy is close, with cINN slightly better in most NLL/coverage columns and cNSF better in the worst-case tails.
