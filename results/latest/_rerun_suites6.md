# Comprehensive Regression Leaderboard

- Target coverage: **0.90**
- Methods ranked: **152**
- Primary ranking metric: **IntervalScore** (lower is better)

## Leaderboard (suite-mean by method)

| Rank | Method | Library | N Suites | N Runs | RMSE | IntervalScore | Coverage | Coverage Dev | ECE |
|---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 1 | NF_Ensemble_K20 | torchregress | 1 | 4 | - | 1.8764 | 0.8639 | 0.0438 | - |
| 2 | NF_Ensemble_K10 | torchregress | 1 | 4 | - | 1.9029 | 0.8534 | 0.0523 | - |
| 3 | NF_Ensemble_K5 | torchregress | 1 | 4 | - | 1.9139 | 0.8381 | 0.0619 | - |
| 4 | MDN_K3 | torchregress | 1 | 4 | - | 1.9948 | 0.8296 | 0.0704 | - |
| 5 | GaussianNLL | torchregress | 1 | 4 | - | 2.0227 | 0.8395 | 0.0685 | - |
| 6 | NF_Ensemble_K2 | torchregress | 1 | 4 | - | 2.0659 | 0.8133 | 0.0867 | - |
| 7 | torchregress_NSF | torchregress | 1 | 4 | - | 2.1221 | 0.7720 | 0.1280 | - |
| 8 | Split_CP | torchregress | 1 | 4 | - | 2.2221 | 0.9018 | 0.0068 | - |
| 9 | Evidential | torchregress | 1 | 4 | - | 2.3535 | 0.9816 | 0.0816 | - |
| 10 | NormalizingFlow | torchregress | 1 | 4 | - | 2.6207 | 0.7442 | 0.1558 | - |
| 11 | pzflow_NSF | pzflow | 1 | 4 | - | 3.7699 | 0.9002 | 0.0072 | - |
| 12 | CQR+MLP | torchregress | 2 | 8 | 32571.7537 | 108687.3212 | 0.9052 | 0.0107 | - |
| 13 | MAPIE_CrossCQR+GBR | MAPIE | 1 | 4 | - | 121170.4403 | 0.9131 | 0.0131 | - |
| 14 | MAPIE_CQR+GBR | MAPIE | 1 | 4 | - | 124464.4064 | 0.9059 | 0.0129 | - |
| 15 | crepes_Split+TabICL | tabicl | 1 | 4 | - | 131468.1127 | 0.9138 | 0.0139 | - |
| 16 | Split+TabICL | tabicl | 1 | 4 | - | 131468.1152 | 0.9138 | 0.0139 | - |
| 17 | MAPIE_Split+TabICL | tabicl | 1 | 4 | - | 131468.1152 | 0.9138 | 0.0139 | - |
| 18 | Split+MLP | torchregress | 2 | 8 | 33106.2160 | 157411.9957 | 0.9100 | 0.0144 | - |
| 19 | SplitCP+MSE_WarmupMC | torchregress+CP | 1 | 4 | 33327.0502 | 157470.5971 | 0.8995 | 0.0059 | 0.0073 |
| 20 | SplitCP+FunctionalEIV | torchregress+CP | 1 | 4 | 33765.8958 | 160275.6235 | 0.9075 | 0.0075 | 0.0065 |
| 21 | R2C+Softmax | torchregress | 1 | 4 | 37794.0087 | 160889.4964 | 0.9753 | 0.0753 | - |
| 22 | SplitCP+InputNoiseMC | torchregress+CP | 1 | 4 | 35022.0473 | 163893.9303 | 0.8987 | 0.0090 | 0.0064 |
| 23 | MAPIE_Split+MLP | MAPIE | 1 | 4 | - | 164231.2231 | 0.8999 | 0.0125 | - |
| 24 | R2C+CORAL | torchregress | 1 | 4 | 40325.3741 | 171391.1832 | 0.9416 | 0.0416 | - |
| 25 | R2C+CumulativeLink | torchregress | 1 | 4 | 38497.3820 | 174046.1155 | 0.9754 | 0.0754 | - |
| 26 | Split+Linear(post-hoc) | torchregress | 1 | 4 | - | 232385.3472 | 0.9012 | 0.0070 | - |
| 27 | torchcp_Split+Linear | torchcp | 1 | 4 | - | 232385.7958 | 0.9012 | 0.0070 | - |
| 28 | crepes_Split+Linear | crepes | 1 | 4 | - | 232385.7961 | 0.9012 | 0.0070 | - |
| 29 | MAPIE_Split+Linear | MAPIE | 1 | 4 | - | 232385.7982 | 0.9012 | 0.0070 | - |
| 30 | Split+Linear | torchregress | 1 | 4 | - | 232650.0997 | 0.8929 | 0.0188 | - |
| 31 | torchcp_R2C | torchcp | 1 | 4 | 577300.1755 | 1174297.5256 | 0.6005 | 0.3495 | - |
| 32 | RC+FunctionalEIV | torchregress | 1 | 4 | 31708.6532 | - | - | - | - |
| 33 | FunctionalEIV | torchregress | 1 | 4 | 32019.5207 | - | - | - | - |
| 34 | StructuralEIV | torchregress | 1 | 4 | 32019.5207 | - | - | - | - |
| 35 | MSE_NoisyPredict | torchregress | 1 | 4 | 32177.7693 | - | - | - | - |
| 36 | MSE | torchregress | 1 | 4 | 32284.2794 | - | - | - | - |
| 37 | FunctionalEIV_MC | torchregress | 1 | 4 | 32676.3639 | - | - | - | - |
| 38 | Uncalibrated | torchregress | 1 | 4 | 32736.8225 | - | - | - | 0.0035 |
| 39 | VarianceTemperatureScaler | torchregress | 1 | 4 | 32736.8225 | - | - | - | 0.0037 |
| 40 | RC+GaussianNLL | torchregress | 1 | 4 | 32839.2869 | - | - | - | 0.0042 |
| 41 | RC | torchregress | 1 | 4 | 32874.5174 | - | - | - | - |
| 42 | GaussianNLL+InputNoiseMC | torchregress | 1 | 4 | 32997.1016 | - | - | - | 0.0036 |
| 43 | IsotonicMeanCalibrator | torchregress | 1 | 4 | 33287.6780 | - | - | - | 0.0036 |
| 44 | InputNoiseMC | torchregress | 1 | 4 | 33291.2336 | - | - | - | - |
| 45 | sklearn_IsotonicRegression | sklearn | 1 | 4 | 33789.8649 | - | - | - | 0.0038 |
| 46 | EnsembleEIV | torchregress | 1 | 4 | 34299.4103 | - | - | - | - |
| 47 | MSE_WarmupMC | torchregress | 1 | 4 | 34859.1038 | - | - | - | - |
| 48 | LatentNN | torchregress | 1 | 4 | 46170.5682 | - | - | - | - |
| 49 | SIMEX | torchregress | 1 | 4 | 57315.0920 | - | - | - | - |
| 50 | ODR | torchregress | 1 | 4 | 243431.6826 | - | - | - | - |
| 51 | TLS | numpy | 1 | 4 | 801963.3931 | - | - | - | - |
| 52 | TLS_Init+FunctionalEIV | torchregress | 1 | 4 | 1314494.8634 | - | - | - | - |
| 53 | NF_Ensemble_K20/bin_5 | torchregress | 1 | 3 | - | - | 0.8816 | 0.0305 | - |
| 54 | Split_CP/bin_5 | torchregress | 1 | 3 | - | - | 0.8793 | 0.0321 | - |
| 55 | NF_Ensemble_K10/bin_5 | torchregress | 1 | 3 | - | - | 0.8718 | 0.0391 | - |
| 56 | MDN_K3/bin_5 | torchregress | 1 | 3 | - | - | 0.8581 | 0.0419 | - |
| 57 | NF_Ensemble_K5/bin_5 | torchregress | 1 | 3 | - | - | 0.8624 | 0.0450 | - |
| 58 | GaussianNLL/bin_5 | torchregress | 1 | 3 | - | - | 0.8629 | 0.0492 | - |
| 59 | GaussianNLL/bin_0 | torchregress | 1 | 4 | - | - | 0.8465 | 0.0573 | - |
| 60 | Evidential/bin_9 | torchregress | 1 | 3 | - | - | 0.9585 | 0.0585 | - |
| 61 | NF_Ensemble_K2/bin_5 | torchregress | 1 | 3 | - | - | 0.8328 | 0.0672 | - |
| 62 | MDN_K3/bin_9 | torchregress | 1 | 3 | - | - | 0.8234 | 0.0766 | - |
| 63 | Evidential/bin_5 | torchregress | 1 | 3 | - | - | 0.9822 | 0.0822 | - |
| 64 | MDN_K3/bin_0 | torchregress | 1 | 4 | - | - | 0.8503 | 0.0824 | - |
| 65 | GaussianNLL/bin_9 | torchregress | 1 | 3 | - | - | 0.8287 | 0.0849 | - |
| 66 | Evidential/bin_0 | torchregress | 1 | 4 | - | - | 0.9370 | 0.0870 | - |
| 67 | NormalizingFlow/bin_5 | torchregress | 1 | 3 | - | - | 0.7636 | 0.1364 | - |
| 68 | NF_Ensemble_K10/bin_9 | torchregress | 1 | 3 | - | - | 0.7353 | 0.1770 | - |
| 69 | NF_Ensemble_K20/bin_9 | torchregress | 1 | 3 | - | - | 0.7379 | 0.1795 | - |
| 70 | NF_Ensemble_K5/bin_9 | torchregress | 1 | 3 | - | - | 0.7163 | 0.1837 | - |
| 71 | NF_Ensemble_K2/bin_9 | torchregress | 1 | 3 | - | - | 0.7066 | 0.1934 | - |
| 72 | Split_CP/bin_0 | torchregress | 1 | 4 | - | - | 0.7003 | 0.2046 | - |
| 73 | NormalizingFlow/bin_9 | torchregress | 1 | 3 | - | - | 0.6737 | 0.2263 | - |
| 74 | NF_Ensemble_K20/bin_0 | torchregress | 1 | 4 | - | - | 0.6583 | 0.2572 | - |
| 75 | NF_Ensemble_K10/bin_0 | torchregress | 1 | 4 | - | - | 0.6645 | 0.2683 | - |
| 76 | NF_Ensemble_K5/bin_0 | torchregress | 1 | 4 | - | - | 0.6466 | 0.2689 | - |
| 77 | NF_Ensemble_K2/bin_0 | torchregress | 1 | 4 | - | - | 0.6350 | 0.2805 | - |
| 78 | NormalizingFlow/bin_0 | torchregress | 1 | 4 | - | - | 0.6057 | 0.2943 | - |
| 79 | Split_CP/bin_9 | torchregress | 1 | 3 | - | - | 0.6074 | 0.2959 | - |
| 80 | normflows_NSF | normflows | 1 | 4 | - | - | - | - | - |
| 81 | Split_CP/dev_summary | torchregress | 1 | 4 | - | - | - | - | - |
| 82 | GaussianNLL/dev_summary | torchregress | 1 | 4 | - | - | - | - | - |
| 83 | NormalizingFlow/dev_summary | torchregress | 1 | 4 | - | - | - | - | - |
| 84 | NF_Ensemble_K2/dev_summary | torchregress | 1 | 4 | - | - | - | - | - |
| 85 | NF_Ensemble_K5/dev_summary | torchregress | 1 | 4 | - | - | - | - | - |
| 86 | NF_Ensemble_K10/dev_summary | torchregress | 1 | 4 | - | - | - | - | - |
| 87 | NF_Ensemble_K20/dev_summary | torchregress | 1 | 4 | - | - | - | - | - |
| 88 | MDN_K3/dev_summary | torchregress | 1 | 4 | - | - | - | - | - |
| 89 | Evidential/dev_summary | torchregress | 1 | 4 | - | - | - | - | - |
| 90 | Split_CP/bin_1 | torchregress | 1 | 3 | - | - | - | - | - |
| 91 | Split_CP/bin_2 | torchregress | 1 | 3 | - | - | - | - | - |
| 92 | Split_CP/bin_3 | torchregress | 1 | 3 | - | - | - | - | - |
| 93 | Split_CP/bin_4 | torchregress | 1 | 3 | - | - | - | - | - |
| 94 | Split_CP/bin_6 | torchregress | 1 | 3 | - | - | - | - | - |
| 95 | Split_CP/bin_7 | torchregress | 1 | 3 | - | - | - | - | - |
| 96 | Split_CP/bin_8 | torchregress | 1 | 3 | - | - | - | - | - |
| 97 | GaussianNLL/bin_1 | torchregress | 1 | 3 | - | - | - | - | - |
| 98 | GaussianNLL/bin_2 | torchregress | 1 | 3 | - | - | - | - | - |
| 99 | GaussianNLL/bin_3 | torchregress | 1 | 3 | - | - | - | - | - |
| 100 | GaussianNLL/bin_4 | torchregress | 1 | 3 | - | - | - | - | - |
| 101 | GaussianNLL/bin_6 | torchregress | 1 | 3 | - | - | - | - | - |
| 102 | GaussianNLL/bin_7 | torchregress | 1 | 3 | - | - | - | - | - |
| 103 | GaussianNLL/bin_8 | torchregress | 1 | 3 | - | - | - | - | - |
| 104 | NormalizingFlow/bin_1 | torchregress | 1 | 3 | - | - | - | - | - |
| 105 | NormalizingFlow/bin_2 | torchregress | 1 | 3 | - | - | - | - | - |
| 106 | NormalizingFlow/bin_3 | torchregress | 1 | 3 | - | - | - | - | - |
| 107 | NormalizingFlow/bin_4 | torchregress | 1 | 3 | - | - | - | - | - |
| 108 | NormalizingFlow/bin_6 | torchregress | 1 | 3 | - | - | - | - | - |
| 109 | NormalizingFlow/bin_7 | torchregress | 1 | 3 | - | - | - | - | - |
| 110 | NormalizingFlow/bin_8 | torchregress | 1 | 3 | - | - | - | - | - |
| 111 | NF_Ensemble_K2/bin_1 | torchregress | 1 | 3 | - | - | - | - | - |
| 112 | NF_Ensemble_K2/bin_2 | torchregress | 1 | 3 | - | - | - | - | - |
| 113 | NF_Ensemble_K2/bin_3 | torchregress | 1 | 3 | - | - | - | - | - |
| 114 | NF_Ensemble_K2/bin_4 | torchregress | 1 | 3 | - | - | - | - | - |
| 115 | NF_Ensemble_K2/bin_6 | torchregress | 1 | 3 | - | - | - | - | - |
| 116 | NF_Ensemble_K2/bin_7 | torchregress | 1 | 3 | - | - | - | - | - |
| 117 | NF_Ensemble_K2/bin_8 | torchregress | 1 | 3 | - | - | - | - | - |
| 118 | NF_Ensemble_K5/bin_1 | torchregress | 1 | 3 | - | - | - | - | - |
| 119 | NF_Ensemble_K5/bin_2 | torchregress | 1 | 3 | - | - | - | - | - |
| 120 | NF_Ensemble_K5/bin_3 | torchregress | 1 | 3 | - | - | - | - | - |
| 121 | NF_Ensemble_K5/bin_4 | torchregress | 1 | 3 | - | - | - | - | - |
| 122 | NF_Ensemble_K5/bin_6 | torchregress | 1 | 3 | - | - | - | - | - |
| 123 | NF_Ensemble_K5/bin_7 | torchregress | 1 | 3 | - | - | - | - | - |
| 124 | NF_Ensemble_K5/bin_8 | torchregress | 1 | 3 | - | - | - | - | - |
| 125 | NF_Ensemble_K10/bin_1 | torchregress | 1 | 3 | - | - | - | - | - |
| 126 | NF_Ensemble_K10/bin_2 | torchregress | 1 | 3 | - | - | - | - | - |
| 127 | NF_Ensemble_K10/bin_3 | torchregress | 1 | 3 | - | - | - | - | - |
| 128 | NF_Ensemble_K10/bin_4 | torchregress | 1 | 3 | - | - | - | - | - |
| 129 | NF_Ensemble_K10/bin_6 | torchregress | 1 | 3 | - | - | - | - | - |
| 130 | NF_Ensemble_K10/bin_7 | torchregress | 1 | 3 | - | - | - | - | - |
| 131 | NF_Ensemble_K10/bin_8 | torchregress | 1 | 3 | - | - | - | - | - |
| 132 | NF_Ensemble_K20/bin_1 | torchregress | 1 | 3 | - | - | - | - | - |
| 133 | NF_Ensemble_K20/bin_2 | torchregress | 1 | 3 | - | - | - | - | - |
| 134 | NF_Ensemble_K20/bin_3 | torchregress | 1 | 3 | - | - | - | - | - |
| 135 | NF_Ensemble_K20/bin_4 | torchregress | 1 | 3 | - | - | - | - | - |
| 136 | NF_Ensemble_K20/bin_6 | torchregress | 1 | 3 | - | - | - | - | - |
| 137 | NF_Ensemble_K20/bin_7 | torchregress | 1 | 3 | - | - | - | - | - |
| 138 | NF_Ensemble_K20/bin_8 | torchregress | 1 | 3 | - | - | - | - | - |
| 139 | MDN_K3/bin_1 | torchregress | 1 | 3 | - | - | - | - | - |
| 140 | MDN_K3/bin_2 | torchregress | 1 | 3 | - | - | - | - | - |
| 141 | MDN_K3/bin_3 | torchregress | 1 | 3 | - | - | - | - | - |
| 142 | MDN_K3/bin_4 | torchregress | 1 | 3 | - | - | - | - | - |
| 143 | MDN_K3/bin_6 | torchregress | 1 | 3 | - | - | - | - | - |
| 144 | MDN_K3/bin_7 | torchregress | 1 | 3 | - | - | - | - | - |
| 145 | MDN_K3/bin_8 | torchregress | 1 | 3 | - | - | - | - | - |
| 146 | Evidential/bin_1 | torchregress | 1 | 3 | - | - | - | - | - |
| 147 | Evidential/bin_2 | torchregress | 1 | 3 | - | - | - | - | - |
| 148 | Evidential/bin_3 | torchregress | 1 | 3 | - | - | - | - | - |
| 149 | Evidential/bin_4 | torchregress | 1 | 3 | - | - | - | - | - |
| 150 | Evidential/bin_6 | torchregress | 1 | 3 | - | - | - | - | - |
| 151 | Evidential/bin_7 | torchregress | 1 | 3 | - | - | - | - | - |
| 152 | Evidential/bin_8 | torchregress | 1 | 3 | - | - | - | - | - |

## Per-Suite Breakdown

### Suite: `calibration`

#### Dataset: `california_housing`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| IsotonicMeanCalibrator | torchregress | 0.6009 | - | - | 0.0033 |
| Uncalibrated | torchregress | 0.6011 | - | - | 0.0042 |
| VarianceTemperatureScaler | torchregress | 0.6011 | - | - | 0.0041 |
| sklearn_IsotonicRegression | sklearn | 0.6044 | - | - | 0.0035 |

#### Dataset: `diabetes`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| Uncalibrated | torchregress | 55.3688 | - | - | 0.0041 |
| VarianceTemperatureScaler | torchregress | 55.3688 | - | - | 0.0041 |
| IsotonicMeanCalibrator | torchregress | 57.3051 | - | - | 0.0060 |
| sklearn_IsotonicRegression | sklearn | 58.0110 | - | - | 0.0043 |

#### Dataset: `house_sales`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| Uncalibrated | torchregress | 130890.9446 | - | - | 0.0024 |
| VarianceTemperatureScaler | torchregress | 130890.9446 | - | - | 0.0034 |
| IsotonicMeanCalibrator | torchregress | 133092.4134 | - | - | 0.0028 |
| sklearn_IsotonicRegression | sklearn | 135100.4446 | - | - | 0.0033 |

#### Dataset: `synthetic_hetero`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| Uncalibrated | torchregress | 0.3757 | - | - | 0.0033 |
| VarianceTemperatureScaler | torchregress | 0.3757 | - | - | 0.0032 |
| IsotonicMeanCalibrator | torchregress | 0.3925 | - | - | 0.0022 |
| sklearn_IsotonicRegression | sklearn | 0.3993 | - | - | 0.0041 |

### Suite: `conformal`

#### Dataset: `california_housing`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| CQR+MLP | torchregress | - | 2.3930 | 0.8907 | - |
| crepes_Split+TabICL | tabicl | - | 2.4136 | 0.9013 | - |
| Split+TabICL | tabicl | - | 2.4136 | 0.9013 | - |
| MAPIE_Split+TabICL | tabicl | - | 2.4136 | 0.9013 | - |
| MAPIE_CrossCQR+GBR | MAPIE | - | 2.6300 | 0.9060 | - |
| MAPIE_CQR+GBR | MAPIE | - | 2.7308 | 0.8860 | - |
| MAPIE_Split+MLP | MAPIE | - | 2.8911 | 0.8887 | - |
| Split+MLP | torchregress | - | 2.8953 | 0.8913 | - |
| Split+Linear | torchregress | - | 3.5755 | 0.8853 | - |
| crepes_Split+Linear | crepes | - | 3.5913 | 0.8907 | - |
| torchcp_Split+Linear | torchcp | - | 3.5913 | 0.8907 | - |
| MAPIE_Split+Linear | MAPIE | - | 3.5913 | 0.8907 | - |
| Split+Linear(post-hoc) | torchregress | - | 3.5913 | 0.8907 | - |

#### Dataset: `diabetes`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| MAPIE_CQR+GBR | MAPIE | - | 213.9020 | 0.9173 | - |
| MAPIE_CrossCQR+GBR | MAPIE | - | 214.8505 | 0.9323 | - |
| Split+TabICL | tabicl | - | 222.6256 | 0.9323 | - |
| MAPIE_Split+TabICL | tabicl | - | 222.6256 | 0.9323 | - |
| crepes_Split+TabICL | tabicl | - | 222.6256 | 0.9323 | - |
| MAPIE_Split+Linear | MAPIE | - | 223.2080 | 0.9098 | - |
| torchcp_Split+Linear | torchcp | - | 223.2080 | 0.9098 | - |
| crepes_Split+Linear | crepes | - | 223.2080 | 0.9098 | - |
| Split+Linear(post-hoc) | torchregress | - | 223.2084 | 0.9098 | - |
| CQR+MLP | torchregress | - | 224.1899 | 0.8947 | - |
| MAPIE_Split+MLP | MAPIE | - | 234.0333 | 0.9248 | - |
| Split+MLP | torchregress | - | 236.1329 | 0.9398 | - |
| Split+Linear | torchregress | - | 248.3531 | 0.8647 | - |

#### Dataset: `house_sales`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| CQR+MLP | torchregress | - | 434504.6726 | 0.9070 | - |
| MAPIE_CrossCQR+GBR | MAPIE | - | 484462.5667 | 0.9007 | - |
| MAPIE_CQR+GBR | MAPIE | - | 497639.1383 | 0.9005 | - |
| crepes_Split+TabICL | tabicl | - | 525645.8431 | 0.8998 | - |
| Split+TabICL | tabicl | - | 525645.8528 | 0.8998 | - |
| MAPIE_Split+TabICL | tabicl | - | 525645.8528 | 0.8998 | - |
| Split+MLP | torchregress | - | 629405.8826 | 0.9007 | - |
| MAPIE_Split+MLP | MAPIE | - | 656686.2163 | 0.8943 | - |
| Split+Linear(post-hoc) | torchregress | - | 929312.9025 | 0.8977 | - |
| torchcp_Split+Linear | torchcp | - | 929314.6976 | 0.8977 | - |
| crepes_Split+Linear | crepes | - | 929314.6984 | 0.8977 | - |
| MAPIE_Split+Linear | MAPIE | - | 929314.7071 | 0.8977 | - |
| Split+Linear | torchregress | - | 930346.7065 | 0.8983 | - |

#### Dataset: `synthetic_hetero`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| Split+TabICL | tabicl | - | 1.5687 | 0.9217 | - |
| MAPIE_Split+TabICL | tabicl | - | 1.5687 | 0.9217 | - |
| crepes_Split+TabICL | tabicl | - | 1.5687 | 0.9217 | - |
| CQR+MLP | torchregress | - | 1.6443 | 0.9150 | - |
| Split+MLP | torchregress | - | 1.6469 | 0.9083 | - |
| MAPIE_Split+Linear | MAPIE | - | 1.6865 | 0.9067 | - |
| crepes_Split+Linear | crepes | - | 1.6865 | 0.9067 | - |
| torchcp_Split+Linear | torchcp | - | 1.6865 | 0.9067 | - |
| Split+Linear(post-hoc) | torchregress | - | 1.6865 | 0.9067 | - |
| MAPIE_CrossCQR+GBR | MAPIE | - | 1.7140 | 0.9133 | - |
| MAPIE_Split+MLP | MAPIE | - | 1.7519 | 0.8917 | - |
| Split+Linear | torchregress | - | 1.7638 | 0.9233 | - |
| MAPIE_CQR+GBR | MAPIE | - | 1.8547 | 0.9200 | - |

### Suite: `distributional_bins`

#### Dataset: `california_housing`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| NF_Ensemble_K20 | torchregress | - | 1.8081 | 0.9153 | - |
| NF_Ensemble_K10 | torchregress | - | 1.8121 | 0.9113 | - |
| NF_Ensemble_K5 | torchregress | - | 1.8498 | 0.8960 | - |
| NF_Ensemble_K2 | torchregress | - | 1.8654 | 0.8673 | - |
| NormalizingFlow | torchregress | - | 1.9854 | 0.8553 | - |
| MDN_K3 | torchregress | - | 2.0693 | 0.8813 | - |
| GaussianNLL | torchregress | - | 2.0970 | 0.9160 | - |
| Split_CP | torchregress | - | 2.4912 | 0.8900 | - |
| Evidential | torchregress | - | 2.7724 | 0.9840 | - |
| Split_CP/bin_0 | torchregress | - | - | 0.8637 | - |
| Split_CP/bin_1 | torchregress | - | - | - | - |
| Split_CP/bin_2 | torchregress | - | - | - | - |
| Split_CP/bin_3 | torchregress | - | - | - | - |
| Split_CP/bin_4 | torchregress | - | - | - | - |
| Split_CP/bin_5 | torchregress | - | - | 0.8955 | - |
| Split_CP/bin_6 | torchregress | - | - | - | - |
| Split_CP/bin_7 | torchregress | - | - | - | - |
| Split_CP/bin_8 | torchregress | - | - | - | - |
| Split_CP/bin_9 | torchregress | - | - | 0.9049 | - |
| Split_CP/dev_summary | torchregress | - | - | - | - |
| GaussianNLL/bin_0 | torchregress | - | - | 0.9075 | - |
| GaussianNLL/bin_1 | torchregress | - | - | - | - |
| GaussianNLL/bin_2 | torchregress | - | - | - | - |
| GaussianNLL/bin_3 | torchregress | - | - | - | - |
| GaussianNLL/bin_4 | torchregress | - | - | - | - |
| GaussianNLL/bin_5 | torchregress | - | - | 0.9181 | - |
| GaussianNLL/bin_6 | torchregress | - | - | - | - |
| GaussianNLL/bin_7 | torchregress | - | - | - | - |
| GaussianNLL/bin_8 | torchregress | - | - | - | - |
| GaussianNLL/bin_9 | torchregress | - | - | 0.9204 | - |
| GaussianNLL/dev_summary | torchregress | - | - | - | - |
| NormalizingFlow/bin_0 | torchregress | - | - | 0.8345 | - |
| NormalizingFlow/bin_1 | torchregress | - | - | - | - |
| NormalizingFlow/bin_2 | torchregress | - | - | - | - |
| NormalizingFlow/bin_3 | torchregress | - | - | - | - |
| NormalizingFlow/bin_4 | torchregress | - | - | - | - |
| NormalizingFlow/bin_5 | torchregress | - | - | 0.8833 | - |
| NormalizingFlow/bin_6 | torchregress | - | - | - | - |
| NormalizingFlow/bin_7 | torchregress | - | - | - | - |
| NormalizingFlow/bin_8 | torchregress | - | - | - | - |
| NormalizingFlow/bin_9 | torchregress | - | - | 0.8408 | - |
| NormalizingFlow/dev_summary | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_0 | torchregress | - | - | 0.8345 | - |
| NF_Ensemble_K2/bin_1 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_2 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_3 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_4 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_5 | torchregress | - | - | 0.8902 | - |
| NF_Ensemble_K2/bin_6 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_7 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_8 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_9 | torchregress | - | - | 0.8680 | - |
| NF_Ensemble_K2/dev_summary | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_0 | torchregress | - | - | 0.8735 | - |
| NF_Ensemble_K5/bin_1 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_2 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_3 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_4 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_5 | torchregress | - | - | 0.9111 | - |
| NF_Ensemble_K5/bin_6 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_7 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_8 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_9 | torchregress | - | - | 0.8971 | - |
| NF_Ensemble_K5/dev_summary | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_0 | torchregress | - | - | 0.8954 | - |
| NF_Ensemble_K10/bin_1 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_2 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_3 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_4 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_5 | torchregress | - | - | 0.9164 | - |
| NF_Ensemble_K10/bin_6 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_7 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_8 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_9 | torchregress | - | - | 0.9184 | - |
| NF_Ensemble_K10/dev_summary | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_0 | torchregress | - | - | 0.8978 | - |
| NF_Ensemble_K20/bin_1 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_2 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_3 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_4 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_5 | torchregress | - | - | 0.9181 | - |
| NF_Ensemble_K20/bin_6 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_7 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_8 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_9 | torchregress | - | - | 0.9262 | - |
| NF_Ensemble_K20/dev_summary | torchregress | - | - | - | - |
| MDN_K3/bin_0 | torchregress | - | - | 0.8613 | - |
| MDN_K3/bin_1 | torchregress | - | - | - | - |
| MDN_K3/bin_2 | torchregress | - | - | - | - |
| MDN_K3/bin_3 | torchregress | - | - | - | - |
| MDN_K3/bin_4 | torchregress | - | - | - | - |
| MDN_K3/bin_5 | torchregress | - | - | 0.8833 | - |
| MDN_K3/bin_6 | torchregress | - | - | - | - |
| MDN_K3/bin_7 | torchregress | - | - | - | - |
| MDN_K3/bin_8 | torchregress | - | - | - | - |
| MDN_K3/bin_9 | torchregress | - | - | 0.8951 | - |
| MDN_K3/dev_summary | torchregress | - | - | - | - |
| Evidential/bin_0 | torchregress | - | - | 0.9781 | - |
| Evidential/bin_1 | torchregress | - | - | - | - |
| Evidential/bin_2 | torchregress | - | - | - | - |
| Evidential/bin_3 | torchregress | - | - | - | - |
| Evidential/bin_4 | torchregress | - | - | - | - |
| Evidential/bin_5 | torchregress | - | - | 0.9826 | - |
| Evidential/bin_6 | torchregress | - | - | - | - |
| Evidential/bin_7 | torchregress | - | - | - | - |
| Evidential/bin_8 | torchregress | - | - | - | - |
| Evidential/bin_9 | torchregress | - | - | 0.9903 | - |
| Evidential/dev_summary | torchregress | - | - | - | - |

#### Dataset: `diabetes`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| NF_Ensemble_K5 | torchregress | - | 2.8294 | 0.7820 | - |
| NF_Ensemble_K20 | torchregress | - | 2.8369 | 0.8045 | - |
| NF_Ensemble_K10 | torchregress | - | 2.8939 | 0.7970 | - |
| MDN_K3 | torchregress | - | 2.9660 | 0.7744 | - |
| Split_CP | torchregress | - | 2.9688 | 0.9098 | - |
| GaussianNLL | torchregress | - | 2.9904 | 0.7820 | - |
| Evidential | torchregress | - | 3.0123 | 0.9699 | - |
| NF_Ensemble_K2 | torchregress | - | 3.1592 | 0.7744 | - |
| NormalizingFlow | torchregress | - | 4.5317 | 0.6917 | - |
| Split_CP/bin_0 | torchregress | - | - | 0.9098 | - |
| Split_CP/dev_summary | torchregress | - | - | - | - |
| GaussianNLL/bin_0 | torchregress | - | - | 0.7820 | - |
| GaussianNLL/dev_summary | torchregress | - | - | - | - |
| NormalizingFlow/bin_0 | torchregress | - | - | 0.6917 | - |
| NormalizingFlow/dev_summary | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_0 | torchregress | - | - | 0.7744 | - |
| NF_Ensemble_K2/dev_summary | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_0 | torchregress | - | - | 0.7820 | - |
| NF_Ensemble_K5/dev_summary | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_0 | torchregress | - | - | 0.7970 | - |
| NF_Ensemble_K10/dev_summary | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_0 | torchregress | - | - | 0.8045 | - |
| NF_Ensemble_K20/dev_summary | torchregress | - | - | - | - |
| MDN_K3/bin_0 | torchregress | - | - | 0.7744 | - |
| MDN_K3/dev_summary | torchregress | - | - | - | - |
| Evidential/bin_0 | torchregress | - | - | 0.9699 | - |
| Evidential/dev_summary | torchregress | - | - | - | - |

#### Dataset: `house_sales`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| MDN_K3 | torchregress | - | 1.1829 | 0.8025 | - |
| GaussianNLL | torchregress | - | 1.2410 | 0.8217 | - |
| NF_Ensemble_K20 | torchregress | - | 1.2770 | 0.8690 | - |
| NF_Ensemble_K10 | torchregress | - | 1.3042 | 0.8535 | - |
| NF_Ensemble_K5 | torchregress | - | 1.3129 | 0.8560 | - |
| NF_Ensemble_K2 | torchregress | - | 1.4208 | 0.8115 | - |
| Evidential | torchregress | - | 1.5325 | 0.9860 | - |
| Split_CP | torchregress | - | 1.7090 | 0.9007 | - |
| NormalizingFlow | torchregress | - | 1.7307 | 0.7248 | - |
| Split_CP/bin_0 | torchregress | - | - | 0.2000 | - |
| Split_CP/bin_1 | torchregress | - | - | - | - |
| Split_CP/bin_2 | torchregress | - | - | - | - |
| Split_CP/bin_3 | torchregress | - | - | - | - |
| Split_CP/bin_4 | torchregress | - | - | - | - |
| Split_CP/bin_5 | torchregress | - | - | 0.8254 | - |
| Split_CP/bin_6 | torchregress | - | - | 0.9323 | - |
| Split_CP/bin_7 | torchregress | - | - | 0.4444 | - |
| Split_CP/bin_8 | torchregress | - | - | - | - |
| Split_CP/bin_9 | torchregress | - | - | 0.1316 | - |
| Split_CP/dev_summary | torchregress | - | - | - | - |
| GaussianNLL/bin_0 | torchregress | - | - | 0.8000 | - |
| GaussianNLL/bin_1 | torchregress | - | - | - | - |
| GaussianNLL/bin_2 | torchregress | - | - | - | - |
| GaussianNLL/bin_3 | torchregress | - | - | - | - |
| GaussianNLL/bin_4 | torchregress | - | - | - | - |
| GaussianNLL/bin_5 | torchregress | - | - | 0.8309 | - |
| GaussianNLL/bin_6 | torchregress | - | - | 0.8195 | - |
| GaussianNLL/bin_7 | torchregress | - | - | 0.8272 | - |
| GaussianNLL/bin_8 | torchregress | - | - | - | - |
| GaussianNLL/bin_9 | torchregress | - | - | 0.8158 | - |
| GaussianNLL/dev_summary | torchregress | - | - | - | - |
| NormalizingFlow/bin_0 | torchregress | - | - | 0.0000 | - |
| NormalizingFlow/bin_1 | torchregress | - | - | - | - |
| NormalizingFlow/bin_2 | torchregress | - | - | - | - |
| NormalizingFlow/bin_3 | torchregress | - | - | - | - |
| NormalizingFlow/bin_4 | torchregress | - | - | - | - |
| NormalizingFlow/bin_5 | torchregress | - | - | 0.7169 | - |
| NormalizingFlow/bin_6 | torchregress | - | - | 0.7312 | - |
| NormalizingFlow/bin_7 | torchregress | - | - | 0.6543 | - |
| NormalizingFlow/bin_8 | torchregress | - | - | - | - |
| NormalizingFlow/bin_9 | torchregress | - | - | 0.3947 | - |
| NormalizingFlow/dev_summary | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_0 | torchregress | - | - | 0.0000 | - |
| NF_Ensemble_K2/bin_1 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_2 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_3 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_4 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_5 | torchregress | - | - | 0.8180 | - |
| NF_Ensemble_K2/bin_6 | torchregress | - | - | 0.8150 | - |
| NF_Ensemble_K2/bin_7 | torchregress | - | - | 0.7654 | - |
| NF_Ensemble_K2/bin_8 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_9 | torchregress | - | - | 0.3947 | - |
| NF_Ensemble_K2/dev_summary | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_0 | torchregress | - | - | 0.0000 | - |
| NF_Ensemble_K5/bin_1 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_2 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_3 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_4 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_5 | torchregress | - | - | 0.8658 | - |
| NF_Ensemble_K5/bin_6 | torchregress | - | - | 0.8588 | - |
| NF_Ensemble_K5/bin_7 | torchregress | - | - | 0.8272 | - |
| NF_Ensemble_K5/bin_8 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_9 | torchregress | - | - | 0.3947 | - |
| NF_Ensemble_K5/dev_summary | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_0 | torchregress | - | - | 0.0000 | - |
| NF_Ensemble_K10/bin_1 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_2 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_3 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_4 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_5 | torchregress | - | - | 0.8557 | - |
| NF_Ensemble_K10/bin_6 | torchregress | - | - | 0.8580 | - |
| NF_Ensemble_K10/bin_7 | torchregress | - | - | 0.8272 | - |
| NF_Ensemble_K10/bin_8 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_9 | torchregress | - | - | 0.3947 | - |
| NF_Ensemble_K10/dev_summary | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_0 | torchregress | - | - | 0.0000 | - |
| NF_Ensemble_K20/bin_1 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_2 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_3 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_4 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_5 | torchregress | - | - | 0.8649 | - |
| NF_Ensemble_K20/bin_6 | torchregress | - | - | 0.8749 | - |
| NF_Ensemble_K20/bin_7 | torchregress | - | - | 0.8519 | - |
| NF_Ensemble_K20/bin_8 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_9 | torchregress | - | - | 0.3947 | - |
| NF_Ensemble_K20/dev_summary | torchregress | - | - | - | - |
| MDN_K3/bin_0 | torchregress | - | - | 0.8000 | - |
| MDN_K3/bin_1 | torchregress | - | - | - | - |
| MDN_K3/bin_2 | torchregress | - | - | - | - |
| MDN_K3/bin_3 | torchregress | - | - | - | - |
| MDN_K3/bin_4 | torchregress | - | - | - | - |
| MDN_K3/bin_5 | torchregress | - | - | 0.8327 | - |
| MDN_K3/bin_6 | torchregress | - | - | 0.7951 | - |
| MDN_K3/bin_7 | torchregress | - | - | 0.8395 | - |
| MDN_K3/bin_8 | torchregress | - | - | - | - |
| MDN_K3/bin_9 | torchregress | - | - | 0.7895 | - |
| MDN_K3/dev_summary | torchregress | - | - | - | - |
| Evidential/bin_0 | torchregress | - | - | 0.8000 | - |
| Evidential/bin_1 | torchregress | - | - | - | - |
| Evidential/bin_2 | torchregress | - | - | - | - |
| Evidential/bin_3 | torchregress | - | - | - | - |
| Evidential/bin_4 | torchregress | - | - | - | - |
| Evidential/bin_5 | torchregress | - | - | 0.9770 | - |
| Evidential/bin_6 | torchregress | - | - | 0.9889 | - |
| Evidential/bin_7 | torchregress | - | - | 0.9753 | - |
| Evidential/bin_8 | torchregress | - | - | - | - |
| Evidential/bin_9 | torchregress | - | - | 0.9211 | - |
| Evidential/dev_summary | torchregress | - | - | - | - |

#### Dataset: `synthetic_hetero`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| NF_Ensemble_K20 | torchregress | - | 1.5837 | 0.8667 | - |
| NF_Ensemble_K10 | torchregress | - | 1.6016 | 0.8517 | - |
| NF_Ensemble_K5 | torchregress | - | 1.6633 | 0.8183 | - |
| Split_CP | torchregress | - | 1.7195 | 0.9067 | - |
| MDN_K3 | torchregress | - | 1.7609 | 0.8600 | - |
| GaussianNLL | torchregress | - | 1.7622 | 0.8383 | - |
| NF_Ensemble_K2 | torchregress | - | 1.8183 | 0.8000 | - |
| Evidential | torchregress | - | 2.0968 | 0.9867 | - |
| NormalizingFlow | torchregress | - | 2.2349 | 0.7050 | - |
| Split_CP/bin_0 | torchregress | - | - | 0.8276 | - |
| Split_CP/bin_1 | torchregress | - | - | - | - |
| Split_CP/bin_2 | torchregress | - | - | - | - |
| Split_CP/bin_3 | torchregress | - | - | - | - |
| Split_CP/bin_4 | torchregress | - | - | - | - |
| Split_CP/bin_5 | torchregress | - | - | 0.9171 | - |
| Split_CP/bin_6 | torchregress | - | - | - | - |
| Split_CP/bin_7 | torchregress | - | - | - | - |
| Split_CP/bin_8 | torchregress | - | - | - | - |
| Split_CP/bin_9 | torchregress | - | - | 0.7857 | - |
| Split_CP/dev_summary | torchregress | - | - | - | - |
| GaussianNLL/bin_0 | torchregress | - | - | 0.8966 | - |
| GaussianNLL/bin_1 | torchregress | - | - | - | - |
| GaussianNLL/bin_2 | torchregress | - | - | - | - |
| GaussianNLL/bin_3 | torchregress | - | - | - | - |
| GaussianNLL/bin_4 | torchregress | - | - | - | - |
| GaussianNLL/bin_5 | torchregress | - | - | 0.8398 | - |
| GaussianNLL/bin_6 | torchregress | - | - | - | - |
| GaussianNLL/bin_7 | torchregress | - | - | - | - |
| GaussianNLL/bin_8 | torchregress | - | - | - | - |
| GaussianNLL/bin_9 | torchregress | - | - | 0.7500 | - |
| GaussianNLL/dev_summary | torchregress | - | - | - | - |
| NormalizingFlow/bin_0 | torchregress | - | - | 0.8966 | - |
| NormalizingFlow/bin_1 | torchregress | - | - | - | - |
| NormalizingFlow/bin_2 | torchregress | - | - | - | - |
| NormalizingFlow/bin_3 | torchregress | - | - | - | - |
| NormalizingFlow/bin_4 | torchregress | - | - | - | - |
| NormalizingFlow/bin_5 | torchregress | - | - | 0.6906 | - |
| NormalizingFlow/bin_6 | torchregress | - | - | - | - |
| NormalizingFlow/bin_7 | torchregress | - | - | - | - |
| NormalizingFlow/bin_8 | torchregress | - | - | - | - |
| NormalizingFlow/bin_9 | torchregress | - | - | 0.7857 | - |
| NormalizingFlow/dev_summary | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_0 | torchregress | - | - | 0.9310 | - |
| NF_Ensemble_K2/bin_1 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_2 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_3 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_4 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_5 | torchregress | - | - | 0.7901 | - |
| NF_Ensemble_K2/bin_6 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_7 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_8 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_9 | torchregress | - | - | 0.8571 | - |
| NF_Ensemble_K2/dev_summary | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_0 | torchregress | - | - | 0.9310 | - |
| NF_Ensemble_K5/bin_1 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_2 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_3 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_4 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_5 | torchregress | - | - | 0.8103 | - |
| NF_Ensemble_K5/bin_6 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_7 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_8 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_9 | torchregress | - | - | 0.8571 | - |
| NF_Ensemble_K5/dev_summary | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_0 | torchregress | - | - | 0.9655 | - |
| NF_Ensemble_K10/bin_1 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_2 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_3 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_4 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_5 | torchregress | - | - | 0.8435 | - |
| NF_Ensemble_K10/bin_6 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_7 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_8 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_9 | torchregress | - | - | 0.8929 | - |
| NF_Ensemble_K10/dev_summary | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_0 | torchregress | - | - | 0.9310 | - |
| NF_Ensemble_K20/bin_1 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_2 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_3 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_4 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_5 | torchregress | - | - | 0.8619 | - |
| NF_Ensemble_K20/bin_6 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_7 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_8 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_9 | torchregress | - | - | 0.8929 | - |
| NF_Ensemble_K20/dev_summary | torchregress | - | - | - | - |
| MDN_K3/bin_0 | torchregress | - | - | 0.9655 | - |
| MDN_K3/bin_1 | torchregress | - | - | - | - |
| MDN_K3/bin_2 | torchregress | - | - | - | - |
| MDN_K3/bin_3 | torchregress | - | - | - | - |
| MDN_K3/bin_4 | torchregress | - | - | - | - |
| MDN_K3/bin_5 | torchregress | - | - | 0.8582 | - |
| MDN_K3/bin_6 | torchregress | - | - | - | - |
| MDN_K3/bin_7 | torchregress | - | - | - | - |
| MDN_K3/bin_8 | torchregress | - | - | - | - |
| MDN_K3/bin_9 | torchregress | - | - | 0.7857 | - |
| MDN_K3/dev_summary | torchregress | - | - | - | - |
| Evidential/bin_0 | torchregress | - | - | 1.0000 | - |
| Evidential/bin_1 | torchregress | - | - | - | - |
| Evidential/bin_2 | torchregress | - | - | - | - |
| Evidential/bin_3 | torchregress | - | - | - | - |
| Evidential/bin_4 | torchregress | - | - | - | - |
| Evidential/bin_5 | torchregress | - | - | 0.9871 | - |
| Evidential/bin_6 | torchregress | - | - | - | - |
| Evidential/bin_7 | torchregress | - | - | - | - |
| Evidential/bin_8 | torchregress | - | - | - | - |
| Evidential/bin_9 | torchregress | - | - | 0.9643 | - |
| Evidential/dev_summary | torchregress | - | - | - | - |

### Suite: `eiv`

#### Dataset: `california_housing`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| MSE | torchregress | 0.5894 | - | - | - |
| MSE_NoisyPredict | torchregress | 0.5917 | - | - | - |
| RC+GaussianNLL | torchregress | 0.6032 | - | - | 0.0039 |
| FunctionalEIV | torchregress | 0.6127 | - | - | - |
| StructuralEIV | torchregress | 0.6127 | - | - | - |
| EnsembleEIV | torchregress | 0.6129 | - | - | - |
| RC | torchregress | 0.6159 | - | - | - |
| RC+FunctionalEIV | torchregress | 0.6230 | - | - | - |
| GaussianNLL+InputNoiseMC | torchregress | 0.6237 | - | - | 0.0053 |
| MSE_WarmupMC | torchregress | 0.6288 | - | - | - |
| InputNoiseMC | torchregress | 0.6324 | - | - | - |
| FunctionalEIV_MC | torchregress | 0.6692 | - | - | - |
| ODR | torchregress | 0.7410 | - | - | - |
| LatentNN | torchregress | 0.7626 | - | - | - |
| SIMEX | torchregress | 1.0319 | - | - | - |
| SplitCP+FunctionalEIV | torchregress+CP | 0.6102 | 2.8973 | 0.9000 | 0.0057 |
| SplitCP+MSE_WarmupMC | torchregress+CP | 0.6180 | 2.9415 | 0.9053 | 0.0048 |
| SplitCP+InputNoiseMC | torchregress+CP | 0.6349 | 2.9693 | 0.8793 | 0.0050 |
| TLS_Init+FunctionalEIV | torchregress | 16.0207 | - | - | - |
| TLS | numpy | 16.0558 | - | - | - |

#### Dataset: `diabetes`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| FunctionalEIV | torchregress | 52.6244 | - | - | - |
| StructuralEIV | torchregress | 52.6244 | - | - | - |
| GaussianNLL+InputNoiseMC | torchregress | 53.0530 | - | - | 0.0040 |
| RC+GaussianNLL | torchregress | 53.1008 | - | - | 0.0073 |
| InputNoiseMC | torchregress | 53.3377 | - | - | - |
| RC+FunctionalEIV | torchregress | 53.3813 | - | - | - |
| RC | torchregress | 53.6673 | - | - | - |
| EnsembleEIV | torchregress | 53.7794 | - | - | - |
| MSE_NoisyPredict | torchregress | 53.8988 | - | - | - |
| MSE | torchregress | 54.0167 | - | - | - |
| ODR | torchregress | 57.4657 | - | - | - |
| SIMEX | torchregress | 57.5278 | - | - | - |
| LatentNN | torchregress | 61.1186 | - | - | - |
| FunctionalEIV_MC | torchregress | 62.1385 | - | - | - |
| MSE_WarmupMC | torchregress | 67.4930 | - | - | - |
| SplitCP+FunctionalEIV | torchregress+CP | 54.1719 | 225.4651 | 0.9173 | 0.0045 |
| SplitCP+InputNoiseMC | torchregress+CP | 54.7880 | 226.3255 | 0.9098 | 0.0067 |
| SplitCP+MSE_WarmupMC | torchregress+CP | 69.0953 | 295.7608 | 0.8872 | 0.0099 |
| TLS_Init+FunctionalEIV | torchregress | 449.5006 | - | - | - |
| TLS | numpy | 449.5673 | - | - | - |

#### Dataset: `house_sales`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| RC+FunctionalEIV | torchregress | 126780.2266 | - | - | - |
| FunctionalEIV | torchregress | 128024.4636 | - | - | - |
| StructuralEIV | torchregress | 128024.4636 | - | - | - |
| MSE_NoisyPredict | torchregress | 128656.2112 | - | - | - |
| MSE | torchregress | 129082.1319 | - | - | - |
| FunctionalEIV_MC | torchregress | 130642.2415 | - | - | - |
| RC+GaussianNLL | torchregress | 131303.0635 | - | - | 0.0016 |
| RC | torchregress | 131443.4050 | - | - | - |
| GaussianNLL+InputNoiseMC | torchregress | 131934.3557 | - | - | 0.0024 |
| InputNoiseMC | torchregress | 133110.5841 | - | - | - |
| EnsembleEIV | torchregress | 137142.8648 | - | - | - |
| MSE_WarmupMC | torchregress | 139367.8717 | - | - | - |
| LatentNN | torchregress | 184619.9864 | - | - | - |
| SIMEX | torchregress | 229201.1373 | - | - | - |
| SplitCP+MSE_WarmupMC | torchregress+CP | 133238.0717 | 629581.9143 | 0.9053 | 0.0060 |
| SplitCP+FunctionalEIV | torchregress+CP | 135008.4153 | 640872.4604 | 0.9043 | 0.0088 |
| SplitCP+InputNoiseMC | torchregress+CP | 140032.3853 | 655344.7594 | 0.9007 | 0.0072 |
| ODR | torchregress | 973668.1069 | - | - | - |
| TLS | numpy | 3207387.5493 | - | - | - |
| TLS_Init+FunctionalEIV | torchregress | 5257513.5458 | - | - | - |

#### Dataset: `synthetic_hetero`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| GaussianNLL+InputNoiseMC | torchregress | 0.3742 | - | - | 0.0029 |
| MSE_NoisyPredict | torchregress | 0.3757 | - | - | - |
| MSE | torchregress | 0.3793 | - | - | - |
| RC+GaussianNLL | torchregress | 0.3800 | - | - | 0.0038 |
| InputNoiseMC | torchregress | 0.3803 | - | - | - |
| RC | torchregress | 0.3813 | - | - | - |
| RC+FunctionalEIV | torchregress | 0.3820 | - | - | - |
| FunctionalEIV | torchregress | 0.3823 | - | - | - |
| StructuralEIV | torchregress | 0.3823 | - | - | - |
| EnsembleEIV | torchregress | 0.3840 | - | - | - |
| TLS_Init+FunctionalEIV | torchregress | 0.3864 | - | - | - |
| TLS | numpy | 0.4000 | - | - | - |
| LatentNN | torchregress | 0.4050 | - | - | - |
| FunctionalEIV_MC | torchregress | 0.4064 | - | - | - |
| ODR | torchregress | 0.4168 | - | - | - |
| MSE_WarmupMC | torchregress | 0.4217 | - | - | - |
| SIMEX | torchregress | 0.6709 | - | - | - |
| SplitCP+InputNoiseMC | torchregress+CP | 0.3812 | 1.6670 | 0.9050 | 0.0067 |
| SplitCP+FunctionalEIV | torchregress+CP | 0.3860 | 1.6712 | 0.9083 | 0.0068 |
| SplitCP+MSE_WarmupMC | torchregress+CP | 0.4159 | 1.7718 | 0.9000 | 0.0083 |

### Suite: `pzflow_parity`

#### Dataset: `california_housing`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| torchregress_NSF | torchregress | - | 1.9276 | 0.8427 | - |
| pzflow_NSF | pzflow | - | 3.8276 | 0.8947 | - |
| normflows_NSF | normflows | - | - | - | - |

#### Dataset: `diabetes`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| torchregress_NSF | torchregress | - | 2.8537 | 0.7594 | - |
| pzflow_NSF | pzflow | - | 3.0834 | 0.8947 | - |
| normflows_NSF | normflows | - | - | - | - |

#### Dataset: `house_sales`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| torchregress_NSF | torchregress | - | 1.5813 | 0.7560 | - |
| pzflow_NSF | pzflow | - | 4.1493 | 0.9148 | - |
| normflows_NSF | normflows | - | - | - | - |

#### Dataset: `synthetic_hetero`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| torchregress_NSF | torchregress | - | 2.1259 | 0.7300 | - |
| pzflow_NSF | pzflow | - | 4.0193 | 0.8967 | - |
| normflows_NSF | normflows | - | - | - | - |

### Suite: `r2c_conformal`

#### Dataset: `california_housing`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| CQR+MLP | torchregress | 0.6741 | 2.3928 | 0.8927 | - |
| R2C+CORAL | torchregress | 0.6478 | 2.6399 | 0.9253 | - |
| R2C+CumulativeLink | torchregress | 0.7068 | 2.6487 | 0.9520 | - |
| R2C+Softmax | torchregress | 0.8017 | 2.6697 | 0.9533 | - |
| Split+MLP | torchregress | 0.6088 | 2.8947 | 0.8913 | - |
| torchcp_R2C | torchcp | 1.2405 | 8.2134 | 0.6427 | - |

#### Dataset: `diabetes`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| CQR+MLP | torchregress | 55.5782 | 223.6176 | 0.9173 | - |
| Split+MLP | torchregress | 55.9686 | 237.5038 | 0.9398 | - |
| R2C+CORAL | torchregress | 61.1563 | 247.1738 | 0.9248 | - |
| R2C+Softmax | torchregress | 67.3498 | 266.1066 | 1.0000 | - |
| R2C+CumulativeLink | torchregress | 62.2526 | 358.3808 | 1.0000 | - |
| torchcp_R2C | torchcp | 78.6300 | 634.9944 | 0.7594 | - |

#### Dataset: `house_sales`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| CQR+MLP | torchregress | 130230.3663 | 434538.0146 | 0.9075 | - |
| Split+MLP | torchregress | 132367.9050 | 629407.3623 | 0.9007 | - |
| R2C+Softmax | torchregress | 151107.4106 | 643287.1834 | 0.9897 | - |
| R2C+CORAL | torchregress | 161239.3092 | 685313.2596 | 0.9728 | - |
| R2C+CumulativeLink | torchregress | 153926.1386 | 695821.2972 | 0.9813 | - |
| torchcp_R2C | torchcp | 2309118.0808 | 4696537.4568 | 0.0000 | - |

#### Dataset: `synthetic_hetero`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| CQR+MLP | torchregress | 0.3963 | 1.6445 | 0.9167 | - |
| Split+MLP | torchregress | 0.3818 | 1.6470 | 0.9083 | - |
| R2C+CORAL | torchregress | 0.3831 | 1.6595 | 0.9433 | - |
| R2C+Softmax | torchregress | 0.4726 | 2.0259 | 0.9583 | - |
| R2C+CumulativeLink | torchregress | 0.4300 | 2.1355 | 0.9683 | - |
| torchcp_R2C | torchcp | 2.7509 | 9.4380 | 1.0000 | - |
