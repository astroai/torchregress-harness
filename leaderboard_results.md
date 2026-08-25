# Comprehensive Regression Leaderboard

- Target coverage: **0.90**
- Methods ranked: **194**
- Primary ranking metric: **IntervalScore** (lower is better)

## Leaderboard (suite-mean by method)

| Rank | Method | Library | N Suites | N Runs | RMSE | IntervalScore | Coverage | Coverage Dev | ECE |
|---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 1 | NF_Ensemble_K20 | torchregress | 1 | 4 | - | 1.8764 | 0.8639 | 0.0438 | - |
| 2 | NF_Ensemble_K10 | torchregress | 1 | 4 | - | 1.9029 | 0.8534 | 0.0523 | - |
| 3 | NF_Ensemble_K5 | torchregress | 1 | 4 | - | 1.9139 | 0.8381 | 0.0619 | - |
| 4 | MDN_K3 | torchregress | 1 | 4 | - | 1.9948 | 0.8296 | 0.0704 | - |
| 5 | GaussianNLL | torchregress | 4 | 16 | 30862.8479 | 2.0227 | 0.8395 | 0.0685 | 0.0040 |
| 6 | NF_Ensemble_K2 | torchregress | 1 | 4 | - | 2.0659 | 0.8133 | 0.0867 | - |
| 7 | torchregress_NSF | torchregress | 1 | 4 | - | 2.1221 | 0.7720 | 0.1280 | - |
| 8 | Split_CP | torchregress | 1 | 4 | - | 2.2221 | 0.9018 | 0.0068 | - |
| 9 | Evidential | torchregress | 1 | 4 | - | 2.3535 | 0.9816 | 0.0816 | - |
| 10 | NormalizingFlow | torchregress | 1 | 4 | - | 2.6207 | 0.7442 | 0.1558 | - |
| 11 | pzflow_NSF | pzflow | 1 | 4 | - | 3.7699 | 0.9002 | 0.0072 | - |
| 12 | CQR+MLP | torchregress | 1 | 4 | 32571.7537 | 108691.4174 | 0.9085 | 0.0122 | - |
| 13 | DeepEnsemble_MultiQuantile | torchregress | 1 | 4 | 29095.6729 | 110749.9233 | 0.7840 | 0.1160 | 0.0092 |
| 14 | sklearn_QuantileGBR | sklearn | 1 | 4 | 37026.7793 | 118317.8723 | 0.8618 | 0.0382 | 0.0045 |
| 15 | MultiQuantile | torchregress | 1 | 4 | 31422.0738 | 123995.9880 | 0.7625 | 0.1375 | 0.0065 |
| 16 | SplitCP+MSE_WarmupMC | torchregress+CP | 1 | 4 | 33412.1768 | 157356.9359 | 0.8990 | 0.0062 | 0.0074 |
| 17 | Split+MLP | torchregress | 1 | 4 | 33106.2160 | 157412.3519 | 0.9100 | 0.0144 | - |
| 18 | SplitCP+FunctionalEIV | torchregress+CP | 1 | 4 | 33765.8958 | 160275.6235 | 0.9075 | 0.0075 | 0.0065 |
| 19 | R2C+Softmax | torchregress | 1 | 4 | 37794.0087 | 160889.4964 | 0.9753 | 0.0753 | - |
| 20 | SplitCP+InputNoiseMC | torchregress+CP | 1 | 4 | 34945.1138 | 163332.7371 | 0.8982 | 0.0092 | 0.0063 |
| 21 | R2C+CORAL | torchregress | 1 | 4 | 40325.3741 | 171391.1832 | 0.9416 | 0.0416 | - |
| 22 | R2C+CumulativeLink | torchregress | 1 | 4 | 38497.3820 | 174046.1155 | 0.9754 | 0.0754 | - |
| 23 | torchcp_R2C | torchcp | 1 | 4 | 577300.1755 | 1174297.5256 | 0.6005 | 0.3495 | - |
| 24 | statsmodels_OrderedModel | statsmodels | 1 | 4 | 941.9790 | - | - | - | - |
| 25 | MSE | torchregress | 4 | 16 | 24213.4182 | - | - | - | - |
| 26 | TabICL | tabicl | 1 | 4 | 27972.7578 | - | - | - | - |
| 27 | DeepEnsemble_K10 | torchregress | 1 | 4 | 29572.1354 | - | - | - | 0.0044 |
| 28 | DeepEnsemble_GaussianNLL | torchregress | 1 | 4 | 29712.2363 | - | - | - | 0.0038 |
| 29 | DeepEnsemble_MSE | torchregress | 1 | 4 | 30380.6472 | - | - | - | - |
| 30 | sklearn_HistGradientBoosting | sklearn | 1 | 4 | 30663.2040 | - | - | - | - |
| 31 | Charbonnier | torchregress | 1 | 4 | 30926.7035 | - | - | - | - |
| 32 | MDN | torchregress | 1 | 4 | 31556.9922 | - | - | - | 0.0044 |
| 33 | FunctionalEIV | torchregress | 1 | 4 | 32019.5207 | - | - | - | - |
| 34 | StructuralEIV | torchregress | 1 | 4 | 32019.5207 | - | - | - | - |
| 35 | MSE_NoisyPredict | torchregress | 1 | 4 | 32177.7693 | - | - | - | - |
| 36 | FunctionalEIV_MC | torchregress | 1 | 4 | 32315.8545 | - | - | - | - |
| 37 | Cauchy | torchregress | 1 | 4 | 32319.6636 | - | - | - | - |
| 38 | sklearn_MLPRegressor | sklearn | 1 | 4 | 32703.2715 | - | - | - | - |
| 39 | Uncalibrated | torchregress | 1 | 4 | 32736.8225 | - | - | - | 0.0035 |
| 40 | VarianceTemperatureScaler | torchregress | 1 | 4 | 32736.8225 | - | - | - | 0.0037 |
| 41 | Tukey | torchregress | 1 | 4 | 32787.8408 | - | - | - | - |
| 42 | sklearn_GradientBoosting | sklearn | 1 | 4 | 32795.5988 | - | - | - | - |
| 43 | PseudoHuber | torchregress | 1 | 4 | 32865.8479 | - | - | - | - |
| 44 | AdaptiveRobust | torchregress | 1 | 4 | 32949.2055 | - | - | - | - |
| 45 | GaussianNLL+InputNoiseMC | torchregress | 1 | 4 | 33118.2908 | - | - | - | 0.0035 |
| 46 | LogCosh | torchregress | 1 | 4 | 33170.0374 | - | - | - | - |
| 47 | sklearn_BaggingRegressor | sklearn | 1 | 4 | 33197.9182 | - | - | - | - |
| 48 | Barron | torchregress | 1 | 4 | 33255.5041 | - | - | - | - |
| 49 | IsotonicMeanCalibrator | torchregress | 1 | 4 | 33287.6780 | - | - | - | 0.0036 |
| 50 | InputNoiseMC | torchregress | 1 | 4 | 33380.7509 | - | - | - | - |
| 51 | Huber | torchregress | 1 | 4 | 33495.2530 | - | - | - | - |
| 52 | sklearn_StackingRegressor | sklearn | 1 | 4 | 33513.9348 | - | - | - | - |
| 53 | MC_Dropout | torchregress | 1 | 4 | 33687.9350 | - | - | - | 0.0049 |
| 54 | sklearn_IsotonicRegression | sklearn | 1 | 4 | 33789.8649 | - | - | - | 0.0038 |
| 55 | MSE_WarmupMC | torchregress | 1 | 4 | 34182.5957 | - | - | - | - |
| 56 | EnsembleEIV | torchregress | 1 | 4 | 34568.9350 | - | - | - | - |
| 57 | sklearn_RandomForest | sklearn | 1 | 4 | 34762.5031 | - | - | - | - |
| 58 | sklearn_ExtraTrees | sklearn | 1 | 4 | 39124.6581 | - | - | - | - |
| 59 | LightGBM | LightGBM | 1 | 4 | 39895.7563 | - | - | - | - |
| 60 | sklearn_VotingRegressor | sklearn | 1 | 4 | 40012.0672 | - | - | - | - |
| 61 | sklearn_DecisionTree | sklearn | 1 | 4 | 45029.5881 | - | - | - | - |
| 62 | LatentNN | torchregress | 1 | 4 | 46170.5682 | - | - | - | - |
| 63 | sklearn_SVR | sklearn | 1 | 4 | 46210.5373 | - | - | - | - |
| 64 | sklearn_KNeighbors | sklearn | 1 | 4 | 46385.4742 | - | - | - | - |
| 65 | sklearn_Ridge | sklearn | 1 | 4 | 51891.9650 | - | - | - | - |
| 66 | OLS | torchregress | 1 | 4 | 51892.0453 | - | - | - | - |
| 67 | sklearn_LinearRegression | sklearn | 1 | 4 | 51892.0455 | - | - | - | - |
| 68 | GaussianNLL_Linear | torchregress | 1 | 4 | 51892.0873 | - | - | - | 0.0044 |
| 69 | MSE_Linear | torchregress | 1 | 4 | 51892.0875 | - | - | - | - |
| 70 | sklearn_GP | sklearn | 1 | 4 | 52077.8174 | - | - | - | 0.0046 |
| 71 | LogCosh_Linear | torchregress | 1 | 4 | 52782.7075 | - | - | - | - |
| 72 | Barron_Linear | torchregress | 1 | 4 | 52819.3752 | - | - | - | - |
| 73 | PseudoHuber_Linear | torchregress | 1 | 4 | 52819.3759 | - | - | - | - |
| 74 | Huber_Linear | torchregress | 1 | 4 | 52829.1433 | - | - | - | - |
| 75 | Tukey_Linear | torchregress | 1 | 4 | 52976.8778 | - | - | - | - |
| 76 | sklearn_ElasticNet | sklearn | 1 | 4 | 53993.6237 | - | - | - | - |
| 77 | statsmodels_RLM | statsmodels | 1 | 4 | 54109.0037 | - | - | - | - |
| 78 | sklearn_HuberRegressor | sklearn | 1 | 4 | 54438.1519 | - | - | - | - |
| 79 | MAE_Linear | torchregress | 1 | 4 | 54951.5308 | - | - | - | - |
| 80 | Quantile_Linear | torchregress | 1 | 4 | 54951.5480 | - | - | - | - |
| 81 | Charbonnier_Linear | torchregress | 1 | 4 | 54954.9352 | - | - | - | - |
| 82 | sklearn_QuantileRegressor | sklearn | 1 | 4 | 54968.1131 | - | - | - | - |
| 83 | Cauchy_Linear | torchregress | 1 | 4 | 55953.2801 | - | - | - | - |
| 84 | sklearn_BayesianRidge | sklearn | 2 | 8 | 56198.5162 | - | - | - | 0.0040 |
| 85 | sklearn_Lasso | sklearn | 1 | 4 | 56488.6511 | - | - | - | - |
| 86 | SIMEX | torchregress | 1 | 4 | 57315.0919 | - | - | - | - |
| 87 | BLR_Linear | torchregress | 1 | 4 | 60506.2738 | - | - | - | 0.0040 |
| 88 | sklearn_AdaBoost | sklearn | 1 | 4 | 62612.7426 | - | - | - | - |
| 89 | BayesianLinearHead | torchregress | 1 | 4 | 70656.6109 | - | - | - | 0.0076 |
| 90 | BoTorch_SingleTaskGP | BoTorch | 1 | 4 | 79695.8803 | - | - | - | 0.0042 |
| 91 | ODR | torchregress | 1 | 4 | 243431.6826 | - | - | - | - |
| 92 | TLS | numpy | 1 | 4 | 801963.3931 | - | - | - | - |
| 93 | TLS_Init+FunctionalEIV | torchregress | 1 | 4 | 1314486.3847 | - | - | - | - |
| 94 | AdaptiveRobust_Linear | torchregress | 1 | 4 | 21803742.0140 | - | - | - | - |
| 95 | NF_Ensemble_K20/bin_5 | torchregress | 1 | 3 | - | - | 0.8813 | 0.0308 | - |
| 96 | Split_CP/bin_5 | torchregress | 1 | 3 | - | - | 0.8790 | 0.0325 | - |
| 97 | NF_Ensemble_K10/bin_5 | torchregress | 1 | 3 | - | - | 0.8715 | 0.0394 | - |
| 98 | MDN_K3/bin_5 | torchregress | 1 | 3 | - | - | 0.8577 | 0.0423 | - |
| 99 | NF_Ensemble_K5/bin_5 | torchregress | 1 | 3 | - | - | 0.8624 | 0.0451 | - |
| 100 | GaussianNLL/bin_5 | torchregress | 1 | 3 | - | - | 0.8626 | 0.0495 | - |
| 101 | GaussianNLL/bin_0 | torchregress | 1 | 4 | - | - | 0.8466 | 0.0573 | - |
| 102 | Evidential/bin_9 | torchregress | 1 | 3 | - | - | 0.9585 | 0.0585 | - |
| 103 | NF_Ensemble_K2/bin_5 | torchregress | 1 | 3 | - | - | 0.8327 | 0.0673 | - |
| 104 | MDN_K3/bin_9 | torchregress | 1 | 3 | - | - | 0.8234 | 0.0766 | - |
| 105 | Evidential/bin_5 | torchregress | 1 | 3 | - | - | 0.9819 | 0.0819 | - |
| 106 | MDN_K3/bin_0 | torchregress | 1 | 4 | - | - | 0.8504 | 0.0824 | - |
| 107 | GaussianNLL/bin_9 | torchregress | 1 | 3 | - | - | 0.8287 | 0.0848 | - |
| 108 | Evidential/bin_0 | torchregress | 1 | 4 | - | - | 0.9370 | 0.0870 | - |
| 109 | NormalizingFlow/bin_5 | torchregress | 1 | 3 | - | - | 0.7635 | 0.1365 | - |
| 110 | NF_Ensemble_K10/bin_9 | torchregress | 1 | 3 | - | - | 0.7353 | 0.1769 | - |
| 111 | NF_Ensemble_K20/bin_9 | torchregress | 1 | 3 | - | - | 0.7379 | 0.1795 | - |
| 112 | NF_Ensemble_K5/bin_9 | torchregress | 1 | 3 | - | - | 0.7163 | 0.1837 | - |
| 113 | NF_Ensemble_K2/bin_9 | torchregress | 1 | 3 | - | - | 0.7059 | 0.1941 | - |
| 114 | Split_CP/bin_0 | torchregress | 1 | 4 | - | - | 0.7004 | 0.2045 | - |
| 115 | NormalizingFlow/bin_9 | torchregress | 1 | 3 | - | - | 0.6736 | 0.2264 | - |
| 116 | NF_Ensemble_K20/bin_0 | torchregress | 1 | 4 | - | - | 0.6584 | 0.2571 | - |
| 117 | NF_Ensemble_K10/bin_0 | torchregress | 1 | 4 | - | - | 0.6645 | 0.2682 | - |
| 118 | NF_Ensemble_K5/bin_0 | torchregress | 1 | 4 | - | - | 0.6467 | 0.2688 | - |
| 119 | NF_Ensemble_K2/bin_0 | torchregress | 1 | 4 | - | - | 0.6357 | 0.2798 | - |
| 120 | NormalizingFlow/bin_0 | torchregress | 1 | 4 | - | - | 0.6058 | 0.2942 | - |
| 121 | Split_CP/bin_9 | torchregress | 1 | 3 | - | - | 0.6073 | 0.2958 | - |
| 122 | Split_CP/dev_summary | torchregress | 1 | 4 | - | - | - | - | - |
| 123 | GaussianNLL/dev_summary | torchregress | 1 | 4 | - | - | - | - | - |
| 124 | NormalizingFlow/dev_summary | torchregress | 1 | 4 | - | - | - | - | - |
| 125 | NF_Ensemble_K2/dev_summary | torchregress | 1 | 4 | - | - | - | - | - |
| 126 | NF_Ensemble_K5/dev_summary | torchregress | 1 | 4 | - | - | - | - | - |
| 127 | NF_Ensemble_K10/dev_summary | torchregress | 1 | 4 | - | - | - | - | - |
| 128 | NF_Ensemble_K20/dev_summary | torchregress | 1 | 4 | - | - | - | - | - |
| 129 | MDN_K3/dev_summary | torchregress | 1 | 4 | - | - | - | - | - |
| 130 | Evidential/dev_summary | torchregress | 1 | 4 | - | - | - | - | - |
| 131 | normflows_NSF | normflows | 1 | 4 | - | - | - | - | - |
| 132 | Split_CP/bin_1 | torchregress | 1 | 3 | - | - | - | - | - |
| 133 | Split_CP/bin_2 | torchregress | 1 | 3 | - | - | - | - | - |
| 134 | Split_CP/bin_3 | torchregress | 1 | 3 | - | - | - | - | - |
| 135 | Split_CP/bin_4 | torchregress | 1 | 3 | - | - | - | - | - |
| 136 | Split_CP/bin_6 | torchregress | 1 | 3 | - | - | - | - | - |
| 137 | Split_CP/bin_7 | torchregress | 1 | 3 | - | - | - | - | - |
| 138 | Split_CP/bin_8 | torchregress | 1 | 3 | - | - | - | - | - |
| 139 | GaussianNLL/bin_1 | torchregress | 1 | 3 | - | - | - | - | - |
| 140 | GaussianNLL/bin_2 | torchregress | 1 | 3 | - | - | - | - | - |
| 141 | GaussianNLL/bin_3 | torchregress | 1 | 3 | - | - | - | - | - |
| 142 | GaussianNLL/bin_4 | torchregress | 1 | 3 | - | - | - | - | - |
| 143 | GaussianNLL/bin_6 | torchregress | 1 | 3 | - | - | - | - | - |
| 144 | GaussianNLL/bin_7 | torchregress | 1 | 3 | - | - | - | - | - |
| 145 | GaussianNLL/bin_8 | torchregress | 1 | 3 | - | - | - | - | - |
| 146 | NormalizingFlow/bin_1 | torchregress | 1 | 3 | - | - | - | - | - |
| 147 | NormalizingFlow/bin_2 | torchregress | 1 | 3 | - | - | - | - | - |
| 148 | NormalizingFlow/bin_3 | torchregress | 1 | 3 | - | - | - | - | - |
| 149 | NormalizingFlow/bin_4 | torchregress | 1 | 3 | - | - | - | - | - |
| 150 | NormalizingFlow/bin_6 | torchregress | 1 | 3 | - | - | - | - | - |
| 151 | NormalizingFlow/bin_7 | torchregress | 1 | 3 | - | - | - | - | - |
| 152 | NormalizingFlow/bin_8 | torchregress | 1 | 3 | - | - | - | - | - |
| 153 | NF_Ensemble_K2/bin_1 | torchregress | 1 | 3 | - | - | - | - | - |
| 154 | NF_Ensemble_K2/bin_2 | torchregress | 1 | 3 | - | - | - | - | - |
| 155 | NF_Ensemble_K2/bin_3 | torchregress | 1 | 3 | - | - | - | - | - |
| 156 | NF_Ensemble_K2/bin_4 | torchregress | 1 | 3 | - | - | - | - | - |
| 157 | NF_Ensemble_K2/bin_6 | torchregress | 1 | 3 | - | - | - | - | - |
| 158 | NF_Ensemble_K2/bin_7 | torchregress | 1 | 3 | - | - | - | - | - |
| 159 | NF_Ensemble_K2/bin_8 | torchregress | 1 | 3 | - | - | - | - | - |
| 160 | NF_Ensemble_K5/bin_1 | torchregress | 1 | 3 | - | - | - | - | - |
| 161 | NF_Ensemble_K5/bin_2 | torchregress | 1 | 3 | - | - | - | - | - |
| 162 | NF_Ensemble_K5/bin_3 | torchregress | 1 | 3 | - | - | - | - | - |
| 163 | NF_Ensemble_K5/bin_4 | torchregress | 1 | 3 | - | - | - | - | - |
| 164 | NF_Ensemble_K5/bin_6 | torchregress | 1 | 3 | - | - | - | - | - |
| 165 | NF_Ensemble_K5/bin_7 | torchregress | 1 | 3 | - | - | - | - | - |
| 166 | NF_Ensemble_K5/bin_8 | torchregress | 1 | 3 | - | - | - | - | - |
| 167 | NF_Ensemble_K10/bin_1 | torchregress | 1 | 3 | - | - | - | - | - |
| 168 | NF_Ensemble_K10/bin_2 | torchregress | 1 | 3 | - | - | - | - | - |
| 169 | NF_Ensemble_K10/bin_3 | torchregress | 1 | 3 | - | - | - | - | - |
| 170 | NF_Ensemble_K10/bin_4 | torchregress | 1 | 3 | - | - | - | - | - |
| 171 | NF_Ensemble_K10/bin_6 | torchregress | 1 | 3 | - | - | - | - | - |
| 172 | NF_Ensemble_K10/bin_7 | torchregress | 1 | 3 | - | - | - | - | - |
| 173 | NF_Ensemble_K10/bin_8 | torchregress | 1 | 3 | - | - | - | - | - |
| 174 | NF_Ensemble_K20/bin_1 | torchregress | 1 | 3 | - | - | - | - | - |
| 175 | NF_Ensemble_K20/bin_2 | torchregress | 1 | 3 | - | - | - | - | - |
| 176 | NF_Ensemble_K20/bin_3 | torchregress | 1 | 3 | - | - | - | - | - |
| 177 | NF_Ensemble_K20/bin_4 | torchregress | 1 | 3 | - | - | - | - | - |
| 178 | NF_Ensemble_K20/bin_6 | torchregress | 1 | 3 | - | - | - | - | - |
| 179 | NF_Ensemble_K20/bin_7 | torchregress | 1 | 3 | - | - | - | - | - |
| 180 | NF_Ensemble_K20/bin_8 | torchregress | 1 | 3 | - | - | - | - | - |
| 181 | MDN_K3/bin_1 | torchregress | 1 | 3 | - | - | - | - | - |
| 182 | MDN_K3/bin_2 | torchregress | 1 | 3 | - | - | - | - | - |
| 183 | MDN_K3/bin_3 | torchregress | 1 | 3 | - | - | - | - | - |
| 184 | MDN_K3/bin_4 | torchregress | 1 | 3 | - | - | - | - | - |
| 185 | MDN_K3/bin_6 | torchregress | 1 | 3 | - | - | - | - | - |
| 186 | MDN_K3/bin_7 | torchregress | 1 | 3 | - | - | - | - | - |
| 187 | MDN_K3/bin_8 | torchregress | 1 | 3 | - | - | - | - | - |
| 188 | Evidential/bin_1 | torchregress | 1 | 3 | - | - | - | - | - |
| 189 | Evidential/bin_2 | torchregress | 1 | 3 | - | - | - | - | - |
| 190 | Evidential/bin_3 | torchregress | 1 | 3 | - | - | - | - | - |
| 191 | Evidential/bin_4 | torchregress | 1 | 3 | - | - | - | - | - |
| 192 | Evidential/bin_6 | torchregress | 1 | 3 | - | - | - | - | - |
| 193 | Evidential/bin_7 | torchregress | 1 | 3 | - | - | - | - | - |
| 194 | Evidential/bin_8 | torchregress | 1 | 3 | - | - | - | - | - |

## Per-Suite Breakdown

### Suite: `bayesian_linear`

#### Dataset: `california_housing`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| BayesianLinearHead | torchregress | 0.8299 | - | - | 0.0066 |
| BoTorch_SingleTaskGP | BoTorch | 0.9044 | - | - | 0.0027 |
| BLR_Linear | torchregress | 1.4212 | - | - | 0.0041 |
| sklearn_BayesianRidge | sklearn | 1.4212 | - | - | 0.0041 |

#### Dataset: `diabetes`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| BLR_Linear | torchregress | 52.8434 | - | - | 0.0062 |
| sklearn_BayesianRidge | sklearn | 52.8434 | - | - | 0.0062 |
| BayesianLinearHead | torchregress | 56.7920 | - | - | 0.0139 |
| BoTorch_SingleTaskGP | BoTorch | 69.3720 | - | - | 0.0074 |

#### Dataset: `house_sales`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| BLR_Linear | torchregress | 241970.4050 | - | - | 0.0036 |
| sklearn_BayesianRidge | sklearn | 241970.4167 | - | - | 0.0036 |
| BayesianLinearHead | torchregress | 282568.3252 | - | - | 0.0053 |
| BoTorch_SingleTaskGP | BoTorch | 318712.6770 | - | - | 0.0035 |

#### Dataset: `synthetic_hetero`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| BLR_Linear | torchregress | 0.4255 | - | - | 0.0018 |
| sklearn_BayesianRidge | sklearn | 0.4255 | - | - | 0.0018 |
| BayesianLinearHead | torchregress | 0.4963 | - | - | 0.0046 |
| BoTorch_SingleTaskGP | BoTorch | 0.5678 | - | - | 0.0031 |

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
| Split_CP/bin_0 | torchregress | - | - | 0.8641 | - |
| Split_CP/bin_1 | torchregress | - | - | - | - |
| Split_CP/bin_2 | torchregress | - | - | - | - |
| Split_CP/bin_3 | torchregress | - | - | - | - |
| Split_CP/bin_4 | torchregress | - | - | - | - |
| Split_CP/bin_5 | torchregress | - | - | 0.8955 | - |
| Split_CP/bin_6 | torchregress | - | - | - | - |
| Split_CP/bin_7 | torchregress | - | - | - | - |
| Split_CP/bin_8 | torchregress | - | - | - | - |
| Split_CP/bin_9 | torchregress | - | - | 0.9047 | - |
| Split_CP/dev_summary | torchregress | - | - | - | - |
| GaussianNLL/bin_0 | torchregress | - | - | 0.9078 | - |
| GaussianNLL/bin_1 | torchregress | - | - | - | - |
| GaussianNLL/bin_2 | torchregress | - | - | - | - |
| GaussianNLL/bin_3 | torchregress | - | - | - | - |
| GaussianNLL/bin_4 | torchregress | - | - | - | - |
| GaussianNLL/bin_5 | torchregress | - | - | 0.9181 | - |
| GaussianNLL/bin_6 | torchregress | - | - | - | - |
| GaussianNLL/bin_7 | torchregress | - | - | - | - |
| GaussianNLL/bin_8 | torchregress | - | - | - | - |
| GaussianNLL/bin_9 | torchregress | - | - | 0.9202 | - |
| GaussianNLL/dev_summary | torchregress | - | - | - | - |
| NormalizingFlow/bin_0 | torchregress | - | - | 0.8350 | - |
| NormalizingFlow/bin_1 | torchregress | - | - | - | - |
| NormalizingFlow/bin_2 | torchregress | - | - | - | - |
| NormalizingFlow/bin_3 | torchregress | - | - | - | - |
| NormalizingFlow/bin_4 | torchregress | - | - | - | - |
| NormalizingFlow/bin_5 | torchregress | - | - | 0.8833 | - |
| NormalizingFlow/bin_6 | torchregress | - | - | - | - |
| NormalizingFlow/bin_7 | torchregress | - | - | - | - |
| NormalizingFlow/bin_8 | torchregress | - | - | - | - |
| NormalizingFlow/bin_9 | torchregress | - | - | 0.8405 | - |
| NormalizingFlow/dev_summary | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_0 | torchregress | - | - | 0.8374 | - |
| NF_Ensemble_K2/bin_1 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_2 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_3 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_4 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_5 | torchregress | - | - | 0.8902 | - |
| NF_Ensemble_K2/bin_6 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_7 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_8 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_9 | torchregress | - | - | 0.8658 | - |
| NF_Ensemble_K2/dev_summary | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_0 | torchregress | - | - | 0.8738 | - |
| NF_Ensemble_K5/bin_1 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_2 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_3 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_4 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_5 | torchregress | - | - | 0.9111 | - |
| NF_Ensemble_K5/bin_6 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_7 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_8 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_9 | torchregress | - | - | 0.8969 | - |
| NF_Ensemble_K5/dev_summary | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_0 | torchregress | - | - | 0.8956 | - |
| NF_Ensemble_K10/bin_1 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_2 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_3 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_4 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_5 | torchregress | - | - | 0.9164 | - |
| NF_Ensemble_K10/bin_6 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_7 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_8 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_9 | torchregress | - | - | 0.9183 | - |
| NF_Ensemble_K10/dev_summary | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_0 | torchregress | - | - | 0.8981 | - |
| NF_Ensemble_K20/bin_1 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_2 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_3 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_4 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_5 | torchregress | - | - | 0.9181 | - |
| NF_Ensemble_K20/bin_6 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_7 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_8 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_9 | torchregress | - | - | 0.9261 | - |
| NF_Ensemble_K20/dev_summary | torchregress | - | - | - | - |
| MDN_K3/bin_0 | torchregress | - | - | 0.8617 | - |
| MDN_K3/bin_1 | torchregress | - | - | - | - |
| MDN_K3/bin_2 | torchregress | - | - | - | - |
| MDN_K3/bin_3 | torchregress | - | - | - | - |
| MDN_K3/bin_4 | torchregress | - | - | - | - |
| MDN_K3/bin_5 | torchregress | - | - | 0.8833 | - |
| MDN_K3/bin_6 | torchregress | - | - | - | - |
| MDN_K3/bin_7 | torchregress | - | - | - | - |
| MDN_K3/bin_8 | torchregress | - | - | - | - |
| MDN_K3/bin_9 | torchregress | - | - | 0.8949 | - |
| MDN_K3/dev_summary | torchregress | - | - | - | - |
| Evidential/bin_0 | torchregress | - | - | 0.9782 | - |
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
| Split_CP/bin_5 | torchregress | - | - | 0.8243 | - |
| Split_CP/bin_6 | torchregress | - | - | 0.9324 | - |
| Split_CP/bin_7 | torchregress | - | - | 0.4500 | - |
| Split_CP/bin_8 | torchregress | - | - | - | - |
| Split_CP/bin_9 | torchregress | - | - | 0.1316 | - |
| Split_CP/dev_summary | torchregress | - | - | - | - |
| GaussianNLL/bin_0 | torchregress | - | - | 0.8000 | - |
| GaussianNLL/bin_1 | torchregress | - | - | - | - |
| GaussianNLL/bin_2 | torchregress | - | - | - | - |
| GaussianNLL/bin_3 | torchregress | - | - | - | - |
| GaussianNLL/bin_4 | torchregress | - | - | - | - |
| GaussianNLL/bin_5 | torchregress | - | - | 0.8298 | - |
| GaussianNLL/bin_6 | torchregress | - | - | 0.8196 | - |
| GaussianNLL/bin_7 | torchregress | - | - | 0.8375 | - |
| GaussianNLL/bin_8 | torchregress | - | - | - | - |
| GaussianNLL/bin_9 | torchregress | - | - | 0.8158 | - |
| GaussianNLL/dev_summary | torchregress | - | - | - | - |
| NormalizingFlow/bin_0 | torchregress | - | - | 0.0000 | - |
| NormalizingFlow/bin_1 | torchregress | - | - | - | - |
| NormalizingFlow/bin_2 | torchregress | - | - | - | - |
| NormalizingFlow/bin_3 | torchregress | - | - | - | - |
| NormalizingFlow/bin_4 | torchregress | - | - | - | - |
| NormalizingFlow/bin_5 | torchregress | - | - | 0.7167 | - |
| NormalizingFlow/bin_6 | torchregress | - | - | 0.7311 | - |
| NormalizingFlow/bin_7 | torchregress | - | - | 0.6625 | - |
| NormalizingFlow/bin_8 | torchregress | - | - | - | - |
| NormalizingFlow/bin_9 | torchregress | - | - | 0.3947 | - |
| NormalizingFlow/dev_summary | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_0 | torchregress | - | - | 0.0000 | - |
| NF_Ensemble_K2/bin_1 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_2 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_3 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_4 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_5 | torchregress | - | - | 0.8178 | - |
| NF_Ensemble_K2/bin_6 | torchregress | - | - | 0.8148 | - |
| NF_Ensemble_K2/bin_7 | torchregress | - | - | 0.7750 | - |
| NF_Ensemble_K2/bin_8 | torchregress | - | - | - | - |
| NF_Ensemble_K2/bin_9 | torchregress | - | - | 0.3947 | - |
| NF_Ensemble_K2/dev_summary | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_0 | torchregress | - | - | 0.0000 | - |
| NF_Ensemble_K5/bin_1 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_2 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_3 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_4 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_5 | torchregress | - | - | 0.8657 | - |
| NF_Ensemble_K5/bin_6 | torchregress | - | - | 0.8587 | - |
| NF_Ensemble_K5/bin_7 | torchregress | - | - | 0.8375 | - |
| NF_Ensemble_K5/bin_8 | torchregress | - | - | - | - |
| NF_Ensemble_K5/bin_9 | torchregress | - | - | 0.3947 | - |
| NF_Ensemble_K5/dev_summary | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_0 | torchregress | - | - | 0.0000 | - |
| NF_Ensemble_K10/bin_1 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_2 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_3 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_4 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_5 | torchregress | - | - | 0.8546 | - |
| NF_Ensemble_K10/bin_6 | torchregress | - | - | 0.8580 | - |
| NF_Ensemble_K10/bin_7 | torchregress | - | - | 0.8375 | - |
| NF_Ensemble_K10/bin_8 | torchregress | - | - | - | - |
| NF_Ensemble_K10/bin_9 | torchregress | - | - | 0.3947 | - |
| NF_Ensemble_K10/dev_summary | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_0 | torchregress | - | - | 0.0000 | - |
| NF_Ensemble_K20/bin_1 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_2 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_3 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_4 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_5 | torchregress | - | - | 0.8638 | - |
| NF_Ensemble_K20/bin_6 | torchregress | - | - | 0.8749 | - |
| NF_Ensemble_K20/bin_7 | torchregress | - | - | 0.8625 | - |
| NF_Ensemble_K20/bin_8 | torchregress | - | - | - | - |
| NF_Ensemble_K20/bin_9 | torchregress | - | - | 0.3947 | - |
| NF_Ensemble_K20/dev_summary | torchregress | - | - | - | - |
| MDN_K3/bin_0 | torchregress | - | - | 0.8000 | - |
| MDN_K3/bin_1 | torchregress | - | - | - | - |
| MDN_K3/bin_2 | torchregress | - | - | - | - |
| MDN_K3/bin_3 | torchregress | - | - | - | - |
| MDN_K3/bin_4 | torchregress | - | - | - | - |
| MDN_K3/bin_5 | torchregress | - | - | 0.8316 | - |
| MDN_K3/bin_6 | torchregress | - | - | 0.7952 | - |
| MDN_K3/bin_7 | torchregress | - | - | 0.8500 | - |
| MDN_K3/bin_8 | torchregress | - | - | - | - |
| MDN_K3/bin_9 | torchregress | - | - | 0.7895 | - |
| MDN_K3/dev_summary | torchregress | - | - | - | - |
| Evidential/bin_0 | torchregress | - | - | 0.8000 | - |
| Evidential/bin_1 | torchregress | - | - | - | - |
| Evidential/bin_2 | torchregress | - | - | - | - |
| Evidential/bin_3 | torchregress | - | - | - | - |
| Evidential/bin_4 | torchregress | - | - | - | - |
| Evidential/bin_5 | torchregress | - | - | 0.9761 | - |
| Evidential/bin_6 | torchregress | - | - | 0.9889 | - |
| Evidential/bin_7 | torchregress | - | - | 0.9875 | - |
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
| EnsembleEIV | torchregress | 0.6102 | - | - | - |
| FunctionalEIV | torchregress | 0.6127 | - | - | - |
| StructuralEIV | torchregress | 0.6127 | - | - | - |
| MSE_WarmupMC | torchregress | 0.6147 | - | - | - |
| GaussianNLL+InputNoiseMC | torchregress | 0.6285 | - | - | 0.0049 |
| InputNoiseMC | torchregress | 0.6372 | - | - | - |
| FunctionalEIV_MC | torchregress | 0.6666 | - | - | - |
| ODR | torchregress | 0.7410 | - | - | - |
| LatentNN | torchregress | 0.7626 | - | - | - |
| SIMEX | torchregress | 1.0319 | - | - | - |
| SplitCP+FunctionalEIV | torchregress+CP | 0.6102 | 2.8973 | 0.9000 | 0.0057 |
| SplitCP+MSE_WarmupMC | torchregress+CP | 0.6181 | 2.9421 | 0.9053 | 0.0051 |
| SplitCP+InputNoiseMC | torchregress+CP | 0.6351 | 2.9699 | 0.8793 | 0.0050 |
| TLS_Init+FunctionalEIV | torchregress | 16.0207 | - | - | - |
| TLS | numpy | 16.0558 | - | - | - |

#### Dataset: `diabetes`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| FunctionalEIV | torchregress | 52.6244 | - | - | - |
| StructuralEIV | torchregress | 52.6244 | - | - | - |
| GaussianNLL+InputNoiseMC | torchregress | 53.0504 | - | - | 0.0040 |
| InputNoiseMC | torchregress | 53.3378 | - | - | - |
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
| FunctionalEIV | torchregress | 128024.4635 | - | - | - |
| StructuralEIV | torchregress | 128024.4635 | - | - | - |
| MSE_NoisyPredict | torchregress | 128656.2112 | - | - | - |
| MSE | torchregress | 129082.1319 | - | - | - |
| FunctionalEIV_MC | torchregress | 129200.2064 | - | - | - |
| GaussianNLL+InputNoiseMC | torchregress | 132419.1100 | - | - | 0.0022 |
| InputNoiseMC | torchregress | 133468.6483 | - | - | - |
| MSE_WarmupMC | torchregress | 136661.8534 | - | - | - |
| EnsembleEIV | torchregress | 138220.9664 | - | - | - |
| LatentNN | torchregress | 184619.9864 | - | - | - |
| SIMEX | torchregress | 229201.1371 | - | - | - |
| SplitCP+MSE_WarmupMC | torchregress+CP | 133578.5779 | 629127.2676 | 0.9052 | 0.0064 |
| SplitCP+FunctionalEIV | torchregress+CP | 135008.4153 | 640872.4604 | 0.9043 | 0.0088 |
| SplitCP+InputNoiseMC | torchregress+CP | 139724.6508 | 653099.9861 | 0.8985 | 0.0067 |
| ODR | torchregress | 973668.1069 | - | - | - |
| TLS | numpy | 3207387.5493 | - | - | - |
| TLS_Init+FunctionalEIV | torchregress | 5257479.6311 | - | - | - |

#### Dataset: `synthetic_hetero`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| GaussianNLL+InputNoiseMC | torchregress | 0.3742 | - | - | 0.0029 |
| MSE_NoisyPredict | torchregress | 0.3757 | - | - | - |
| MSE | torchregress | 0.3793 | - | - | - |
| InputNoiseMC | torchregress | 0.3803 | - | - | - |
| FunctionalEIV | torchregress | 0.3823 | - | - | - |
| StructuralEIV | torchregress | 0.3823 | - | - | - |
| EnsembleEIV | torchregress | 0.3840 | - | - | - |
| TLS_Init+FunctionalEIV | torchregress | 0.3864 | - | - | - |
| TLS | numpy | 0.4000 | - | - | - |
| LatentNN | torchregress | 0.4050 | - | - | - |
| FunctionalEIV_MC | torchregress | 0.4066 | - | - | - |
| ODR | torchregress | 0.4168 | - | - | - |
| MSE_WarmupMC | torchregress | 0.4216 | - | - | - |
| SIMEX | torchregress | 0.6709 | - | - | - |
| SplitCP+InputNoiseMC | torchregress+CP | 0.3812 | 1.6669 | 0.9050 | 0.0067 |
| SplitCP+FunctionalEIV | torchregress+CP | 0.3860 | 1.6712 | 0.9083 | 0.0068 |
| SplitCP+MSE_WarmupMC | torchregress+CP | 0.4160 | 1.7731 | 0.8983 | 0.0082 |

### Suite: `ensembles`

#### Dataset: `california_housing`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| sklearn_StackingRegressor | sklearn | 0.5689 | - | - | - |
| DeepEnsemble_GaussianNLL | torchregress | 0.5718 | - | - | 0.0045 |
| DeepEnsemble_MSE | torchregress | 0.5769 | - | - | - |
| sklearn_BaggingRegressor | sklearn | 0.5842 | - | - | - |
| sklearn_VotingRegressor | sklearn | 0.6055 | - | - | - |
| DeepEnsemble_MultiQuantile | torchregress | 0.5844 | 2.4374 | 0.8107 | 0.0078 |

#### Dataset: `diabetes`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| sklearn_VotingRegressor | sklearn | 51.5670 | - | - | - |
| sklearn_StackingRegressor | sklearn | 51.7277 | - | - | - |
| DeepEnsemble_MSE | torchregress | 52.3985 | - | - | - |
| DeepEnsemble_GaussianNLL | torchregress | 52.7882 | - | - | 0.0061 |
| sklearn_BaggingRegressor | sklearn | 53.9668 | - | - | - |
| DeepEnsemble_MultiQuantile | torchregress | 52.0754 | 238.0689 | 0.7744 | 0.0084 |

#### Dataset: `house_sales`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| DeepEnsemble_GaussianNLL | torchregress | 118795.2176 | - | - | 0.0019 |
| DeepEnsemble_MSE | torchregress | 121469.2433 | - | - | - |
| sklearn_BaggingRegressor | sklearn | 132736.7326 | - | - | - |
| sklearn_StackingRegressor | sklearn | 134003.0696 | - | - | - |
| sklearn_VotingRegressor | sklearn | 159995.7132 | - | - | - |
| DeepEnsemble_MultiQuantile | torchregress | 116329.6614 | 442757.5551 | 0.7725 | 0.0098 |

#### Dataset: `synthetic_hetero`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| DeepEnsemble_GaussianNLL | torchregress | 0.3676 | - | - | 0.0027 |
| DeepEnsemble_MSE | torchregress | 0.3701 | - | - | - |
| sklearn_StackingRegressor | sklearn | 0.3732 | - | - | - |
| sklearn_VotingRegressor | sklearn | 0.3832 | - | - | - |
| sklearn_BaggingRegressor | sklearn | 0.3894 | - | - | - |
| DeepEnsemble_MultiQuantile | torchregress | 0.3705 | 1.6319 | 0.7783 | 0.0108 |

### Suite: `foundation_model`

#### Dataset: `california_housing`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| TabICL | tabicl | 0.4738 | - | - | - |
| GaussianNLL | torchregress | 0.5915 | - | - | 0.0040 |
| MDN | torchregress | 0.6058 | - | - | 0.0048 |

#### Dataset: `diabetes`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| TabICL | tabicl | 51.3598 | - | - | - |
| MDN | torchregress | 51.9826 | - | - | 0.0061 |
| GaussianNLL | torchregress | 53.0771 | - | - | 0.0062 |

#### Dataset: `house_sales`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| TabICL | tabicl | 111838.8437 | - | - | - |
| GaussianNLL | torchregress | 123397.3407 | - | - | 0.0021 |
| MDN | torchregress | 126175.0088 | - | - | 0.0024 |

#### Dataset: `synthetic_hetero`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| TabICL | tabicl | 0.3540 | - | - | - |
| MDN | torchregress | 0.3718 | - | - | 0.0044 |
| GaussianNLL | torchregress | 0.3823 | - | - | 0.0037 |

### Suite: `ordinal`

#### Dataset: `california_housing`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| MSE | torchregress | 0.8148 | - | - | - |
| statsmodels_OrderedModel | statsmodels | 1344.1194 | - | - | - |

#### Dataset: `diabetes`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| MSE | torchregress | 0.8474 | - | - | - |
| statsmodels_OrderedModel | statsmodels | 97.3392 | - | - | - |

#### Dataset: `house_sales`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| MSE | torchregress | 0.8386 | - | - | - |
| statsmodels_OrderedModel | statsmodels | 1525.9789 | - | - | - |

#### Dataset: `synthetic_hetero`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| MSE | torchregress | 0.8387 | - | - | - |
| statsmodels_OrderedModel | statsmodels | 800.4785 | - | - | - |

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
| Split+MLP | torchregress | 132367.9050 | 629407.3619 | 0.9007 | - |
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

### Suite: `robust_regression`

#### Dataset: `california_housing`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| Cauchy | torchregress | 0.5764 | - | - | - |
| Charbonnier | torchregress | 0.5775 | - | - | - |
| LightGBM | LightGBM | 0.5849 | - | - | - |
| Barron | torchregress | 0.5864 | - | - | - |
| PseudoHuber | torchregress | 0.5864 | - | - | - |
| Huber | torchregress | 0.5867 | - | - | - |
| LogCosh | torchregress | 0.5888 | - | - | - |
| MSE | torchregress | 0.5894 | - | - | - |
| AdaptiveRobust | torchregress | 0.5929 | - | - | - |
| Tukey | torchregress | 0.5973 | - | - | - |
| MSE_Linear | torchregress | 0.7984 | - | - | - |
| sklearn_QuantileRegressor | sklearn | 0.8243 | - | - | - |
| Huber_Linear | torchregress | 0.8248 | - | - | - |
| Quantile_Linear | torchregress | 0.8256 | - | - | - |
| MAE_Linear | torchregress | 0.8257 | - | - | - |
| statsmodels_RLM | statsmodels | 0.8260 | - | - | - |
| PseudoHuber_Linear | torchregress | 0.8260 | - | - | - |
| Barron_Linear | torchregress | 0.8260 | - | - | - |
| Charbonnier_Linear | torchregress | 0.8261 | - | - | - |
| LogCosh_Linear | torchregress | 0.8274 | - | - | - |
| Cauchy_Linear | torchregress | 0.8284 | - | - | - |
| Tukey_Linear | torchregress | 0.8800 | - | - | - |
| AdaptiveRobust_Linear | torchregress | 8.1707 | - | - | - |

#### Dataset: `diabetes`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| LightGBM | LightGBM | 50.7366 | - | - | - |
| MSE_Linear | torchregress | 53.1203 | - | - | - |
| statsmodels_RLM | statsmodels | 53.1414 | - | - | - |
| Huber_Linear | torchregress | 53.1641 | - | - | - |
| Tukey_Linear | torchregress | 53.1768 | - | - | - |
| LogCosh_Linear | torchregress | 53.1947 | - | - | - |
| PseudoHuber_Linear | torchregress | 53.2043 | - | - | - |
| Barron_Linear | torchregress | 53.2050 | - | - | - |
| Cauchy_Linear | torchregress | 53.8418 | - | - | - |
| AdaptiveRobust | torchregress | 53.8744 | - | - | - |
| LogCosh | torchregress | 54.0157 | - | - | - |
| MSE | torchregress | 54.0167 | - | - | - |
| Tukey | torchregress | 54.0179 | - | - | - |
| Barron | torchregress | 54.0853 | - | - | - |
| PseudoHuber | torchregress | 54.0853 | - | - | - |
| Huber | torchregress | 54.1256 | - | - | - |
| Charbonnier | torchregress | 54.2062 | - | - | - |
| MAE_Linear | torchregress | 54.6654 | - | - | - |
| Charbonnier_Linear | torchregress | 54.7204 | - | - | - |
| Quantile_Linear | torchregress | 54.7343 | - | - | - |
| sklearn_QuantileRegressor | sklearn | 54.7508 | - | - | - |
| Cauchy | torchregress | 55.2475 | - | - | - |
| AdaptiveRobust_Linear | torchregress | 36454.0163 | - | - | - |

#### Dataset: `house_sales`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| Charbonnier | torchregress | 123651.6408 | - | - | - |
| MSE | torchregress | 129082.1319 | - | - | - |
| Cauchy | torchregress | 129222.4426 | - | - | - |
| Tukey | torchregress | 131096.3631 | - | - | - |
| PseudoHuber | torchregress | 131408.3337 | - | - | - |
| AdaptiveRobust | torchregress | 131741.9697 | - | - | - |
| LogCosh | torchregress | 132625.1590 | - | - | - |
| Barron | torchregress | 132966.9584 | - | - | - |
| Huber | torchregress | 133925.9129 | - | - | - |
| LightGBM | LightGBM | 159531.3150 | - | - | - |
| MSE_Linear | torchregress | 207514.0465 | - | - | - |
| LogCosh_Linear | torchregress | 211076.4232 | - | - | - |
| Barron_Linear | torchregress | 211223.0849 | - | - | - |
| PseudoHuber_Linear | torchregress | 211223.0884 | - | - | - |
| Huber_Linear | torchregress | 211262.1995 | - | - | - |
| Tukey_Linear | torchregress | 211853.0694 | - | - | - |
| statsmodels_RLM | statsmodels | 216381.6624 | - | - | - |
| MAE_Linear | torchregress | 219750.2448 | - | - | - |
| Quantile_Linear | torchregress | 219750.2448 | - | - | - |
| Charbonnier_Linear | torchregress | 219763.8067 | - | - | - |
| sklearn_QuantileRegressor | sklearn | 219816.4901 | - | - | - |
| Cauchy_Linear | torchregress | 223758.0648 | - | - | - |
| AdaptiveRobust_Linear | torchregress | 87178182.9572 | - | - | - |

#### Dataset: `synthetic_hetero`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| MSE | torchregress | 0.3793 | - | - | - |
| Huber_Linear | torchregress | 0.3848 | - | - | - |
| LogCosh_Linear | torchregress | 0.3848 | - | - | - |
| Barron_Linear | torchregress | 0.3848 | - | - | - |
| PseudoHuber_Linear | torchregress | 0.3848 | - | - | - |
| Tukey_Linear | torchregress | 0.3849 | - | - | - |
| Tukey | torchregress | 0.3849 | - | - | - |
| statsmodels_RLM | statsmodels | 0.3850 | - | - | - |
| MSE_Linear | torchregress | 0.3850 | - | - | - |
| AdaptiveRobust | torchregress | 0.3852 | - | - | - |
| Cauchy_Linear | torchregress | 0.3853 | - | - | - |
| LogCosh | torchregress | 0.3862 | - | - | - |
| Barron | torchregress | 0.3863 | - | - | - |
| PseudoHuber | torchregress | 0.3863 | - | - | - |
| Huber | torchregress | 0.3870 | - | - | - |
| MAE_Linear | torchregress | 0.3874 | - | - | - |
| Quantile_Linear | torchregress | 0.3874 | - | - | - |
| sklearn_QuantileRegressor | sklearn | 0.3874 | - | - | - |
| Charbonnier_Linear | torchregress | 0.3874 | - | - | - |
| Cauchy | torchregress | 0.3880 | - | - | - |
| LightGBM | LightGBM | 0.3886 | - | - | - |
| Charbonnier | torchregress | 0.3895 | - | - | - |
| AdaptiveRobust_Linear | torchregress | 322.9119 | - | - | - |

### Suite: `sklearn_baselines`

#### Dataset: `california_housing`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| sklearn_HistGradientBoosting | sklearn | 0.5453 | - | - | - |
| sklearn_GradientBoosting | sklearn | 0.5569 | - | - | - |
| MSE | torchregress | 0.5894 | - | - | - |
| GaussianNLL | torchregress | 0.5915 | - | - | 0.0040 |
| sklearn_RandomForest | sklearn | 0.5940 | - | - | - |
| sklearn_SVR | sklearn | 0.6030 | - | - | - |
| sklearn_ExtraTrees | sklearn | 0.6323 | - | - | - |
| sklearn_MLPRegressor | sklearn | 0.6435 | - | - | - |
| sklearn_KNeighbors | sklearn | 0.6524 | - | - | - |
| sklearn_DecisionTree | sklearn | 0.7668 | - | - | - |
| sklearn_BayesianRidge | sklearn | 0.7980 | - | - | - |
| sklearn_Ridge | sklearn | 0.7982 | - | - | - |
| GaussianNLL_Linear | torchregress | 0.7984 | - | - | 0.0073 |
| sklearn_LinearRegression | sklearn | 0.7984 | - | - | - |
| OLS | torchregress | 0.7984 | - | - | - |
| sklearn_AdaBoost | sklearn | 0.8126 | - | - | - |
| sklearn_ElasticNet | sklearn | 0.8201 | - | - | - |
| sklearn_HuberRegressor | sklearn | 0.8295 | - | - | - |
| sklearn_Lasso | sklearn | 0.8461 | - | - | - |
| MultiQuantile | torchregress | 0.6030 | 2.5457 | 0.8007 | 0.0029 |

#### Dataset: `diabetes`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| sklearn_ExtraTrees | sklearn | 51.4646 | - | - | - |
| sklearn_SVR | sklearn | 52.5452 | - | - | - |
| sklearn_ElasticNet | sklearn | 52.9443 | - | - | - |
| sklearn_MLPRegressor | sklearn | 53.0149 | - | - | - |
| sklearn_BayesianRidge | sklearn | 53.0561 | - | - | - |
| GaussianNLL | torchregress | 53.0771 | - | - | 0.0062 |
| sklearn_Ridge | sklearn | 53.1035 | - | - | - |
| sklearn_LinearRegression | sklearn | 53.1202 | - | - | - |
| OLS | torchregress | 53.1202 | - | - | - |
| GaussianNLL_Linear | torchregress | 53.1203 | - | - | 0.0051 |
| sklearn_HuberRegressor | sklearn | 53.1864 | - | - | - |
| sklearn_AdaBoost | sklearn | 53.3482 | - | - | - |
| sklearn_RandomForest | sklearn | 53.5795 | - | - | - |
| sklearn_Lasso | sklearn | 53.6755 | - | - | - |
| MSE | torchregress | 54.0167 | - | - | - |
| sklearn_HistGradientBoosting | sklearn | 56.7700 | - | - | - |
| sklearn_KNeighbors | sklearn | 56.9648 | - | - | - |
| sklearn_GradientBoosting | sklearn | 58.5282 | - | - | - |
| sklearn_DecisionTree | sklearn | 72.1789 | - | - | - |
| MultiQuantile | torchregress | 52.3653 | 249.0081 | 0.7594 | 0.0058 |

#### Dataset: `house_sales`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| sklearn_HistGradientBoosting | sklearn | 122595.1173 | - | - | - |
| GaussianNLL | torchregress | 123397.3407 | - | - | 0.0021 |
| MSE | torchregress | 129082.1319 | - | - | - |
| sklearn_MLPRegressor | sklearn | 130759.0328 | - | - | - |
| sklearn_GradientBoosting | sklearn | 131122.9236 | - | - | - |
| sklearn_RandomForest | sklearn | 138995.4496 | - | - | - |
| sklearn_ExtraTrees | sklearn | 156446.1557 | - | - | - |
| sklearn_DecisionTree | sklearn | 180044.8483 | - | - | - |
| sklearn_SVR | sklearn | 184788.5861 | - | - | - |
| sklearn_KNeighbors | sklearn | 185483.7952 | - | - | - |
| sklearn_BayesianRidge | sklearn | 207508.7835 | - | - | - |
| sklearn_Ridge | sklearn | 207513.5735 | - | - | - |
| OLS | torchregress | 207513.8776 | - | - | - |
| sklearn_LinearRegression | sklearn | 207513.8784 | - | - | - |
| GaussianNLL_Linear | torchregress | 207514.0455 | - | - | 0.0025 |
| sklearn_ElasticNet | sklearn | 215920.3305 | - | - | - |
| sklearn_HuberRegressor | sklearn | 217698.2065 | - | - | - |
| sklearn_Lasso | sklearn | 225899.6659 | - | - | - |
| sklearn_AdaBoost | sklearn | 250396.3457 | - | - | - |
| MultiQuantile | torchregress | 125634.9492 | 495730.6113 | 0.7382 | 0.0072 |

#### Dataset: `synthetic_hetero`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| MSE | torchregress | 0.3793 | - | - | - |
| sklearn_ExtraTrees | sklearn | 0.3797 | - | - | - |
| GaussianNLL | torchregress | 0.3823 | - | - | 0.0037 |
| sklearn_HistGradientBoosting | sklearn | 0.3834 | - | - | - |
| sklearn_BayesianRidge | sklearn | 0.3850 | - | - | - |
| sklearn_Ridge | sklearn | 0.3850 | - | - | - |
| OLS | torchregress | 0.3850 | - | - | - |
| sklearn_LinearRegression | sklearn | 0.3850 | - | - | - |
| GaussianNLL_Linear | torchregress | 0.3850 | - | - | 0.0028 |
| sklearn_HuberRegressor | sklearn | 0.3850 | - | - | - |
| sklearn_GradientBoosting | sklearn | 0.3865 | - | - | - |
| sklearn_RandomForest | sklearn | 0.3893 | - | - | - |
| sklearn_MLPRegressor | sklearn | 0.3949 | - | - | - |
| sklearn_ElasticNet | sklearn | 0.3997 | - | - | - |
| sklearn_SVR | sklearn | 0.4148 | - | - | - |
| sklearn_Lasso | sklearn | 0.4170 | - | - | - |
| sklearn_AdaBoost | sklearn | 0.4637 | - | - | - |
| sklearn_KNeighbors | sklearn | 0.4843 | - | - | - |
| sklearn_DecisionTree | sklearn | 0.5582 | - | - | - |
| MultiQuantile | torchregress | 0.3778 | 1.7871 | 0.7517 | 0.0103 |

### Suite: `uq`

#### Dataset: `california_housing`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| DeepEnsemble_K10 | torchregress | 0.5778 | - | - | 0.0043 |
| GaussianNLL | torchregress | 0.5915 | - | - | 0.0040 |
| sklearn_GP | sklearn | 0.6004 | - | - | 0.0032 |
| MC_Dropout | torchregress | 0.6478 | - | - | 0.0047 |
| sklearn_QuantileGBR | sklearn | 0.5947 | 2.5518 | 0.8687 | 0.0038 |

#### Dataset: `diabetes`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| DeepEnsemble_K10 | torchregress | 51.4897 | - | - | 0.0084 |
| sklearn_GP | sklearn | 51.8194 | - | - | 0.0091 |
| GaussianNLL | torchregress | 53.0771 | - | - | 0.0062 |
| MC_Dropout | torchregress | 53.3870 | - | - | 0.0064 |
| sklearn_QuantileGBR | sklearn | 53.8981 | 209.2740 | 0.8571 | 0.0062 |

#### Dataset: `house_sales`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| DeepEnsemble_K10 | torchregress | 118236.1060 | - | - | 0.0017 |
| GaussianNLL | torchregress | 123397.3407 | - | - | 0.0021 |
| MC_Dropout | torchregress | 134697.2942 | - | - | 0.0032 |
| sklearn_GP | sklearn | 208258.4688 | - | - | 0.0033 |
| sklearn_QuantileGBR | sklearn | 148052.2507 | 473057.9105 | 0.8832 | 0.0040 |

#### Dataset: `synthetic_hetero`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| DeepEnsemble_K10 | torchregress | 0.3682 | - | - | 0.0033 |
| sklearn_GP | sklearn | 0.3808 | - | - | 0.0027 |
| GaussianNLL | torchregress | 0.3823 | - | - | 0.0037 |
| MC_Dropout | torchregress | 0.4110 | - | - | 0.0051 |
| sklearn_QuantileGBR | sklearn | 0.3739 | 1.7531 | 0.8383 | 0.0042 |
