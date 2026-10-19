# PZ core (8-head) results — all 16 p(z) datasets

| method | n | mean RMSE | median NLL | mean CRPS | mean coverage90 | mean sigma_NMAD | mean train_s |
|---|--:|--:|--:|--:|--:|--:|--:|
| MSE_MLP | 16 | 0.0535 | — | — | — | 0.0166 | 13.3399 |
| GaussianNLL_MLP | 16 | 0.0663 | -2.1445 | 0.0218 | 0.8997 | 0.0147 | 19.0216 |
| MDN_K3 | 16 | 0.0594 | -2.2065 | 0.0199 | 0.8897 | 0.0138 | 22.4023 |
| Quantile_MLP | 16 | 0.0548 | — | — | 0.8968 | 0.0145 | 18.2716 |
| NormalizingFlow | 16 | 0.0480 | -2.2209 | 0.0174 | 0.8670 | 0.0128 | 2129.0167 |
| TabM_MSE | 16 | 0.0500 | — | — | — | 0.0147 | 24.1381 |
| TabM_GaussianNLL | 16 | 0.0730 | -2.0874 | 0.0393 | 0.9211 | 0.0147 | 26.9917 |
| TabM_SplitCP | 16 | 0.0500 | — | — | 0.9030 | 0.0147 | 43.4417 |
