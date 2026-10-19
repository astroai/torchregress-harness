# PZ OOD / selection-blend shift

Model trained on task-1 cardinal 10yr; same model applied to tasksets 2-4 (held-out).

| method | scenario | RMSE | sigma_NMAD | coverage90 | coverage68 |
|---|---|--:|--:|--:|--:|
| GaussianNLL_MLP | ID_task1 | 0.0369 | 0.0107 | 0.8997 | 0.69295 |
| GaussianNLL_MLP | OOD_task2 | 0.1094 | 0.0145 | 0.86215 | 0.6454 |
| GaussianNLL_MLP | OOD_task3 | 0.2223 | 0.0108 | 0.89076425585614 | 0.6761574256644846 |
| GaussianNLL_MLP | OOD_task4 | 0.2372 | 0.0115 | 0.872429256223906 | 0.6561775166228545 |
| MDN_K3 | ID_task1 | 0.0311 | 0.0100 | 0.89195 | 0.67775 |
| MDN_K3 | OOD_task2 | 0.0868 | 0.0126 | 0.8643 | 0.6513 |
| MDN_K3 | OOD_task3 | 0.1440 | 0.0094 | 0.894076819938481 | 0.6864106002050635 |
| MDN_K3 | OOD_task4 | 0.1779 | 0.0106 | 0.8660893768362455 | 0.6538580485541983 |
| MSE_MLP | ID_task1 | 0.0255 | 0.0119 | — | — |
| MSE_MLP | OOD_task2 | 0.0641 | 0.0153 | — | — |
| MSE_MLP | OOD_task3 | 0.1128 | 0.0124 | — | — |
| MSE_MLP | OOD_task4 | 0.1288 | 0.0137 | — | — |
| SplitCP_MLP | ID_task1 | 0.0278 | 0.0125 | 0.89925 | — |
| SplitCP_MLP | OOD_task2 | 0.0758 | 0.0164 | 0.77485 | — |
| SplitCP_MLP | OOD_task3 | 0.1149 | 0.0156 | 0.7700922785708653 | — |
| SplitCP_MLP | OOD_task4 | 0.1307 | 0.0168 | 0.7374362146281119 | — |
| TabM_MSE | ID_task1 | 0.0236 | 0.0106 | — | — |
| TabM_MSE | OOD_task2 | 0.0539 | 0.0135 | — | — |
| TabM_MSE | OOD_task3 | 0.0925 | 0.0109 | — | — |
| TabM_MSE | OOD_task4 | 0.1056 | 0.0123 | — | — |
