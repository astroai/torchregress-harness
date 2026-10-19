# n(z) challenge results — taskset 1, all four simulations/scenarios

DDF truth training -> 1M-object WFD; reserved WFD truth is the evaluation target.  PPI uses a 5% labeled WFD slice + predictions on the rest; nominal 90% CIs.

| dataset | method | RMSE(z) | n(z) L1 | n(z) KS | mean-z bias | bin accuracy | bin purity | PPI mean-z coverage | PPI bin coverage | PPI bin error |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| nz_challenge_t1_cardinal_1yr | GaussianNLL_MLP | 0.0435 | 0.0457 | 0.0085 | -0.0012 | 0.9704 | 0.6495 | 1.0 | 1.0 | 0.0099 |
| nz_challenge_t1_cardinal_1yr | MDN_K3 | 0.0424 | 0.0512 | 0.0101 | 0.0008 | 0.9751 | 0.8717 | 1.0 | 1.0 | 0.0063 |
| nz_challenge_t1_cardinal_1yr | MSE_MLP | 0.0388 | 0.0965 | 0.0246 | -0.0137 | 0.9661 | 0.6876 | 1.0 | 1.0 | 0.0140 |
| nz_challenge_t1_cardinal_1yr | TabM_GaussianNLL | 0.0429 | 0.0422 | 0.0056 | -0.0006 | 0.9749 | 0.6507 | 1.0 | 1.0 | 0.0067 |
| nz_challenge_t1_cardinal_4yr | GaussianNLL_MLP | 0.0511 | 0.0432 | 0.0088 | -0.0052 | 0.9753 | 0.7281 | 1.0 | 1.0 | 0.0086 |
| nz_challenge_t1_cardinal_4yr | MDN_K3 | 0.0263 | 0.0343 | 0.0051 | -0.0004 | 0.9789 | 0.9217 | 1.0 | 1.0 | 0.0062 |
| nz_challenge_t1_cardinal_4yr | MSE_MLP | 0.0246 | 0.0888 | 0.0324 | -0.0224 | 0.9605 | 0.9272 | 1.0 | 0.5 | 0.0190 |
| nz_challenge_t1_cardinal_4yr | TabM_GaussianNLL | 0.0326 | 0.0414 | 0.0064 | -0.0020 | 0.9780 | 0.8291 | 1.0 | 1.0 | 0.0071 |
| nz_challenge_t1_flagship_1yr | GaussianNLL_MLP | 0.0401 | 0.0374 | 0.0082 | 0.0030 | 0.9677 | 0.9648 | 1.0 | 1.0 | 0.0035 |
| nz_challenge_t1_flagship_1yr | MDN_K3 | 0.0365 | 0.0375 | 0.0046 | 0.0007 | 0.9662 | 0.9716 | 1.0 | 1.0 | 0.0070 |
| nz_challenge_t1_flagship_1yr | MSE_MLP | 0.0356 | 0.0858 | 0.0268 | -0.0200 | 0.9547 | 0.9710 | 1.0 | 0.75 | 0.0175 |
| nz_challenge_t1_flagship_1yr | TabM_GaussianNLL | 0.0396 | 0.0392 | 0.0062 | 0.0018 | 0.9677 | 0.9591 | 1.0 | 1.0 | 0.0048 |
| nz_challenge_t1_flagship_4yr | GaussianNLL_MLP | 0.0387 | 0.0413 | 0.0073 | -0.0037 | 0.9747 | 0.9785 | 1.0 | 1.0 | 0.0076 |
| nz_challenge_t1_flagship_4yr | MDN_K3 | 0.0296 | 0.0369 | 0.0045 | -0.0021 | 0.9756 | 0.9765 | 1.0 | 1.0 | 0.0072 |
| nz_challenge_t1_flagship_4yr | MSE_MLP | 0.0244 | 0.0958 | 0.0327 | -0.0199 | 0.9611 | 0.9793 | 1.0 | 1.0 | 0.0172 |
| nz_challenge_t1_flagship_4yr | TabM_GaussianNLL | 0.0305 | 0.0353 | 0.0060 | 0.0012 | 0.9768 | 0.9617 | 1.0 | 1.0 | 0.0063 |
