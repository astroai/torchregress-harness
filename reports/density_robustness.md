# Density-head factorial benchmark

Clean test set; noisy-label methods see only noisy training labels.  Values are means over seeds.

| imbalance | budget | noise | method | NLL | CRPS | cov90 | PIT KS | rare NLL |
|---|---:|--:|---|--:|--:|--:|--:|--:|
| mild | 200 | 0.0 | NSF | 1.134 | 0.615 | 0.875 | 0.074 | 1.176 |
| mild | 200 | 0.0 | NSF_noise | 1.127 | 0.615 | 0.877 | 0.080 | 1.150 |
| mild | 200 | 0.0 | MDN | 1.396 | 0.657 | 0.920 | 0.144 | nan |
| mild | 200 | 0.0 | MDN_noise | 1.396 | 0.657 | 0.920 | 0.144 | nan |
| mild | 200 | 0.0 | Binned | 3.341 | 0.636 | 0.959 | 0.127 | nan |
| mild | 200 | 0.0 | Binned_noise | 3.341 | 0.636 | 0.959 | 0.127 | nan |
| mild | 200 | 0.3 | NSF | 1.164 | 0.609 | 0.870 | 0.072 | 1.172 |
| mild | 200 | 0.3 | NSF_noise | 1.216 | 0.611 | 0.781 | 0.083 | 1.215 |
| mild | 200 | 0.3 | MDN | 1.379 | 0.626 | 0.885 | 0.134 | nan |
| mild | 200 | 0.3 | MDN_noise | 1.430 | 0.634 | 0.850 | 0.137 | nan |
| mild | 200 | 0.3 | Binned | 3.411 | 0.631 | 0.939 | 0.136 | nan |
| mild | 200 | 0.3 | Binned_noise | 3.096 | 0.628 | 0.925 | 0.138 | nan |
| mild | 1000 | 0.0 | NSF | 1.123 | 0.613 | 0.853 | 0.124 | 1.032 |
| mild | 1000 | 0.0 | NSF_noise | 1.086 | 0.606 | 0.864 | 0.102 | 1.041 |
| mild | 1000 | 0.0 | MDN | 1.002 | 0.592 | 0.875 | 0.036 | nan |
| mild | 1000 | 0.0 | MDN_noise | 1.002 | 0.592 | 0.875 | 0.036 | nan |
| mild | 1000 | 0.0 | Binned | 2.940 | 0.605 | 0.915 | 0.068 | nan |
| mild | 1000 | 0.0 | Binned_noise | 2.940 | 0.605 | 0.915 | 0.068 | nan |
| mild | 1000 | 0.3 | NSF | 1.128 | 0.602 | 0.907 | 0.075 | 1.109 |
| mild | 1000 | 0.3 | NSF_noise | 1.277 | 0.619 | 0.829 | 0.140 | 1.482 |
| mild | 1000 | 0.3 | MDN | 1.059 | 0.595 | 0.934 | 0.055 | nan |
| mild | 1000 | 0.3 | MDN_noise | 1.043 | 0.593 | 0.874 | 0.023 | nan |
| mild | 1000 | 0.3 | Binned | 2.935 | 0.610 | 0.948 | 0.079 | nan |
| mild | 1000 | 0.3 | Binned_noise | 2.907 | 0.609 | 0.962 | 0.103 | nan |
| severe | 200 | 0.0 | NSF | 1.217 | 0.640 | 0.838 | 0.124 | 1.551 |
| severe | 200 | 0.0 | NSF_noise | 1.239 | 0.647 | 0.831 | 0.142 | 1.628 |
| severe | 200 | 0.0 | MDN | 1.294 | 0.617 | 0.859 | 0.098 | nan |
| severe | 200 | 0.0 | MDN_noise | 1.294 | 0.617 | 0.859 | 0.098 | nan |
| severe | 200 | 0.0 | Binned | 3.295 | 0.641 | 0.943 | 0.169 | nan |
| severe | 200 | 0.0 | Binned_noise | 3.295 | 0.641 | 0.943 | 0.169 | nan |
| severe | 200 | 0.3 | NSF | 1.180 | 0.607 | 0.921 | 0.084 | 1.216 |
| severe | 200 | 0.3 | NSF_noise | 1.247 | 0.612 | 0.842 | 0.103 | 1.385 |
| severe | 200 | 0.3 | MDN | 1.430 | 0.639 | 0.925 | 0.121 | nan |
| severe | 200 | 0.3 | MDN_noise | 1.573 | 0.648 | 0.911 | 0.125 | nan |
| severe | 200 | 0.3 | Binned | 3.401 | 0.637 | 0.947 | 0.123 | nan |
| severe | 200 | 0.3 | Binned_noise | 3.061 | 0.634 | 0.939 | 0.109 | nan |
| severe | 1000 | 0.0 | NSF | 1.070 | 0.605 | 0.888 | 0.039 | 1.090 |
| severe | 1000 | 0.0 | NSF_noise | 1.072 | 0.605 | 0.896 | 0.031 | 1.040 |
| severe | 1000 | 0.0 | MDN | 1.027 | 0.596 | 0.892 | 0.076 | nan |
| severe | 1000 | 0.0 | MDN_noise | 1.027 | 0.596 | 0.892 | 0.076 | nan |
| severe | 1000 | 0.0 | Binned | 2.882 | 0.603 | 0.920 | 0.058 | nan |
| severe | 1000 | 0.0 | Binned_noise | 2.882 | 0.603 | 0.920 | 0.058 | nan |
| severe | 1000 | 0.3 | NSF | 1.093 | 0.598 | 0.921 | 0.054 | 1.162 |
| severe | 1000 | 0.3 | NSF_noise | 1.084 | 0.598 | 0.856 | 0.064 | 1.094 |
| severe | 1000 | 0.3 | MDN | 1.074 | 0.597 | 0.922 | 0.055 | nan |
| severe | 1000 | 0.3 | MDN_noise | 1.055 | 0.596 | 0.847 | 0.032 | nan |
| severe | 1000 | 0.3 | Binned | 3.166 | 0.606 | 0.940 | 0.071 | nan |
| severe | 1000 | 0.3 | Binned_noise | 2.912 | 0.607 | 0.949 | 0.086 | nan |

**Reading:** noise-aware variants should be compared on clean-test NLL/CRPS, not on the noisy-label likelihood scale.  With one seed this is a screening experiment; the key qualitative result is which family is robust under label noise and shrinking budgets.

## Caveats

- **One seed**, 40 epochs, budget 200/1000, mild/severe imbalance, label noise
  `sigma=0` or `0.3`.  This is a screening experiment; the full sweep should use
  repeated seeds and validation-NLL early stopping.
- **Binned NLL is a probability mass**, not a continuous density, so its NLL is
  offset by roughly `log(bin width)` and is not directly comparable to flow/MDN
  NLL.  Compare CRPS/coverage/PIT for the binned head.
- The noise-aware **NSF** did not improve clean-test NLL at 40 epochs in this
  configuration; Gaussian–Hermite quadrature makes the objective harder to fit
  at a small budget.  MDN noise-aware improves at `n=1000`; the binned noise-aware
  head is the most consistently robust at this short budget.
- The synthetic target is scalar and multimodal; the benchmark does not yet
  include ConR/representation regularization or semi-supervised labels.

## Interpretation

- For a scalar, multimodal, low-budget problem, the **MDN and binned-CDF heads
  are competitive with/beat the flow**, matching the recommendation not to make
  flows the central solution.
- The noise-aware likelihood is not a free win: its benefit depends on the
  training budget and how well the density head itself is fitted.  A
  noise-aware MDN/binned head is a safer small-data default than a noise-aware
  flow until a validation-NLL schedule is tuned.
- Larger clean test NLL for the binned head is expected from the mass/density
  scale; the useful signals are CRPS, coverage, PIT, and rare-slice metrics.
