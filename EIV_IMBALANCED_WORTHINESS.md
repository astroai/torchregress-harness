# EIV and imbalanced regression: where the methods are worth it

Measured on the torchregress-harness GPU runs `release_20261018_1746` (EIV),
`release_20261018_1626` (imbalanced) and `release_20261018_1747` (image/SBI).
All rows use `device=cuda`, seed 0 and the suites' shared budgets; treat these
as directional single-seed results, not a ranking.

## Errors-in-variables (EIV)

EIV methods are worth it when **measurement error in the input is large and the
relationship is close to linear**, and when the *slope / clean-X prediction*,
not only the noisy-X RMSE, is the decision metric.

| setting | best measured | MSE baseline | what to use |
|---|---:|---:|---|
| linear, moderate noise | TLS/ODR | 0.249 | `TLS`/`SciPy ODR` or `TLS_Init+FunctionalEIV`; `RC+GaussianNLL` for uncertainty |
| linear, high noise | `EnsembleEIV` 0.780 | 0.794 | `EnsembleEIV`, `RC`, or `MSE_NoisyPredict` |
| linear, very high noise | `EnsembleEIV` 0.974 | 0.994 | `EnsembleEIV`; EIV functional/structural still help attenuation but not RMSE |
| nonlinear | `InputNoiseMDN` 0.824 | 0.843 | input-noise MC / MDN / binned-PDF; do **not** use linear TLS/ODR |
| nonlinear, high/very high noise | `InputNoiseMC`/`BinnedPDF` 0.831–0.839 | 0.849–0.860 | input-noise marginalization; linear EIV losses are worse |

Key observations:

- On the linear synthetic designs, `FunctionalEIV`/`StructuralEIV` improve the
  attenuation diagnostic (e.g. MSE attenuation 0.478 → 0.600 at high noise)
  but not necessarily RMSE; TLS/ODR are the point-estimation winners at low
  noise. At very high noise, `TLS`/`ODR` can be disastrous (RMSE >2).
- On nonlinear designs the linear-EIV losses (`FunctionalEIV`, `StructuralEIV`,
  `ODR`, `TLS`) are **not appropriate**; input-noise marginalization and
  ensembles are more robust. `SIMEX` is consistently expensive (~120 s) and
  not a default winner.
- `RC` (regression calibration) is a cheap, safe baseline; `RC+GaussianNLL`
  gives calibrated predictive intervals at low cost.
- `flow_eiv` (synthetic bimodal, correlated heteroscedastic errors): `FlowEB`
  has the best in-distribution RMSE/NLL and cuts slope bias to ~−3% versus
  −28% for OLS, while `ForwardInversion` is the best OOD extrapolator but has
  larger ID RMSE. Use FlowEB when the latent distribution and measurement
  errors are the target; use forward inversion when extrapolation matters.
- Practical rule: evaluate **clean-X RMSE, slope attenuation, and interval
  coverage** together. A method that improves only noisy-X RMSE may have
  learned an unphysical attenuation.

Caveat: `photoz_eiv`/`galaxiesml_eiv` real-data rows were still being rerun at
the time of writing; synthetic conclusions above are complete. The flow-EIV
baseline is a documented simplified ML-EB approximation, not the paper's
`raddest` code.

## Imbalanced regression

Imbalanced methods are worth it when **tail / worst-region error or calibrated
uncertainty matters**, not when average RMSE is the only metric.

| dataset | best RMSE | best worst-group MAE | best tail RMSE | best NLL / CRPS |
|---|---:|---:|---:|---:|
| California housing | `MSE_BMC` 0.598 | `MSE_GroupDRO` 0.611 | `MSE_BMC` 1.113 | `VIR` 0.783 / 0.297 |
| stress_imbalanced | `MSE` 0.596 | `MSE_ConR` 0.578 | `MSE_GroupDRO` 1.014 | `VIR` 0.939 / 0.329 |

Measured tradeoffs:

- On the **real** California-housing target, `BMC` improves RMSE, worst-region
  MAE and tail RMSE over MSE/PRIME/ConR/RankSim at ~20 s training; `GroupDRO`
  remains the best worst-group method (0.611) but pays average RMSE (0.718).
  `VIR` is the only method with strong calibrated NLL/CRPS and also improves
  worst-region error (0.799), at ~42 s.
- On the **synthetic** stress target, simpler methods win: ConR best worst-group
  (0.578), GroupDRO best tail (1.014), MSE best average RMSE. BMC is roughly
  neutral (0.615 worst, 1.044 tail); RankSim is slightly worse under this
  shared budget (0.629/1.087) and shows no gain.
- `PRIME` and `ConR` are representation-level and cost ~30–38 s; they are
  justified only when feature collapse/geometry is the diagnosed problem.
- `RankSim` is neutral here: it needs a tuned `gamma`, larger batches/feature
  widths, and a task where ranking geometry matters; do not expect a free win.
- `VIR` is worth it when you need an evidential predictive distribution
  (NLL/CRPS) under imbalance, not when you only need point accuracy; its
  intervals should be calibrated/validated because the NIG uncertainty is
  evidential, not a Bayesian posterior.
- `BMC` is worth it when a target-noise scale is known or can be fitted and
  mini-batches are large enough (batch-coupled); the harness uses a documented
  fixed `sigma=0.15*std(y)` for comparability, while the library supports a
  fitted prior and optional sigma.

## Recommended defaults

1. **EIV:** start with `MSE`/`RC` + clean-X evaluation; if linear and noise is
   significant, try `TLS`/`ODR` and `FunctionalEIV`/`StructuralEIV`; if
   nonlinear, use input-noise marginalization or ensembles. Use conformal
   wrappers for intervals.
2. **Imbalanced:** report tail/worst-region metrics alongside RMSE; try
   `GroupDRO` for worst-group robustness, `BMC` for noise-aware tail
   improvement, and `VIR` when calibrated uncertainty is required. Add
   PRIME/ConR/RankSim only when representation collapse is the diagnosis.
3. Do **not** use resampling/SMOTE; it leaks and distorts the target
   distribution, and the harness has no such paths.
