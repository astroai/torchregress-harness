# Comprehensive Regression Leaderboard

- Target coverage: **0.90**
- Methods ranked: **8**
- Primary ranking metric: **IntervalScore** (lower is better)

## Leaderboard (suite-mean by method)

| Rank | Method | Library | N Suites | N Runs | RMSE | IntervalScore | Coverage | Coverage Dev | ECE |
|---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 1 | NF_Ensemble_K10 | torchregress | 1 | 1 | - | 1.0873 | 0.9133 | 0.0133 | - |
| 2 | NF_Ensemble_K20 | torchregress | 1 | 1 | - | 1.0954 | 0.9133 | 0.0133 | - |
| 3 | NF_Ensemble_K5 | torchregress | 1 | 1 | - | 1.0987 | 0.9200 | 0.0200 | - |
| 4 | GaussianNLL | torchregress | 1 | 1 | - | 1.1038 | 0.9133 | 0.0133 | - |
| 5 | NF_Ensemble_K2 | torchregress | 1 | 1 | - | 1.1148 | 0.8933 | 0.0067 | - |
| 6 | MultivariateGaussian | torchregress | 1 | 1 | - | 1.1487 | 0.9467 | 0.0467 | - |
| 7 | NormalizingFlow | torchregress | 1 | 1 | - | 1.2191 | 0.8533 | 0.0467 | - |
| 8 | SLS | torchregress | 1 | 1 | - | 1.4748 | 0.9867 | 0.0867 | - |

## Per-Suite Breakdown

### Suite: `multivariate_intervals`

#### Dataset: `synthetic_multivariate`

| Method | Library | RMSE | IntervalScore | Coverage | ECE |
|:---|:---|---:|---:|---:|---:|
| NF_Ensemble_K10 | torchregress | - | 1.0873 | 0.9133 | - |
| NF_Ensemble_K20 | torchregress | - | 1.0954 | 0.9133 | - |
| NF_Ensemble_K5 | torchregress | - | 1.0987 | 0.9200 | - |
| GaussianNLL | torchregress | - | 1.1038 | 0.9133 | - |
| NF_Ensemble_K2 | torchregress | - | 1.1148 | 0.8933 | - |
| MultivariateGaussian | torchregress | - | 1.1487 | 0.9467 | - |
| NormalizingFlow | torchregress | - | 1.2191 | 0.8533 | - |
| SLS | torchregress | - | 1.4748 | 0.9867 | - |
