# OPTIONAL DEPS — Coverage Map for `torchregress-harness`

Canonical reference for optional dependencies: which library gates which suite/method, and how to install it.

Cross-references:

* [`pyproject.toml`](pyproject.toml) — `[project.optional-dependencies]`
* [`PARITY_REPORT.md`](PARITY_REPORT.md)
* [`PROPOSALS.md`](PROPOSALS.md)

Default development uses **pixi** (`pixi install`). Optional benchmark libraries are package extras installed with pip into that env (or a venv) when needed.

---

## 1. Quick start

### Core (pixi)

```bash
pixi install
pixi run test
```

### Install everything optional (full benchmark coverage)

```bash
pip install -e ".[all]"
```

This pulls in **conformal + bayesian + tweedie + foundation + boosting + causal + test + dev**.

### Install just one option group

| Group | Command | Libraries |
|---|---|---|
| Conformal | `pip install -e ".[conformal]"` | `mapie`, `crepes`, `torchcp` |
| Bayesian | `pip install -e ".[bayesian]"` | `botorch`, `gpytorch` |
| Tweedie   | `pip install -e ".[tweedie]"`  | `scikit-lego` |
| Foundation| `pip install -e ".[foundation]"` | `tabpfn`, `tabicl` |
| Boosting  | `pip install -e ".[boosting]"`   | `lightgbm`, `xgboost` |
| Causal    | `pip install -e ".[causal]"`     | `econml`, `dowhy` |
| Image     | `pip install -e ".[image]"`      | `torchvision`, `lightning`, `timm`, … |
| Test      | `pip install -e ".[test]"`       | `pytest`, `pytest-cov`, `PyYAML` |
| Dev       | `pip install -e ".[dev]"`        | `ruff`, `ty` |

Extras are **unpinned** in `pyproject.toml` (resolve latest compatible). Prefer soft floors only when a known break requires them.

### Common subsets

```bash
pip install -e ".[conformal,bayesian,tweedie,foundation]"
pip install -e ".[conformal,bayesian,foundation]"
```

---

## 2. Library → suite/method coverage

| Extra | Suites / methods gated |
|---|---|
| `conformal` | MAPIE, crepes, torchcp conformal suites |
| `bayesian` | BoTorch / GPyTorch Bayesian linear |
| `tweedie` | scikit-lego Tweedie |
| `foundation` | TabPFN, TabICL |
| `boosting` | LightGBM, XGBoost robust regression |
| `causal` | EconML / DoWhy causal suites |
| `image` | torchvision / lightning / timm image suites |

Skipped rows in harness JSON usually mean the optional library is missing — install the matching extra and re-run.
