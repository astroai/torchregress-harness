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

This pulls in **conformal + bayesian + tweedie + foundation + boosting + image + test + dev**.

### Install just one option group

| Group | Command | Libraries |
|---|---|---|
| Conformal | `pip install -e ".[conformal]"` | `mapie>=1.4.1` (git pin in `EXTERNAL_PINS.md`), `crepes`, `torchcp` |
| Bayesian | `pip install -e ".[bayesian]"` | `botorch`, `gpytorch` |
| Tweedie   | `pip install -e ".[tweedie]"`  | `scikit-lego` |
| Foundation| `pip install -e ".[foundation]"` | `tabpfn`, `tabicl` |
| Boosting  | `pip install -e ".[boosting]"`   | `lightgbm`, `xgboost` |
| Image     | `pip install -e ".[image]"`      | `torchvision`, `torch-uncertainty>=0.13.0` (git pin in `EXTERNAL_PINS.md`), `lightning`, `lightning-uq-box>=0.3.0` (git pin `v0.3.0`), `timm`, … |
| Test      | `pip install -e ".[test]"`       | `pytest`, `pytest-cov`, `PyYAML` |
| Dev       | `pip install -e ".[dev]"`        | `ruff`, `ty` |

Extras are **unpinned** in `pyproject.toml` (resolve latest compatible). Prefer soft floors only when a known break requires them.

> **⚠️ Paper-grade comparisons require the latest *git* versions of MAPIE and torch-uncertainty** (PyPI lags upstream).
> Exact commit SHAs, rationale, and install commands: **[`EXTERNAL_PINS.md`](EXTERNAL_PINS.md)**.
> The `conformal` and `image` groups below list the PyPI floors; install from git per `EXTERNAL_PINS.md` before running paper benchmarks.

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
| `image` | torchvision / lightning / timm image suites; torch-uncertainty `torch_uncertainty` row (>=0.13); lightning-uq-box `luqbox_*` rows (image_ensembles_uq, flattened-pixel comparators); timm `timm_*` rows |

Skipped rows in harness JSON usually mean the optional library is missing — install the matching extra and re-run.
