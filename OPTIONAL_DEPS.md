# OPTIONAL DEPS — Coverage Map for `torchregress-harness`

Canonical reference for optional dependencies: which library gates which suite/method, and how to install it.

Cross-references:

* [`pyproject.toml`](pyproject.toml) — `[project.optional-dependencies]`
* [`EXTERNAL_PINS.md`](EXTERNAL_PINS.md) — git pins for paper-grade runs
* [`PARITY_REPORT.md`](PARITY_REPORT.md)
* [`PROPOSALS.md`](PROPOSALS.md)

Default development uses **pixi** (`pixi install`, `linux-64` only). The default pixi env already ships `torchvision`, `lightning`, `timm`, `zuko`, `scoringrules` and `properscoring`. Other benchmark libraries are package extras installed with pip into that env (or a venv) when needed.

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

This pulls in **conformal + bayesian + tweedie + foundation + boosting + image + sbi + causal + distributional + scoring + test + dev**.

### Install just one option group

| Group | Command | Libraries |
|---|---|---|
| Conformal | `pip install -e ".[conformal]"` | `mapie>=1.5.0` (git pin in `EXTERNAL_PINS.md`), `crepes`, `torchcp` |
| Bayesian | `pip install -e ".[bayesian]"` | `botorch`, `gpytorch` |
| Tweedie   | `pip install -e ".[tweedie]"`  | `scikit-lego` |
| Foundation| `pip install -e ".[foundation]"` | `tabpfn>=9.1`, `tabicl>=2.2`, `structured-data-models` (git `main`) |
| Boosting  | `pip install -e ".[boosting]"`   | `lightgbm`, `xgboost` |
| Image     | `pip install -e ".[image]"`      | `torchvision`, `torch-uncertainty>=0.13.0`, `lightning`, `lightning-uq-box>=0.4.0` (git pins in `EXTERNAL_PINS.md`), `timm`, `requests`, `Pillow`, `h5py` |
| SBI       | `pip install -e ".[sbi]"`        | `sbi` (git pin), `pyro-ppl`; then `pip install --no-deps "sbibm==1.1.0"` |
| Causal    | `pip install -e ".[causal]"`     | `econml>=0.17.0`, `doubleml>=0.11.4` |
| Distributional | `pip install -e ".[distributional]"` | `ngboost>=0.5.11`, `laplace-torch>=0.2.3` |
| Scoring   | `pip install -e ".[scoring]"`    | `scoringrules>=0.11.0`, `properscoring` |
| Test      | `pip install -e ".[test]"`       | `pytest`, `pytest-cov`, `PyYAML` |
| Dev       | `pip install -e ".[dev]"`        | `ruff`, `ty` |

Most extras are **unpinned** in `pyproject.toml` (resolve latest compatible); soft floors exist only where a known break requires them.

> **⚠️ Paper-grade comparisons require the latest *git* versions of MAPIE, torch-uncertainty and Lightning-UQ-Box** (PyPI lags upstream).
> Exact commit SHAs, rationale, and install commands: **[`EXTERNAL_PINS.md`](EXTERNAL_PINS.md)**.
> The table above lists the PyPI floors; install from git per `EXTERNAL_PINS.md` before running paper benchmarks.

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
| `tweedie` | scikit-lego Tweedie (`tweedie`) |
| `foundation` | TabPFN, TabICL, sdm KumoTabular / TabFM (`foundation_model`, FM UQ grid) |
| `boosting` | LightGBM, XGBoost robust regression |
| `image` | torchvision / lightning / timm image suites; torch-uncertainty `torch_uncertainty` row (>=0.13); lightning-uq-box `luqbox_*` rows (image_ensembles_uq, flattened-pixel comparators); timm `timm_*` rows; `h5py` / `Pillow` dataset loaders (GalaxiesML, image datasets) |
| `sbi` | `sbi_parity`, `density_parity` (sbi estimators); sbibm tasks and reference posteriors |
| `causal` | EconML `causal`, EconML + DoubleML `orthogonal_inference` |
| `distributional` | Declared, not yet imported by any suite |
| `scoring` | Declared, not yet imported; reserved for the metrics parity tests vs `scoringrules` (`RELEASE_0.3_STATUS.md`) |

Skipped rows in harness JSON usually mean the optional library is missing — install the matching extra and re-run.
