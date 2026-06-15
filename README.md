# torchregress-harness

[![CI](https://github.com/sfabbro/torchregress-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/sfabbro/torchregress-harness/actions/workflows/ci.yml)

**Benchmarking harness for [torchregress](https://github.com/sfabbro/torchregress).**

Compares every torchregress regression method against external software on known tabular and image datasets. Produces machine-readable JSON artifacts and human-readable Markdown reports.

**Related:** [torchregress](https://github.com/sfabbro/torchregress) (library) ·
[torchregress-research](https://github.com/sfabbro/torchregress-research) (NeurIPS SAGE/SPT manuscripts and benchmarks — not part of this harness).

## Quick Start

```bash
# Install with core dependencies
uv pip install -e ".[all]"

# Run a smoke check (tiny data, fast)
uv run python tools/run_all.py --smoke --output results/latest/harness_smoke.json

# Render the report
uv run python tools/render_report.py results/latest/harness_smoke.json

# Run a specific suite on a specific dataset
uv run python tools/run_all.py --suites conformal --datasets diabetes --output results/latest/conformal_diabetes.json
```

## Available Suites

| Suite | Description | External software |
|---|---|---|
| `conformal` | Conformal prediction intervals | MAPIE, crepes, torchcp |
| `bayesian_linear` | Low-shot Bayesian linear regression | BoTorch |
| `tweedie` | Tweedie / compound-Poisson regression | scikit-lego |
| `foundation_model` | Tabular foundation models | TabPFN |
| `robust_regression` | Robust regression with outliers | LightGBM, XGBoost |

*More suites coming in later phases: imbalanced, censored, ordinal, EIV, causal, calibration, ensemble uncertainty, normalizing flows, and image regression (depth, age, keypoints).*

## Datasets

### Tabular (real)
- **California Housing** (8 features, ~20K samples)
- **Diabetes** (10 features, 442 samples)
- **OpenML Year** (90 features, ~515K samples)

### Tabular (synthetic stress tests)
- **Heteroscedastic** — varying noise with input magnitude
- **Outliers** — 15% extreme outliers
- **Tweedie** — zero-inflated compound Poisson-Gamma
- **Censored** — right-censored targets
- **EIV** — errors-in-variables (noisy inputs)

### Image (coming soon)
- NYUv2 depth estimation
- IMDB-WIKI age regression

## Repository Structure

```
torchregress-harness/
├── src/harness/          # Core library: core, metrics, datasets, runners, report
├── suites/               # Benchmark suite implementations
│   ├── tabular/          # Tabular regression suites
│   └── image/            # Image regression suites (coming soon)
├── datasets/             # Cached dataset files (gitignored)
├── results/              # Generated harness artifacts
│   ├── latest/           # Most recent run
│   └── archive/          # Historical runs
├── tests/                # Harness tests
└── tools/                # CLI entry points (run_all, render_report, compare_runs)
```

## Output Schema

Every harness run produces a JSON artifact with this structure:

```json
{
  "artifact": "harness_run",
  "version": 1,
  "timestamp": "2026-06-15T...",
  "environment": { "python": "3.12", "torch": "2.6", ... },
  "suites": ["conformal", "tweedie"],
  "datasets": ["diabetes", "california_housing"],
  "config": { "seed": 42, "epochs": 60 },
  "rows": [
    {
      "Method": "conformal/Split+MLP",
      "Library": "torchregress",
      "Dataset": "diabetes",
      "Suite": "conformal",
      "RMSE": 0.234,
      "Coverage": 0.895,
      "Width": 1.23,
      "train_s": 2.3,
      "eval_s": 0.05,
      "Notes": "point MLP + ConformalLoss(split)"
    }
  ],
  "aggregate": { "n_total": 120, "n_ok": 115, "n_skipped": 5 }
}
```

## CI

- **`lint-test`**: ruff + pytest + harness smoke on every PR/push
- **`heavy`**: full harness run on main pushes and manual triggers

## License

MIT
