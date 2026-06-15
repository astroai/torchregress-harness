# AGENTS.md

This file provides guidance to coding agents (Claude, Gemini, Codex) when working in this repository.

## Project Overview

**torchregress-harness** is a standalone benchmarking harness for [torchregress](https://github.com/sfabbro/torchregress). It compares torchregress regression methods against external software on known tabular and image datasets, producing structured JSON artifacts and Markdown reports.

## Development Commands

This project uses [uv](https://github.com/astral-sh/uv) as the package manager.

### Setup
```bash
uv pip install -e ".[all]"
```

### Testing
```bash
uv run pytest
uv run pytest tests/test_core.py -v
```

### Code Quality
```bash
uv run ruff check src/harness tests tools suites
uv run ruff format --check src/harness tests tools suites
```

### Run Harness
```bash
# Smoke check (tiny data, fast)
uv run python tools/run_all.py --smoke --output results/latest/harness_smoke.json

# Specific suite + dataset
uv run python tools/run_all.py --suites conformal --datasets diabetes

# Full run
uv run python tools/run_all.py --output results/latest/harness_full.json

# Render report
uv run python tools/render_report.py results/latest/harness_smoke.json
```

## Architecture

### Harness Core (`src/harness/`)

| Module | Purpose |
|---|---|
| `core.py` | Seeding, timing, summary printing, JSON artifact writing |
| `metrics.py` | Unified metric computation (point, interval, distribution, calibration) |
| `datasets.py` | Dataset registry with real tabular + synthetic + image datasets |
| `runners.py` | `HarnessRunner` ABC + `HarnessConfig` dataclass |
| `report.py` | JSON → Markdown rendering |

### Suite Protocol

Every suite inherits from `HarnessRunner` and implements:

```python
class MySuite(HarnessRunner):
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def supported_datasets(self) -> list[str]: ...
    def methods(self) -> list[str]: ...
    def fit(self, method, splits, config) -> Any: ...
    def predict(self, method, model, X_test, config) -> dict[str, np.ndarray]: ...
```

The `run()` method is already implemented and orchestrates fit/predict/evaluate.

### Adding a New Suite

1. Create `suites/tabular/my_suite.py` with a `HarnessRunner` subclass
2. Implement `fit()`, `predict()`, `methods()`, and properties
3. Register it in `suites/tabular/__init__.py`
4. Add it to `ALL_SUITES` in `tools/run_all.py`
5. Add smoke tests in `tests/test_suite_smoke.py`

### Dataset Registry

All datasets live in `DATASET_REGISTRY` in `datasets.py`. Each entry has a `loader` callable that returns `{X_train, X_test, y_train, y_test}` as numpy arrays. Use `load_dataset(name)` to fetch data.

To add a dataset:
1. Write a `_load_my_dataset()` function
2. Add a `DatasetInfo` entry to `DATASET_REGISTRY`

## Key Design Rules

- **torchregress is a pip dependency** — suites call torchregress APIs directly; no vendoring
- **External software is optional** — use `try/except ImportError` guards; emit `None` metrics + skip note when unavailable
- **Suites must be reproducible** — use `set_harness_seed()`; declare seed/budget policies
- **Every run produces valid JSON** — no partial artifacts; rows with skipped methods are emitted with null metrics
- **Stubs before implementation** — placeholder suites raise `NotImplementedError` until Phase 2/3
