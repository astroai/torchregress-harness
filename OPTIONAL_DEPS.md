# OPTIONAL DEPS — Coverage Map for `torchregress-harness`

This guide is the canonical "what's gated by what" reference for the harness's
optional dependencies.  Every optional library, the suite(s) it gates, the
methods it controls, and the `uv pip install -e ".[...]"` one-liner that enables
it, in one place.

Cross-references:

* [`pyproject.toml`](pyproject.toml) — `[project.optional-dependencies]` (verbatim below)
* [`PARITY_REPORT.md`](PARITY_REPORT.md) §12 — `normflows_NSF` broken on single-channel input
* [`PROPOSALS.md`](PROPOSALS.md) §2-§15 — open PROPOSALS for which library features would close which gaps

---

## 1. Quick start

### Install everything (recommended for full benchmark coverage)

```bash
uv pip install -e ".[all]"
```

This pulls in **conformal + bayesian + tweedie + foundation + boosting + causal + test + dev** extras in one go.

### Install just one option group

| Group | Command | Libraries |
|---|---|---|
| Conformal | `uv pip install -e ".[conformal]"` | `mapie`, `crepes`, `torchcp` |
| Bayesian | `uv pip install -e ".[bayesian]"` | `botorch`, `gpytorch` |
| Tweedie   | `uv pip install -e ".[tweedie]"`  | `scikit-lego` |
| Foundation| `uv pip install -e ".[foundation]"` | `tabpfn`, `tabicl` |
| Boosting  | `uv pip install -e ".[boosting]"`   | `lightgbm`, `xgboost` |
| Causal    | `uv pip install -e ".[causal]"`     | `econml`, `dowhy` |
| Test      | `uv pip install -e ".[test]"`       | `pytest`, `pytest-cov`, `PyYAML` |
| Dev       | `uv pip install -e ".[dev]"`        | `ruff`, `mypy` |

### Common subsets

```bash
# Most external benchmark coverage — every suite except causal/boosting:
uv pip install -e ".[conformal,bayesian,tweedie,foundation]"

# Production-only — no test/dev deps, no causal/boosting:
uv pip install -e ".[conformal,bayesian,foundation]"
```

---

## 2. `pyproject.toml` `[project.optional-dependencies]` (verbatim)

```toml
[project.optional-dependencies]
conformal = [
    "mapie>=0.8.0",
    "crepes>=0.7.0",
    "torchcp>=0.3.0",
]
bayesian = [
    "botorch>=0.10.0",
    "gpytorch>=1.11.0",
]
tweedie = [
    "scikit-lego>=0.7.0",
]
foundation = [
    "tabpfn>=1.0.0",
    "tabicl",
]
boosting = [
    "lightgbm>=4.0.0",
    "xgboost>=2.0.0",
]
causal = [
    "econml>=0.15.0",
    "dowhy>=0.11.0",
]
test = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
    "PyYAML>=6.0.0",
]
dev = [
    "ruff>=0.5.0",
    "mypy>=1.10.0",
]
all = [
    "torchregress-harness[conformal,bayesian,tweedie,foundation,boosting,causal,test,dev]",
]
```

---

## 3. Library → suite/method coverage matrix

### 3a. Default-installed (already in this env's `.venv`)

| Library              | Version | Suite(s) gated                    | Source line                        |
|----------------------|---------|-----------------------------------|------------------------------------|
| `mapie`              | 1.4.1   | `conformal`                       | `suites/tabular/conformal.py:56-58` |
| `crepes`             | 0.9.1   | `conformal`                       | `suites/tabular/conformal.py:49-51` |
| `tabicl`             | 2.1.1   | `conformal` + `foundation_model`  | `suites/tabular/conformal.py:70-72` + `foundation_model.py:46-49` |
| `statsmodels`        | 0.14.6  | `robust_regression` + `tweedie`   | `suites/tabular/robust_regression.py:66-68` + `tweedie.py:38-40` |
| `pzflow`             | 4.0.0   | `pzflow_parity`                   | `suites/tabular/pzflow_parity.py:225` |
| `normflows`          | 1.7.3   | `pzflow_parity`                   | `suites/tabular/pzflow_parity.py` (intrinsic broadcast bug on 1-D input — suite short-circuits and emits a `known-bad:` row per §3c) |

### 3b. NOT installed (skipping gates `Notes="skipped: ImportError"` rows in the JSON artifact)

| Library        | `extras` group | Suite(s) gated              | Source line                          |
|----------------|----------------|------------------------------|--------------------------------------|
| `botorch`      | `bayesian`     | `bayesian_linear`            | `suites/tabular/bayesian_linear.py:39-41` |
| `gpytorch`     | `bayesian`     | `bayesian_linear` (transitive) | pulled in by `botorch` install      |
| `tabpfn`       | `foundation`   | `foundation_model`           | `suites/tabular/foundation_model.py:37-40` |
| `torchcp`      | `conformal`    | `conformal`                  | `suites/tabular/conformal.py:64-65`  |
| `scikit-lego`  | `tweedie`      | `tweedie`                    | `suites/tabular/tweedie.py:32-33`    |
| `lightgbm`     | `boosting`     | `robust_regression`          | `suites/tabular/robust_regression.py:52-54` |
| `xgboost`      | `boosting`     | `robust_regression`          | `suites/tabular/robust_regression.py:59-61` |
| `econml`       | `causal`       | **No suite exists yet**      | n/a — extras group declared, no implementation |
| `dowhy`        | `causal`       | **No suite exists yet**      | n/a — extras group declared, no implementation |

### 3c. Known-broken even when installed

| Library       | Suite              | Issue                                                                 |
|---------------|---------------------|-----------------------------------------------------------------------|
| `normflows`   | `pzflow_parity`     | `normflows_NSF` is intrinsically broken on the harness's 1-D tabular target (broadcast error in the normflows library when combining `num_blocks>1` with `num_context_channels≠None` on single-channel input — NOT an import/availability issue; the library itself works on its own benchmarks).  The suite short-circuits the method and emits a `Notes="known-bad: normflows_NSF pzflow_parity #broadcast error on 1-D tabular inputs"` row in every leaderboard artifact.  See [`PARITY_REPORT.md`](PARITY_REPORT.md) §12 for the root-cause investigation. |

---

## 4. Per-suite gate details

### `conformal` — 9 methods, 4 gated dependencies

* **Always-on methods** (no optional dep): `Split_CP`, `Split+Linear`, `Split+MLP`, `CQR+MLP`
* Gated methods → effect of missing the dep:
  * `mapie` missing → 4 `MAPIE_*` methods emit `skipped` rows
  * `crepes` missing → `crepes_*` method emits `skipped` row
  * `torchcp` missing → `torchcp_*` method(s) emit `skipped` row(s)
  * `tabicl` missing → 3 `*TabICL` methods emit `skipped` rows (Split+TabICL, MAPIE_Split+TabICL, crepes_Split+TabICL)

### `bayesian_linear` — 3 methods

* **Always-on**: `BLR_Linear` (torchregress, parity with sklearn `BayesianRidge`), `sklearn_BayesianRidge`
* **Gated on `botorch`** (and transitive `gpytorch`): `BoTorch_SingleTaskGP` — appears in `methods()` registry but `if _BOTORCH_AVAILABLE` guard causes `skipped` row when missing

### `tweedie` — 2 methods

* **Always-on**: `Tweedie_Linear` (parity with `statsmodels.GLM`)
* **Gated on `scikit-lego`** (`_SKLEGO_AVAILABLE` flag at `tweedie.py:32-33`): the scikit-lego TweedieRegressor parity reference

### `robust_regression` — multiple methods, 3 gated dependencies

* Per-method guards at `robust_regression.py:137-141`:
  * `if _STATSMODELS_AVAILABLE` → gates `Huber_Linear` (statsmodels.RLM parity) + `Quantile_Linear` (sklearn QuantileRegressor parity, mentioned in PARITY_REPORT §2)
  * `if _LIGHTGBM_AVAILABLE` → gates LightGBM-* methods
  * `if _XGBOOST_AVAILABLE` → gates XGBoost-* methods

### `foundation_model` — 2 method families, 2 dependencies

* `_TABPFN_AVAILABLE` flag (`foundation_model.py:37-40`) → gates TabPFN-* methods
* `_TABICL_AVAILABLE` flag (`foundation_model.py:46-49`) → gates TabICL-* methods

### `pzflow_parity` — 3 methods, 2 dependencies

* `pzflow` present (default) → `pzflow_NSF` works (it does have the wider-intervals and slower NLL but still runs)
* `normflows` present (default) → `normflows_NSF` is intrinsically broken (broadcast error per `PARITY_REPORT.md` §12); suite emits a `known-bad:` row instead of crashing
* `torchregress_NSF` (no optional dep, baseline method) is always available

### `causal` group

**Extras group declared but no suite implemented.** `econml` + `dowhy` install cleanly but no methods become available because `suites/causal/` doesn't exist. Tracked as a future-work item.

### `boosting` group

* `lightgbm` + `xgboost` install cleanly
* Methods gated under `robust_regression` (`if _LIGHTGBM_AVAILABLE` / `if _XGBOOST_AVAILABLE`)
* No boosting-only suite yet — boosting methods live in `robust_regression`'s method registry

---

## 5. When a library is missing — row behavior

When a gated method's `_X_AVAILABLE` flag is False, the suite's `fit()` branch falls through to:

```python
except ImportError as exc:
    row = {
        "Method": f"{suite.name}/{method}",
        "Library": library_tag,
        "Dataset": dataset_name,
        "Suite": suite.name,
        "Coverage": None, "Width": None, "IntervalScore": None,
        "train_s": None, "eval_s": None,
        "Notes": f"skipped: {type(exc).__name__}: {exc}",
    }
```

The aggregator in `tools/leaderboard_regression.py` recognises the `Notes` substring `"skipped"`, `"failed"`, `"error"` to exclude the row from rankings (see `tests/test_leaderboard.py::test_is_skipped_handles_variants`).

---

## 6. Notable gap — `statsmodels` is not in any extras group

`statsmodels` is used as a **parity reference** in 3 suites:

* `suites/tabular/robust_regression.py` — `Huber_Linear` parity (statsmodels RLM)
* `suites/tabular/robust_regression.py` — quantile-regression methods
* `suites/tabular/tweedie.py` — `Tweedie_Linear` parity (statsmodels GLM)

It is **not** declared in any `[project.optional-dependencies]` group in `pyproject.toml` yet it is installed in the default `.venv` (transitively, likely via `scikit-learn`'s dependency chain). On a **fresh setup without scikit-lego/botorch/etc. installed**, `statsmodels` may not auto-install — add `uv pip install statsmodels` explicitly if you see `skipped: ImportError: ...statsmodels...` rows on `Huber_Linear` or `Tweedie_Linear`.

**Recommended follow-up**: add `statsmodels` to the `tweedie` extras group (or a new `parity` group) so the install one-liner covers it.

---

## 7. Verify your install

```bash
# Quick sanity — every optional library imports cleanly:
uv run python - <<'PY'
import importlib
libs = [
    "mapie", "crepes", "torchcp",
    "botorch", "gpytorch",
    "statsmodels",
    "tabpfn", "tabicl",
    "lightgbm", "xgboost",
    "scikit_leo",
    "pzflow", "normflows",
    "econml", "dowhy",
]
for lib in libs:
    try:
        importlib.import_module(lib)
        print(f"  \u2713 {lib}")
    except ImportError as exc:
        print(f"  \u2717 {lib}: {exc!s:.80}")
PY
```

Expected: every library reports `\u2713` if `uv pip install -e ".[all]"` succeeded.

---

## 8. Common gotchas

| Symptom                                                       | Cause                                                                 | Fix                                                                 |
|---------------------------------------------------------------|-----------------------------------------------------------------------|---------------------------------------------------------------------|
| `skipped: ImportError: No module named 'tabpfn'`              | `tabpfn` not installed                                                | `uv pip install -e ".[foundation]"`                                  |
| `BoTorch_SingleTaskGP` rows always `skipped`                   | `botorch` (and `gpytorch`) not installed                              | `uv pip install -e ".[bayesian]"`                                   |
| `normflows_NSF` rows always `known-bad: normflows_NSF pzflow_parity #broadcast error on 1-D tabular inputs` | Intrinsically broken — the suite short-circuits and emits a deterministic `known-bad:` row by design | See `PARITY_REPORT.md` §12.  If you want to test the path, monkey-patch `PzflowParitySuite._KNOWN_BAD` to remove the entry; this is NOT recommended on a clean env because it reproduces the upstream normflows bug. |
| `Huber_Linear` rows `skipped` even after `[tweedie]` install  | `statsmodels` is **not** in any extras group — see §6                  | `uv pip install statsmodels`                                        |
| `tabpfn` install hangs (downloads a 5 GB checkpoint)            | TabPFN's first-use download                                           | Pre-download: `python -c "import tabpfn; tabpfn.TabPFNClassifier()"` then set `HF_HOME` env var |
| `MAPIE_CQR+GBR` runs but produces NaN widths                   | Old MAPIE 0.x API — current code targets MAPIE 1.x                    | Ensure `mapie>=1.0.0` (the extras constraint is `>=0.8.0`; pin higher) |
| `tests/test_suite_smoke.py::test_inference_*_X5_default` fails | `_NF_ENSEMBLE_K_REFERENCE` differs between scalar + MV suites          | Both suites must keep the same value — see PROPOSALS bump history    |

---

## 9. Incrementally enable everything

```bash
# 1. Clone + uv-install with ``[all]``:
git clone https://github.com/.../torchregress-harness && cd torchregress-harness
uv pip install -e ".[all]"

# 2. Confirm the full benchmark surface is reachable:
uv run python tools/leaderboard_regression.py --smoke --datasets diabetes --suites conformal

# 3. Full benchmark (warning: ~30 min on CPU):
uv run python tools/leaderboard_regression.py --datasets diabetes,california_housing --suites conformal --epochs 60 --output results/latest/leaderboard_full.json
```

---

## 10. Where to look when something goes wrong

| Symptom type                              | First file to read                                  |
|-------------------------------------------|------------------------------------------------------|
| Skipped row's `Notes` is unhelpful        | `src/harness/runners.py:382-401` (canonical try/except + Notes string) |
| New optional lib doesn't gate a method   | Mimic `conformal.py:49-72` pattern: top-level `_X_AVAILABLE` flag + per-method `if _X_AVAILABLE` guard |
| Test suite fails on a gated method        | `tests/test_suite_smoke.py` (smoke runner) + `tests/test_leaderboard.py::test_is_skipped_handles_variants` |
| Adding a new optional dep to pyproject    | Update `pyproject.toml` `[project.optional-dependencies]`, then this file's §2 table, then re-run `uv pip install -e ".[<new_group>]"` to confirm |
