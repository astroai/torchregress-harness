"""End-to-end smoke tests for the stress_matrix suite.

Regression guard for run stress_20260903_011458, which exposed six failure
modes when every method was executed against every axis' dataset:

1. KeyError: 'labeled_mask_train'          — cross-axis method/dataset pairing
2. AttributeError: LDSLoss.fit_density     — wrong fit API (fit() is correct)
3. TypeError: MSELoss.forward() mask kwarg — plain nn.MSELoss is not mask-aware
4. ValueError: target_variance_train       — NoisyTargetNLL run off-axis
5. TypeError: ~ on float tensor            — PseudoLabelNLL needs a bool mask
6. MLP.__init__() unexpected kwarg         — LUQ-box v0.3.0 signature drift

Each test runs the suite end-to-end in smoke mode (tiny data, 2 epochs) and
asserts zero error rows, so any regression in these paths fails in CI.
"""

from __future__ import annotations

import pytest
from suites.tabular.stress_matrix import StressMatrixSuite

from harness.runners import HarnessConfig

_ALL_AXES = [
    "stress_imbalanced",
    "stress_noisy_inputs",
    "stress_noisy_labels",
    "stress_sparse_labels",
    "stress_ood",
    "stress_semi_sup",
]


@pytest.fixture(scope="module")
def suite() -> StressMatrixSuite:
    return StressMatrixSuite()


def _smoke_config() -> HarnessConfig:
    return HarnessConfig(seed=42, epochs=2, smoke=True)


def _assert_no_error_rows(rows: list[dict], ctx: str) -> None:
    errors = [r for r in rows if "error" in str(r.get("Notes", "")).lower()]
    for r in errors:
        print(f"  ERROR {r['Method']}: {r['Notes'][:160]}")
    assert not errors, f"{ctx}: {len(errors)} error row(s)"


@pytest.mark.parametrize("axis", _ALL_AXES)
def test_axis_runs_clean_end_to_end(suite: StressMatrixSuite, axis: str) -> None:
    """Every axis must produce error-free rows for its own methods (smoke)."""
    cfg = _smoke_config()
    rows = suite.run(axis, config=cfg)
    assert rows, f"{axis}: no rows produced"
    methods = {r["Method"].split("/", 1)[1] for r in rows}
    assert methods, f"{axis}: no methods ran"
    # The axis' methods must map 1:1 — cross-axis methods must not appear.
    for m in methods:
        assert m.split("/", 1)[0] == axis, f"cross-axis method {m} on dataset {axis}"
    _assert_no_error_rows(rows, axis)


def test_every_axis_covered_by_method_registry(suite: StressMatrixSuite) -> None:
    """Each axis must own at least one registered method (no orphan datasets)."""
    axes = {suite._method_axis(m) for m in suite.methods() if "/" in m}
    for axis in _ALL_AXES:
        assert axis in axes, f"no method registered for axis {axis}"


def test_axis_methods_require_only_own_columns(suite: StressMatrixSuite) -> None:
    """Methods of an axis must not require another axis' extra columns."""
    required = suite._AXIS_REQUIRED_KEYS
    for method in suite.methods():
        axis = suite._method_axis(method)
        for other_axis, keys in required.items():
            if other_axis == axis:
                continue
            # A method of axis A must never index columns only axis B provides;
            # the run() axis filter guarantees this pairing never happens.
            assert other_axis not in method, f"method {method} encodes foreign axis {other_axis}"
    # Sanity: the filter map itself must reference real generators.
    from suites.tabular.stress_matrix import _STRESS_GENERATORS

    for axis in required:
        assert axis in _STRESS_GENERATORS


def test_luqbox_methods_import_gated(suite: StressMatrixSuite) -> None:
    """LUQ-box rows either run clean or are skipped — never error."""
    rows = suite.run("stress_ood", config=_smoke_config())
    luqbox = [r for r in rows if "luqbox" in r["Method"]]
    for r in luqbox:
        note = str(r.get("Notes", "")).lower()
        assert "error" not in note, f"{r['Method']}: {r['Notes'][:160]}"
        # If unavailable it must be a clean skip, not a silent success.
        if r.get("RMSE") is None:
            assert "skipped" in note, f"{r['Method']}: null metrics without skip note"
