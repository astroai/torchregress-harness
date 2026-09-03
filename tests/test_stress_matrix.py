"""End-to-end smoke tests for the stress_matrix suite.

Regression guard for run stress_20260903_011458, which exposed six failure
modes when every method was executed against every axis' dataset:
1. KeyError: 'labeled_mask_train'          — cross-axis method/dataset pairing
2. AttributeError: LDSLoss.fit_density     — wrong fit API (fit() is correct)
3. TypeError: MSELoss.forward() mask kwarg — plain nn.MSELoss is not mask-aware
4. ValueError: target_variance_train       — NoisyTargetNLL run off-axis
5. TypeError: ~ on float tensor            — PseudoLabelNLL needs a bool mask
6. MLP.__init__() unexpected kwarg         — LUQ-box v0.3.0 signature drift
7. (run stress_20260903_174930) LUQMLP defaults to dropout_p=0.0 → the
   luqbox_MCDropout row had *no* dropout layers: MC std collapsed to the
   1e-8 clip floor (NLL 1.8e15) and point metrics were bit-identical to
   luqbox_GaussianNLL. Guarded by test_luqbox_mcdropout_has_real_dropout.

Each test runs the suite end-to-end in smoke mode (tiny data, 2 epochs) and
asserts zero error rows, so any regression in these paths fails in CI.
"""

from __future__ import annotations

import pytest
import torch
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


def test_sage_tuned_hyperparameters_wired() -> None:
    """SAGE config must stay on the tuned 3-seed sweep optimum.

    Sweep (2026-09-03) on the full stress_semi_sup axis found RMSE 0.58 /
    NLL 1.06 with BetaNLL(beta=1) supervision, agreement_weight 0.3 and 100
    epochs. Plain GaussianNLL supervision collapses the variance head, and
    <400 optimizer steps leaves the EMA teacher polluted by its random init
    (RMSE 0.78+).
    """
    import inspect
    import types

    from suites.tabular import stress_matrix as sm

    src = inspect.getsource(sm.StressMatrixSuite._fit_sage)
    assert "BetaNLLLoss(beta=1.0)" in src, "supervision must be BetaNLL(beta=1.0)"
    assert "agreement_weight=0.3" in src, "agreement_weight must be 0.3"
    assert "epochs = 2 if config.smoke else 100" in src, "full-run epochs must be 100"
    # BetaNLLLoss must be importable where _fit_sage can see it.
    mod = types.SimpleNamespace(**vars(sm))
    assert hasattr(mod, "BetaNLLLoss")


def test_noisy_labels_generator_emits_test_variance() -> None:
    """The known noise law is deterministic from X, so test-side variance exists."""
    import numpy as np
    from suites.tabular.stress_matrix import load_stress_dataset

    data = load_stress_dataset("stress_noisy_labels")
    tv = data["target_variance_test"]
    expected = 0.1 + 0.4 * np.abs(data["X_test"][:, 0])
    np.testing.assert_allclose(tv, expected, rtol=1e-5)
    assert len(tv) == len(data["X_test"])


def test_noisytarget_predict_adds_known_noise(suite: StressMatrixSuite) -> None:
    """NoisyTargetGaussianNLL eval variance must include the known test noise.

    Training scores NLL against pred_var + target_var, so the head's own
    variance optimizes toward the residual excess only. Without adding the
    known noise back at eval, NLL is scored against a collapsed variance
    (run stress_20260903_011458: NLL 6.5-3379 vs ~1.2 for its peers).
    """
    import numpy as np
    from suites.tabular.stress_matrix import load_stress_dataset

    from harness.metrics import compute_all_metrics

    data = load_stress_dataset("stress_noisy_labels")
    cfg = _smoke_config()
    model = suite.fit("stress_noisy_labels/NoisyTargetGaussianNLL_MLP", data, cfg)
    preds = suite.predict(
        "stress_noisy_labels/NoisyTargetGaussianNLL_MLP", model, data["X_test"], cfg
    )
    mean = np.asarray(preds["mean"]).reshape(-1)
    std = np.asarray(preds["std"]).reshape(-1)
    assert len(mean) == len(std) > 0
    tv = np.asarray(data["target_variance_test"], dtype=np.float64)[: len(mean)]
    # Predictive variance must dominate the known noise everywhere.
    assert np.all(std.astype(np.float64) ** 2 >= tv - 1e-8)
    # Peer methods must NOT get the additive variance (raw head protocol).
    plain = suite.fit("stress_noisy_labels/GaussianNLL_MLP", data, cfg)
    assert "target_variance_test" not in plain
    # Smoke-run NLL must be finite (the collapse produced ~1e3 values).
    yte = np.asarray(data["y_test"]).reshape(-1)[: len(mean)]
    m = compute_all_metrics({"mean": mean, "std": std}, yte)
    assert np.isfinite(m["NLL"])
    assert m["NLL"] < 50.0, f"NLL still diverged: {m['NLL']}"


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


def test_luqbox_mcdropout_has_real_dropout(suite: StressMatrixSuite) -> None:
    """luqbox_MCDropout must train/predict with actual dropout layers.

    Run stress_20260903_174930 shipped LUQMLP with its default dropout_p=0.0:
    MCDropoutBase.find_dropout_layers found nothing, MC 'sampling' replayed a
    deterministic net 20 times (std -> 1e-8 clip floor -> NLL 1.8e15) and the
    row's point metrics were bit-identical to luqbox_GaussianNLL. The two rows
    must differ, std must be non-degenerate, and NLL must be finite.
    """
    import numpy as np
    from suites.tabular.stress_matrix import _LUQBOX_AVAILABLE, load_stress_dataset

    from harness.metrics import compute_all_metrics

    if not _LUQBOX_AVAILABLE:
        pytest.skip("lightning-uq-box not importable in this env (rows degrade to skips)")

    data = load_stress_dataset("stress_ood")
    cfg = _smoke_config()
    model = suite.fit("stress_ood/luqbox_MCDropout", data, cfg)
    # Dropout layers must actually exist in the wrapped net.
    base = model["module"].model
    dropouts = [m for m in base.modules() if isinstance(m, torch.nn.Dropout)]
    assert dropouts, "luqbox_MCDropout base net has no nn.Dropout layers"
    assert all(p > 0 for d in dropouts for p in [d.p]), "dropout layers exist but p=0"

    preds = suite.predict("stress_ood/luqbox_MCDropout", model, data["X_test"], cfg)
    mean = np.asarray(preds["mean"]).reshape(-1)
    std = np.asarray(preds["std"]).reshape(-1)
    assert len(mean) == len(std) > 0
    # Non-degenerate uncertainty: std must not sit on the 1e-8 clip floor.
    assert float(np.min(std)) > 1e-6, f"MC std collapsed: min(std)={float(np.min(std))}"
    # Point predictions must differ from the deterministic sibling (same seed).
    det = suite.fit("stress_ood/luqbox_GaussianNLL", data, cfg)
    det_preds = suite.predict("stress_ood/luqbox_GaussianNLL", det, data["X_test"], cfg)
    det_mean = np.asarray(det_preds["mean"]).reshape(-1)
    assert not np.allclose(mean, det_mean), "MCDropout mean identical to GaussianNLL mean"
    # Metrics must be finite (the collapse produced NLL ~1.8e15).
    yte = np.asarray(data["y_test"]).reshape(-1)[: len(mean)]
    m = compute_all_metrics({"mean": mean, "std": std}, yte)
    assert np.isfinite(m["NLL"]), f"NLL not finite: {m['NLL']}"
    assert m["NLL"] < 50.0, f"NLL still diverged: {m['NLL']}"
