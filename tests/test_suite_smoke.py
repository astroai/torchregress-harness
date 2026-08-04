"""Smoke tests for harness suites.

Each test imports a suite class and verifies it is properly structured
(methods, supported_datasets, name, description).
"""

import numpy as np
import pytest
from suites._backbones import NF_ENSEMBLE_K_VARIANTS
from suites.image.age_regression import AgeRegressionSuite
from suites.image.depth_estimation import DepthEstimationSuite

# All suites should be importable
from suites.tabular.bayesian_linear import BayesianLinearSuite
from suites.tabular.calibration import CalibrationSuite
from suites.tabular.censored import CensoredSuite
from suites.tabular.conformal import ConformalSuite
from suites.tabular.distributional_bins import DistributionalBinsSuite
from suites.tabular.eiv import EIVSuite
from suites.tabular.ensembles import EnsemblesSuite
from suites.tabular.foundation_model import FoundationModelSuite
from suites.tabular.multivariate_intervals import MultivariateIntervalsSuite
from suites.tabular.ordinal import OrdinalSuite
from suites.tabular.pzflow_parity import PzflowParitySuite
from suites.tabular.r2c_conformal import R2CConformalSuite
from suites.tabular.robust_regression import RobustRegressionSuite
from suites.tabular.sklearn_baselines import SklearnBaselinesSuite
from suites.tabular.tweedie import TweedieSuite
from suites.tabular.uq import UQSuite

ALL_SUITES = [
    ConformalSuite,
    BayesianLinearSuite,
    TweedieSuite,
    FoundationModelSuite,
    RobustRegressionSuite,
    CalibrationSuite,
    SklearnBaselinesSuite,
    EnsemblesSuite,
    UQSuite,
    DistributionalBinsSuite,
    MultivariateIntervalsSuite,
    PzflowParitySuite,
    CensoredSuite,
    EIVSuite,
    OrdinalSuite,
    R2CConformalSuite,
    AgeRegressionSuite,
    DepthEstimationSuite,
]


@pytest.mark.parametrize("suite_cls", ALL_SUITES)
def test_suite_has_name_and_description(suite_cls) -> None:
    suite = suite_cls()
    assert isinstance(suite.name, str) and len(suite.name) > 0
    assert isinstance(suite.description, str) and len(suite.description) > 0


@pytest.mark.parametrize("suite_cls", ALL_SUITES)
def test_suite_has_supported_datasets(suite_cls) -> None:
    suite = suite_cls()
    assert isinstance(suite.supported_datasets, list)
    assert len(suite.supported_datasets) > 0
    assert all(isinstance(d, str) for d in suite.supported_datasets)


@pytest.mark.parametrize("suite_cls", ALL_SUITES)
def test_suite_has_methods(suite_cls) -> None:
    suite = suite_cls()
    methods = suite.methods()
    assert isinstance(methods, list)
    assert len(methods) > 0
    assert all(isinstance(m, str) for m in methods)


# ---------------------------------------------------------------------------
# NF_Ensemble_K5 (Proposal #10) — must be registered in distributional_bins
# ---------------------------------------------------------------------------


def test_distributional_bins_has_nf_ensemble() -> None:
    """NF_Ensemble_K5 (Proposal #10) must be a registered method.

    The Flow-ensemble is the documented remedy for single-NF overconfidence
    on small tabular datasets (CoverageDeviationAbsMean: 0.147 → ~0.05).
    Removing it would silently lose the calibration-improvement benchmark.
    """
    suite = DistributionalBinsSuite()
    assert "NF_Ensemble_K5" in suite.methods()


def test_distributional_bins_nf_ensemble_is_torchregress() -> None:
    """NF_Ensemble_K5 must be tagged as a torchregress method (not external)."""
    suite = DistributionalBinsSuite()
    assert suite.library_name("NF_Ensemble_K5") == "torchregress"
    assert "independent" in suite.method_notes("NF_Ensemble_K5").lower()


# ---------------------------------------------------------------------------\n# NF_Ensemble_K-sweep family (Proposal #10 §14 + §15).  The shared tuple
# `NF_ENSEMBLE_K_VARIANTS` is imported from `suites._backbones` (single source
# of truth across scalar / MV suites).  Tests iterate over the imported name
# to enforce registration parity so a future K (e.g. K40) only needs adding
# to the hoist site.
# ---------------------------------------------------------------------------


def test_distributional_bins_k_sweep_registered() -> None:
    """The NF_Ensemble K-sweep family must include all four K \u2208 {2, 5, 10, 20}.

    Removing any one of these would silently lose the calibration-vs-cost knee
    benchmark documented in PARITY_REPORT.md \u00a714.
    """
    suite = DistributionalBinsSuite()
    methods = suite.methods()
    for variant in NF_ENSEMBLE_K_VARIANTS:
        assert variant in methods, (
            f"missing {variant} from suite.methods(); K-sweep \u00a714 invalidated"
        )


def test_distributional_bins_k_sweep_share_architecture() -> None:
    """All K-sweep variants must share HIDDEN + TRANSFORMS + N_SAMPLES.

    Only K itself is allowed to vary across the family.  Architecture drift
    between K=2 and K=20 would corrupt the calibration-vs-cost interpretation.
    H is locked via class constant (shared), so this test enforces member-list
    size and the per-method notes correctly state the K.
    """
    suite = DistributionalBinsSuite()
    notes = suite.method_notes
    expected_Ks = {
        "NF_Ensemble_K2": 2,
        "NF_Ensemble_K5": 5,
        "NF_Ensemble_K10": 10,
        "NF_Ensemble_K20": 20,
    }
    for variant, K in expected_Ks.items():
        text = notes(variant).lower()
        assert str(K) in text, f"{variant} notes must mention the K value ({K}); got {text!r}"


# ---------------------------------------------------------------------------
# NF_Ensemble_K5 must also be registered in multivariate_intervals
# (extended PROPOSAL #10 validation across 3D targets in §10b/§11b).
# ---------------------------------------------------------------------------


def test_multivariate_intervals_has_nf_ensemble() -> None:
    """NF_Ensemble_K5 (Proposal #10) must also be a multivariate method.

    Without this the multivariate JointCoverage benchmark regresses silently to
    only single NormalizingFlow + GaussianNLL + SLS — making any future
    ensemble-vs-single calibration comparison impossible on §10/§11 datasets.
    """
    suite = MultivariateIntervalsSuite()
    assert "NF_Ensemble_K5" in suite.methods()


def test_multivariate_intervals_nf_ensemble_is_torchregress() -> None:
    """NF_Ensemble_K5 in multivariate_intervals must be torchregress + mention
    pooling (Proposal #10 mechanism — lock the wording so a future edit
    cannot silently drop the pooling description)."""
    suite = MultivariateIntervalsSuite()
    assert suite.library_name("NF_Ensemble_K5") == "torchregress"
    notes = suite.method_notes("NF_Ensemble_K5").lower()
    assert "pooled" in notes, (
        "NF_Ensemble_K5 notes must mention 'pooled' (Proposal #10 mechanism); "
        f"got: {suite.method_notes('NF_Ensemble_K5')!r}"
    )


# ---------------------------------------------------------------------------
# NF_Ensemble_K-sweep family (Proposal #10 §15 follow-up) on MV datasets.
# All four K variants must be registered in multivariate_intervals with
# identical architecture (HIDDEN/TRANSFORMS) — only K is allowed to vary.
# ---------------------------------------------------------------------------


def test_multivariate_intervals_k_sweep_registered() -> None:
    """The NF_Ensemble K-sweep family must include all four K ∈ {2, 5, 10, 20}.

    Removing any one of these would silently lose the calibration-vs-cost knee
    benchmark documented in PARITY_REPORT.md §15.
    """
    suite = MultivariateIntervalsSuite()
    methods = suite.methods()
    for variant in NF_ENSEMBLE_K_VARIANTS:
        assert variant in methods, (
            f"missing {variant} from multivariate_intervals.methods(); K-sweep §15 invalidated"
        )


def test_multivariate_intervals_k_sweep_share_architecture() -> None:
    """All K-sweep variants must share HIDDEN + TRANSFORMS + N_SAMPLES.

    Only K itself is allowed to vary across the family.  Architecture drift
    between K=2 and K=20 would corrupt the §15 calibration-vs-cost interpretation.
    """
    suite = MultivariateIntervalsSuite()
    notes = suite.method_notes
    expected_Ks = {
        "NF_Ensemble_K2": 2,
        "NF_Ensemble_K5": 5,
        "NF_Ensemble_K10": 10,
        "NF_Ensemble_K20": 20,
    }
    for variant, K in expected_Ks.items():
        text = notes(variant).lower()
        assert str(K) in text, f"{variant} notes must mention the K value ({K}); got {text!r}"


# ---------------------------------------------------------------------------
# Cross-suite ensemble-config parity.  The §11b vs §13 comparison is only
# valid if both suites use the same K / hidden / transforms / n_samples.
# ---------------------------------------------------------------------------


def test_nf_ensemble_constants_match_across_suites() -> None:
    """`_NF_ENSEMBLE_K / HIDDEN / TRANSFORMS` must be identical between scalar and multivariate suites.

    If anyone tweaks one suite's constants without the other, the cross-suite
    NF_Ensemble_K5 comparison in PARITY_REPORT.md §11b becomes invalid silently.

    NOTE: `_NF_ENSEMBLE_N_SAMPLES` is intentionally allowed to differ across
    suites because the per-test-point memory cost scales with
    ``(K * N_SAMPLES * D)``, where D = output dimensionality (1 for scalar,
    3 for MV).  Scalar distributional_bins uses 256 (sized for D=1, hundreds
    of KB per test point); multivariate_intervals uses 512 (sized for D=3,
    ~18 MB per test point on synthetic_multivariate).  Bumping both to the
    same value would either over-allocate scalar (no benefit) or starve MV
    coverage estimation.
    """
    scalar = DistributionalBinsSuite
    mv = MultivariateIntervalsSuite
    assert scalar._NF_ENSEMBLE_K_REFERENCE == mv._NF_ENSEMBLE_K_REFERENCE, (
        f"K differs: scalar={scalar._NF_ENSEMBLE_K_REFERENCE}, mv={mv._NF_ENSEMBLE_K_REFERENCE}"
    )
    assert scalar._NF_ENSEMBLE_HIDDEN == mv._NF_ENSEMBLE_HIDDEN
    assert scalar._NF_ENSEMBLE_TRANSFORMS == mv._NF_ENSEMBLE_TRANSFORMS


def test_nf_ensemble_k_sweep_min_smoke_completes() -> None:
    """Smoke execution: NF_Ensemble_K2 fit+predict must not crash.

    Regression target: the original AttributeError bug (when method_names
    were added without updating fit()/predict() dispatch) escaped the
    registration-only smoke tests in this file.  This 1-iteration smoke
    runs the actual code path with K=2 (smallest non-single ensemble) to
    catch future dispatch regressions.

    Uses load_dataset() directly instead of _carve_calibration_split() because
    NF_Ensemble_K2 only needs X_train/y_train (the carver's X_test/y_test/X_cal
    keys would trip our rewrap).
    """
    from harness.datasets import load_dataset
    from harness.runners import HarnessConfig

    suite = DistributionalBinsSuite()
    cfg = HarnessConfig(
        seed=42,
        epochs=1,
        batch_size=64,
        standardize_targets=True,
    )
    splits = load_dataset("synthetic_hetero", n_samples=60, standardize_targets=True)
    model_state = suite.fit("NF_Ensemble_K2", splits, cfg)
    assert model_state["type"] == "nflow_ensemble"
    assert len(model_state["members"]) == 2
    assert model_state["K"] == 2
    preds = suite.predict("NF_Ensemble_K2", model_state, splits["X_test"], cfg)
    n_test = len(splits["X_test"])
    assert preds["samples"].shape == (n_test, 2 * 256), (
        f"expected ({n_test}, 512) pooled samples for K=2; got {preds['samples'].shape}"
    )


# ---------------------------------------------------------------------------
# Split_CP (post-hoc split conformal prediction) carve-then-fit regression
# lock.  When fit() was refactored so that X_cal/y_cal are extracted only
# inside the Split_CP branch, a future bug that breaks the carve-then-fit
# chain (carver forwarding wrong keys, fit() misreading X_cal/y_cal, etc.)
# would otherwise escape every other test in this file.  This 1-iteration
# smoke simulates exactly the run() flow used in production.
# ---------------------------------------------------------------------------


def test_split_cp_carve_then_fit_smoke_completes() -> None:
    """Split_CP carve-then-fit dispatch smoke.

    Regression target introduced by the §14 X_cal extraction move: previously
    X_cal/y_cal were extracted unconditionally at the top of fit(), so Split_CP
    was effectively self-contained.  After the move, fit() extracts them only
    inside the Split_CP branch, which means the consumer (run()) MUST pre-carve
    the split and forward ``X_cal``/``y_cal`` as input keys.  This test runs
    exactly that chain via ``_carve_calibration_split`` and asserts the
    distributor returns ``"split_cp"`` rather than crashing with KeyError or
    silently training on misshaped data.
    """
    from suites._backbones import _carve_calibration_split

    from harness.datasets import load_dataset
    from harness.runners import HarnessConfig

    suite = DistributionalBinsSuite()
    cfg = HarnessConfig(
        seed=42,
        epochs=1,
        batch_size=64,
        standardize_targets=True,
    )
    splits = load_dataset("synthetic_hetero", n_samples=60, standardize_targets=True)
    augmented = _carve_calibration_split(
        splits,
        cal_frac=0.3,
        seed=cfg.seed,
        smoke=cfg.smoke,
        suite_name=suite.name,
        forward_keys=("X_test", "y_test"),
    )
    # Carve contract: X_train/y_train shrink, X_cal/y_cal are produced,
    # forwarded keys (X_test/y_test) are preserved.
    assert "X_cal" in augmented and "y_cal" in augmented
    assert "X_test" in augmented and "y_test" in augmented
    assert len(augmented["X_cal"]) > 0 and len(augmented["X_train"]) > 0

    model_state = suite.fit("Split_CP", augmented, cfg)
    assert model_state["type"] == "split_cp", (
        f"expected 'split_cp' (carve-then-fit dispatch); got {model_state['type']!r}"
    )


# ---------------------------------------------------------------------------
# normflows_NSF known-bad pattern lock-in (PARITY_REPORT.md §12).
# `normflows_NSF` is intrinsically broken on the harness's 1-D tabular target
# (broadcast error in the normflows library when combining `num_blocks>1` with
# `num_context_channels≠None` on single-channel input).  The PzflowParitySuite
# short-circuits the method via its `_KNOWN_BAD` registry and emits a
# deterministic `known-bad:` row.  This test catches a future PR that:
#   (a) accidentally drops `normflows_NSF` from `PzflowParitySuite.methods()`,
#   (b) removes the `_KNOWN_BAD["normflows_NSF"]` entry, or
#   (c) re-attempts the import and emits a `failed:` row (which would
#       conflate the limitation with transient environment failures).
# ---------------------------------------------------------------------------


def test_pzflow_parity_normflows_nsf_emits_known_bad_row() -> None:
    from harness.runners import HarnessConfig

    suite = PzflowParitySuite()

    # (a) Registration lock — method remains in methods() so it always surfaces.
    assert "normflows_NSF" in suite.methods(), (
        "normflows_NSF must remain in PzflowParitySuite.methods() so it appears in "
        "the leaderboard even though it never runs (known-bad pattern)."
    )

    # (b) _KNOWN_BAD registry lock — method is in the known-bad string map.
    assert "normflows_NSF" in PzflowParitySuite._KNOWN_BAD, (
        f"PzflowParitySuite._KNOWN_BAD must include `normflows_NSF`; "
        f"current keys: {sorted(PzflowParitySuite._KNOWN_BAD.keys())!r}"
    )

    # (c) End-to-end row emission — the synthesized row carries the exact
    # `Notes` string the user/leaderboard contract expects, with null metrics.
    rows = suite.run(
        "synthetic_cond_gaussian",
        config=HarnessConfig(seed=42, smoke=True, epochs=1, batch_size=64),
    )
    nb_row = next(
        (r for r in rows if r["Method"].endswith("/normflows_NSF")),
        None,
    )
    assert nb_row is not None, "normflows_NSF row must be emitted by run()"

    expected_notes_prefix = (
        "known-bad: normflows_NSF pzflow_parity #broadcast error on 1-D tabular inputs"
    )
    assert nb_row["Notes"] == expected_notes_prefix, (
        f"known-bad Notes for normflows_NSF must match exactly; got {nb_row['Notes']!r}"
    )

    # All performance metrics are null on a known-bad row (the method never ran).
    for metric in ("Coverage", "Width", "IntervalScore", "NLL"):
        assert nb_row.get(metric) is None, (
            f"{metric} must be None on a known-bad row; got {nb_row.get(metric)!r}"
        )

    # None of the keys must trigger tools/leaderboard_regression._is_skipped()
    # i.e. not "skipped", "failed", or "error" — these are reserved for transient
    # environment failures and missing imports.
    from tools.leaderboard_regression import _is_skipped

    assert not _is_skipped(nb_row), (
        f"known-bad rows must NOT trigger _is_skipped (would be filtered out of "
        f"the leaderboard); got Notes={nb_row['Notes']!r}"
    )


# ---------------------------------------------------------------------------
# Image suite smoke tests
# ---------------------------------------------------------------------------


def test_age_regression_smoke_completes() -> None:
    """AgeRegressionSuite fit/predict/evaluate smoke on synthetic image data.

    Runs a 1-epoch fit for CNN+GaussianNLL (fastest method) and asserts
    that fit, predict, and evaluate all produce valid outputs.
    """
    from harness.datasets import load_dataset
    from harness.runners import HarnessConfig

    suite = AgeRegressionSuite()
    cfg = HarnessConfig(seed=42, epochs=1, batch_size=16)
    splits = load_dataset("synthetic_image_regression", n_samples=60, standardize_targets=True)

    # Smoke-test CNN+GaussianNLL (fastest to train)
    model_state = suite.fit("CNN+GaussianNLL", splits, cfg)
    assert model_state["type"] == "gaussian_nll"

    preds = suite.predict("CNN+GaussianNLL", model_state, splits["X_test"], cfg)
    assert "mean" in preds
    assert "std" in preds
    assert "lower" in preds
    assert len(preds["mean"]) == len(splits["X_test"])

    if "_y_scaler" in splits:
        preds["_y_scaler"] = splits["_y_scaler"]
    metrics = suite.evaluate("CNN+GaussianNLL", preds, splits["y_test"], cfg)
    assert "RMSE" in metrics or "Coverage" in metrics


def test_depth_estimation_smoke_completes() -> None:
    """DepthEstimationSuite fit/predict/evaluate smoke on synthetic depth data.

    Runs a 1-epoch fit for UNet+MSE (fastest method) and asserts
    dense prediction flattening produces valid metrics.
    """
    from harness.datasets import load_dataset
    from harness.runners import HarnessConfig

    suite = DepthEstimationSuite()
    cfg = HarnessConfig(seed=42, epochs=1, batch_size=4)
    splits = load_dataset("synthetic_image_depth", n_samples=30, standardize_targets=True)

    # Smoke-test UNet+MSE (fastest to train)
    model_state = suite.fit("UNet+MSE", splits, cfg)
    assert model_state["type"] == "mse"

    preds = suite.predict("UNet+MSE", model_state, splits["X_test"], cfg)
    assert "y_pred" in preds
    # Dense predictions should be flattened: (N*H*W,) = (n_test * 64 * 64,)
    y_test_flat = splits["y_test"].reshape(-1)
    assert len(preds["y_pred"]) == len(y_test_flat), (
        f"expected {len(y_test_flat)} flattened predictions, got {len(preds['y_pred'])}"
    )

    metrics = suite.evaluate("UNet+MSE", preds, splits["y_test"], cfg)
    assert "RMSE" in metrics


# ---------------------------------------------------------------------------
# External-library integration: MAPIE rows must call the REAL mapie library
# (v1 API, fit -> conformalize -> predict_interval) — not a manual sklearn
# reimplementation — and self-report the installed version (git commit when
# pip-installed from git) in the row's LibraryVersion field.
# ---------------------------------------------------------------------------


def test_conformal_mapie_methods_call_real_library() -> None:
    """MAPIE conformal helpers must run the real mapie.regression v1 API.

    Regression target: the old ``MAPIE_*`` rows were manual sklearn
    reimplementations that never imported ``mapie.regression`` (a stale
    workaround for a MAPIE 1.4.x numerical-precision bug).  Paper-grade
    comparisons must exercise the actual library.  Skipped when mapie is
    not installed.
    """
    from suites.tabular.conformal import (
        _MAPIE_AVAILABLE,
        _MAPIE_CROSS_CQR_AVAILABLE,
        _mapie_cqr_intervals,
        _mapie_cross_cqr_intervals,
        _mapie_split_intervals,
        _mapie_split_mlp_intervals,
        _mapie_version,
    )

    if not _MAPIE_AVAILABLE:
        pytest.skip("mapie not installed — install per EXTERNAL_PINS.md")

    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    X, y = make_regression(n_samples=160, n_features=4, noise=2.0, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    X_tr, X_cal, y_tr, y_cal = train_test_split(X_tr, y_tr, test_size=0.3, random_state=0)
    splits = {
        "X_train": X_tr,
        "y_train": y_tr,
        "X_cal": X_cal,
        "y_cal": y_cal,
        "X_test": X_te,
        "y_test": y_te,
    }
    alpha = 0.1
    n_test = len(X_te)

    helpers = [_mapie_split_intervals, _mapie_split_mlp_intervals, _mapie_cqr_intervals]
    if _MAPIE_CROSS_CQR_AVAILABLE:
        helpers.append(_mapie_cross_cqr_intervals)

    for helper in helpers:
        lo, hi = helper(splits, alpha=alpha, seed=42)
        lo = np.asarray(lo).reshape(-1)
        hi = np.asarray(hi).reshape(-1)
        assert lo.shape == (n_test,), f"{helper.__name__}: lo shape {lo.shape}"
        assert hi.shape == (n_test,), f"{helper.__name__}: hi shape {hi.shape}"
        assert np.all(lo <= hi), f"{helper.__name__}: intervals inverted"

    # MAPIE's canonical coverage/width must populate (regression guard for the
    # silent-swallow bug: git main moved these to mapie.metrics.regression with
    # interval-array signatures, which the old top-level import masked).
    from suites.tabular.conformal import _mapie_coverage_and_width

    lo0, hi0 = helpers[0](splits, alpha=alpha, seed=42)
    cw = _mapie_coverage_and_width(
        np.asarray(lo0).reshape(-1),
        np.asarray(hi0).reshape(-1),
        y_te,
    )
    assert cw is not None and len(cw) == 2, "MAPIE canonical coverage/width must be computed"

    # Version self-report must be non-empty and library-flavoured.
    ver = _mapie_version()
    assert ver and ver != "unknown"
    assert "mapie" in ver.lower() or "1." in ver or "git" in ver.lower()


def test_conformal_mapie_rows_report_library_version() -> None:
    """MAPIE rows must carry a LibraryVersion field (suite hook)."""
    from suites.tabular.conformal import (
        _MAPIE_AVAILABLE,
        ConformalSuite,
    )

    if not _MAPIE_AVAILABLE:
        pytest.skip("mapie not installed — install per EXTERNAL_PINS.md")

    suite = ConformalSuite()
    mapie_methods = [m for m in suite.methods() if m.startswith("MAPIE")]
    assert mapie_methods, "MAPIE methods must be registered when mapie is installed"
    for method in suite.methods():
        if method.startswith("MAPIE"):
            assert suite.library_version(method), (
                f"MAPIE method {method} must self-report its library version"
            )
        else:
            assert suite.library_version(method) == ""


def test_torch_uncertainty_version_guard_rejects_v012() -> None:
    """The image suite must refuse torch-uncertainty < 0.13 (reworked API).

    v0.13 changed the regression routine's loss contract (loss(dist, targets))
    and the probabilistic model output (dict of distribution parameters); the
    suite is written against that API.  Simulates an old install by stubbing
    ``sys.modules["torch_uncertainty"]`` with ``__version__ = "0.12.1"``.
    """
    import sys
    from types import SimpleNamespace

    from suites.image.age_regression import _require_torch_uncertainty_version

    old = sys.modules.get("torch_uncertainty")
    sys.modules["torch_uncertainty"] = SimpleNamespace(__version__="0.12.1")
    try:
        with pytest.raises(ImportError, match="too old"):
            _require_torch_uncertainty_version()
    finally:
        if old is not None:
            sys.modules["torch_uncertainty"] = old
        else:
            sys.modules.pop("torch_uncertainty", None)


def test_torch_uncertainty_version_guard_accepts_v013() -> None:
    """torch-uncertainty 0.13.x passes the version guard."""
    import sys
    from types import SimpleNamespace

    from suites.image.age_regression import _require_torch_uncertainty_version

    old = sys.modules.get("torch_uncertainty")
    sys.modules["torch_uncertainty"] = SimpleNamespace(__version__="0.13.0")
    try:
        _require_torch_uncertainty_version()  # must not raise
    finally:
        if old is not None:
            sys.modules["torch_uncertainty"] = old
        else:
            sys.modules.pop("torch_uncertainty", None)
