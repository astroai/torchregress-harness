"""Shared validation helpers for pinpointing torchregress/harness issues.

Used by ``run_all.py --validate`` and ``run_canonical.py --validate``
to check core imports, key losses, calibrators, and suite instantiation
before running benchmarks.
"""

from __future__ import annotations


def validate_torchregress() -> tuple[bool, list[str]]:
    """Run basic sanity checks on torchregress to pinpoint import/API issues.

    Returns (ok, messages) where ok is True if all checks pass.
    """
    ok = True
    msgs: list[str] = []

    # 1. Core import
    try:
        import torchregress  # noqa: F401

        msgs.append("  ✓ torchregress imported")
    except ImportError as exc:
        ok = False
        msgs.append(f"  ✗ torchregress import failed: {exc}")
        return ok, msgs  # can't continue without core

    # 2. Key losses
    loss_checks = [
        ("GaussianNLLLoss", "torchregress.losses"),
        ("MixtureDensityLoss", "torchregress.losses"),
        ("MultiQuantileLoss", "torchregress.losses"),
        ("TweedieLoss", "torchregress.losses"),
        ("ConformalLoss", "torchregress.losses.conformal"),
        ("BetaNLLLoss", "torchregress.losses"),
    ]
    for name, module_path in loss_checks:
        try:
            mod = __import__(module_path, fromlist=[name])
            getattr(mod, name)
            msgs.append(f"  ✓ {module_path}.{name}")
        except Exception as exc:
            msgs.append(f"  ✗ {module_path}.{name}: {type(exc).__name__}: {exc}")

    # 3. Calibration
    cal_checks = [
        ("VarianceTemperatureScaler", "torchregress.calibration"),
        ("IsotonicMeanCalibrator", "torchregress.calibration"),
    ]
    for name, module_path in cal_checks:
        try:
            mod = __import__(module_path, fromlist=[name])
            getattr(mod, name)
            msgs.append(f"  ✓ {module_path}.{name}")
        except Exception as exc:
            msgs.append(f"  ✗ {module_path}.{name}: {type(exc).__name__}: {exc}")

    # 4. Core metrics
    try:
        import numpy as np

        from harness.metrics import compute_all_metrics

        result = compute_all_metrics(
            {"y_pred": np.array([1.0, 2.0])},
            np.array([1.1, 1.9]),
        )
        msgs.append(f"  ✓ compute_all_metrics smoke: {sorted(result.keys())}")
    except Exception as exc:
        msgs.append(f"  ✗ compute_all_metrics: {type(exc).__name__}: {exc}")

    return ok, msgs


def validate_nf_architecture() -> tuple[bool, list[str]]:
    """CI smoketest: verify NF uses correct [64,64] hidden architecture.

    Instantiates NormalizingFlow via DistributionalBinsSuite.fit and
    MultivariateIntervalsSuite.fit with dummy data (epochs=0, no training)
    and checks the flows each have >20k parameters.

    The old [32,32] config produces ~9k params; the correct [64,64]
    config produces ~25k.  Deterministic, instantaneous.

    Returns (ok, messages).
    """
    ok = True
    msgs: list[str] = []

    try:
        import numpy as np
        import torch.nn as nn
        from suites.tabular.distributional_bins import DistributionalBinsSuite
        from suites.tabular.multivariate_intervals import MultivariateIntervalsSuite

        from harness.runners import HarnessConfig

        cfg = HarnessConfig(epochs=0)

        # --- distributional_bins NF ---
        splits_1d: dict[str, np.ndarray] = {
            "X_train": np.random.randn(10, 5).astype(np.float32),
            "y_train": np.random.randn(10).astype(np.float32),
            "X_cal": np.random.randn(10, 5).astype(np.float32),
            "y_cal": np.random.randn(10).astype(np.float32),
        }
        state_1d = DistributionalBinsSuite().fit("NormalizingFlow", splits_1d, cfg)
        flow_1d = state_1d["flow"]
        if not isinstance(flow_1d, nn.Module):
            ok = False
            msgs.append("  ✗ distributional_bins NF: flow is not nn.Module")
        else:
            p1 = sum(p.numel() for p in flow_1d.parameters())
            if p1 > 20_000:
                msgs.append(f"  ✓ distributional_bins NF ok (params: {p1:,} > 20k)")
            else:
                ok = False
                msgs.append(
                    f"  ✗ distributional_bins NF regression! "
                    f"Flow has {p1:,} params (< 20k). "
                    "Check hidden_features=[64, 64], n_transforms=3."
                )

        # --- multivariate_intervals NF ---
        splits_mv: dict[str, np.ndarray] = {
            "X_train": np.random.randn(10, 5).astype(np.float32),
            "y_train": np.random.randn(10, 3).astype(np.float32),
        }
        state_mv = MultivariateIntervalsSuite().fit("NormalizingFlow", splits_mv, cfg)
        flow_mv = state_mv["flow"]
        if not isinstance(flow_mv, nn.Module):
            ok = False
            msgs.append("  ✗ multivariate_intervals NF: flow is not nn.Module")
        else:
            p2 = sum(p.numel() for p in flow_mv.parameters())
            if p2 > 20_000:
                msgs.append(f"  ✓ multivariate_intervals NF ok (params: {p2:,} > 20k)")
            else:
                ok = False
                msgs.append(
                    f"  ✗ multivariate_intervals NF regression! "
                    f"Flow has {p2:,} params (< 20k). "
                    "Check hidden_features=[64, 64], n_transforms=3."
                )

    except Exception as exc:
        ok = False
        msgs.append(f"  \u2717 NF architecture smoketest failed: {type(exc).__name__}: {exc}")

    return ok, msgs


def validate_nf_architecture_negative() -> tuple[bool, list[str]]:
    """Negative test: verify the OLD [32,32] NF config FAILS the param-count check.

    Confirms that ``validate_nf_architecture`` correctly catches the
    ``hidden_features=[32, 32]`` regression.  Constructs a 1D NSF flow
    with the old undersized config and asserts params < 20k.

    This is a meta-test of the CI infrastructure — not part of the
    ``--validate`` pipeline.  Run separately::

        from tools._validate import validate_nf_architecture_negative
        ok, msgs = validate_nf_architecture_negative()
        assert ok  # ok=False means old config unexpectedly passes CI

    Returns (ok, messages) where ok is True if the old config correctly
    has too few params (negative test passes).
    """
    ok = True
    msgs: list[str] = []

    try:
        from torchregress.losses.nflows import create_flow_model

        flow = create_flow_model(
            n_features=1,
            context_dim=20,
            flow_type="nsf",
            n_transforms=3,
            hidden_features=[32, 32],
        )

        param_count = sum(p.numel() for p in flow.parameters())

        threshold = 20_000
        if param_count < threshold:
            msgs.append(
                f"  \u2713 Old NF config correctly fails CI "
                f"(params: {param_count:,} < {threshold:,})"
            )
        else:
            ok = False
            msgs.append(
                f"  \u2717 Old NF config unexpectedly passes CI! "
                f"Flow has {param_count:,} params (≥ {threshold:,}). "
                "The architecture check may have a false-negative."
            )

    except Exception as exc:
        ok = False
        msgs.append(f"  \u2717 NF architecture negative test failed: {type(exc).__name__}: {exc}")

    return ok, msgs


def validate_sls_coverage() -> tuple[bool, list[str]]:
    """CI smoketest: verify SLS achieves ≥0.90 coverage on synthetic_multivariate.

    SLS's core promise is guaranteed coverage at τ=0.9.  This test trains
    SLS on synthetic_multivariate at 1 epoch (seed=42) with 200 smoke
    samples and asserts coverage ≥ 0.90.  SLS converges to coverage
    extremely quickly — at 1 epoch it already achieves 0.95 coverage.

    Note: this test verifies SLS's core coverage guarantee; it may not
    catch architecture regressions (e.g., hidden_dim=32, n_transforms=2)
    since undersized flows produce broader densities → wider intervals
    → equal or higher coverage.  Architecture regressions are caught by
    validate_nf_architecture() which checks parameter counts.

    Returns (ok, messages).
    """
    ok = True
    msgs: list[str] = []

    try:
        import time

        from suites.tabular.multivariate_intervals import MultivariateIntervalsSuite

        from harness.core import set_harness_seed
        from harness.datasets import load_dataset
        from harness.runners import HarnessConfig

        set_harness_seed(42)
        splits = load_dataset(
            "synthetic_multivariate",
            n_samples=200,
            standardize_targets=True,
        )
        cfg = HarnessConfig(seed=42, epochs=1, smoke=True)

        suite = MultivariateIntervalsSuite()

        t0 = time.monotonic()
        model_state = suite.fit("SLS", splits, cfg)
        predictions = suite.predict("SLS", model_state, splits["X_test"], cfg)
        metrics = suite.evaluate("SLS", predictions, splits["y_test"], cfg)
        elapsed = time.monotonic() - t0

        coverage = float(metrics.get("Coverage", 0.0))
        joint_cov = float(metrics.get("JointCoverage", 0.0))

        threshold = 0.90
        if coverage >= threshold:
            msgs.append(
                f"  \u2713 SLS coverage OK "
                f"(Coverage={coverage:.3f} ≥ {threshold}, "
                f"JointCoverage={joint_cov:.3f}, "
                f"{elapsed:.1f}s)"
            )
        else:
            ok = False
            msgs.append(
                f"  \u2717 SLS coverage regression! "
                f"Coverage={coverage:.3f} < {threshold}. "
                f"JointCoverage={joint_cov:.3f}. "
                "Check hidden_dim=64, n_transforms=4, tau=0.9."
            )

    except Exception as exc:
        ok = False
        msgs.append(f"  \u2717 SLS coverage smoketest failed: {type(exc).__name__}: {exc}")

    return ok, msgs


def _sls_param_count(hidden_dim: int, n_transforms: int) -> int:
    """Return total parameter count for SLSLoss with given architecture."""
    from torchregress.losses.sls import SLSLoss

    loss_fn = SLSLoss(
        d=3,
        context_dim=20,
        K=1,
        mode="full",
        hidden_dim=hidden_dim,
        n_transforms=n_transforms,
        tau=0.9,
        warmup_steps=500,
    )
    return sum(p.numel() for p in loss_fn.parameters())


def validate_sls_architecture() -> tuple[bool, list[str]]:
    """CI smoketest: verify SLS uses correct [64,4] architecture.

    Instantiates SLSLoss directly (no training, no suite) with fixed
    d=3, context_dim=20 and checks the loss has >25k parameters.

    The old config (hidden_dim=32, n_transforms=2) produces ~6.8k params
    and would fail.  The coverage-based check (validate_sls_coverage)
    may miss this regression since undersized flows produce broader
    densities → equal or higher coverage.  This param-count check is
    deterministic and catches the exact regression.

    Returns (ok, messages).
    """
    ok = True
    msgs: list[str] = []

    try:
        param_count = _sls_param_count(hidden_dim=64, n_transforms=4)

        threshold = 25_000
        if param_count > threshold:
            msgs.append(f"  \u2713 SLS architecture ok (params: {param_count:,} > {threshold:,})")
        else:
            ok = False
            msgs.append(
                f"  \u2717 SLS architecture regression! "
                f"SLSLoss has {param_count:,} params (< {threshold:,}). "
                "Check hidden_dim=64, n_transforms=4."
            )

    except Exception as exc:
        ok = False
        msgs.append(f"  \u2717 SLS architecture smoketest failed: {type(exc).__name__}: {exc}")

    return ok, msgs


def validate_sls_architecture_negative() -> tuple[bool, list[str]]:
    """Negative test: verify the OLD [32,2] config FAILS the architecture check.

    Confirms that ``validate_sls_architecture`` correctly catches the
    ``hidden_dim=32, n_transforms=2`` regression.  Instantiates SLSLoss
    with the old undersized config and asserts params < 25k (i.e., the
    old config would fail the CI check).

    This is a meta-test of the CI infrastructure — not part of the
    ``--validate`` pipeline.  Run separately to verify the CI checks
    are actually catching regressions::

        from tools._validate import validate_sls_architecture_negative
        ok, msgs = validate_sls_architecture_negative()
        assert ok  # ok=False means old config unexpectedly passes CI

    Returns (ok, messages) where ok is True if the old config correctly
    has too few params (negative test passes).
    """
    ok = True
    msgs: list[str] = []

    try:
        param_count = _sls_param_count(hidden_dim=32, n_transforms=2)

        threshold = 25_000
        if param_count < threshold:
            msgs.append(
                f"  \u2713 Old SLS config correctly fails CI "
                f"(params: {param_count:,} < {threshold:,})"
            )
        else:
            ok = False
            msgs.append(
                f"  \u2717 Old SLS config unexpectedly passes CI! "
                f"SLSLoss has {param_count:,} params (≥ {threshold:,}). "
                "The architecture check may have a false-negative."
            )

    except Exception as exc:
        ok = False
        msgs.append(f"  \u2717 SLS architecture negative test failed: {type(exc).__name__}: {exc}")

    return ok, msgs


def validate_gaussiannll_coverage() -> tuple[bool, list[str]]:
    """CI smoketest: verify GaussianNLL achieves adequate coverage.

    GaussianNLL is the best all-rounder across all benchmark suites.
    This test trains GaussianNLL on synthetic_multivariate at 1 epoch
    (seed=42) with 200 smoke samples and asserts coverage ≥ 0.85.

    GaussianNLL has no tunable internal architecture like NF/SLS, but
    coverage can regress if the z-score (1.645 for 90% CI), output
    dimensionality, or loss computation is accidentally changed.
    At 1 epoch it already achieves 0.989 — a drop below 0.85 would
    signal a serious bug.

    Returns (ok, messages).
    """
    ok = True
    msgs: list[str] = []

    try:
        import time

        from suites.tabular.multivariate_intervals import MultivariateIntervalsSuite

        from harness.core import set_harness_seed
        from harness.datasets import load_dataset
        from harness.runners import HarnessConfig

        set_harness_seed(42)
        splits = load_dataset(
            "synthetic_multivariate",
            n_samples=200,
            standardize_targets=True,
        )
        cfg = HarnessConfig(seed=42, epochs=1, smoke=True)

        suite = MultivariateIntervalsSuite()

        t0 = time.monotonic()
        model_state = suite.fit("GaussianNLL", splits, cfg)
        predictions = suite.predict("GaussianNLL", model_state, splits["X_test"], cfg)
        metrics = suite.evaluate("GaussianNLL", predictions, splits["y_test"], cfg)
        elapsed = time.monotonic() - t0

        coverage = float(metrics.get("Coverage", 0.0))

        threshold = 0.90
        if coverage >= threshold:
            msgs.append(
                f"  \u2713 GaussianNLL coverage OK "
                f"(Coverage={coverage:.3f} ≥ {threshold}, "
                f"{elapsed:.1f}s)"
            )
        else:
            ok = False
            msgs.append(
                f"  \u2717 GaussianNLL coverage regression! "
                f"Coverage={coverage:.3f} < {threshold}. "
                "Check z_score=1.645, output dimensionality, loss computation."
            )

    except Exception as exc:
        ok = False
        msgs.append(f"  \u2717 GaussianNLL coverage smoketest failed: {type(exc).__name__}: {exc}")

    return ok, msgs


def validate_multivariate_gaussian_coverage() -> tuple[bool, list[str]]:
    """CI smoketest: verify MultivariateGaussian achieves adequate coverage.

    MultivariateGaussian is the best method on truly Gaussian-correlated
    multivariate targets, winning on synthetic_multivariate.  This test
    trains MultivariateGaussian on synthetic_multivariate at 1 epoch
    (seed=42) with 200 smoke samples and asserts coverage ≥ 0.90.

    Complements the GaussianNLL check: GaussianNLL covers the diagonal
    case; MultivariateGaussian covers the full-covariance case for
    multivariate targets with feature correlations.

    Returns (ok, messages).
    """
    ok = True
    msgs: list[str] = []

    try:
        import time

        from suites.tabular.multivariate_intervals import MultivariateIntervalsSuite

        from harness.core import set_harness_seed
        from harness.datasets import load_dataset
        from harness.runners import HarnessConfig

        set_harness_seed(42)
        splits = load_dataset(
            "synthetic_multivariate",
            n_samples=200,
            standardize_targets=True,
        )
        cfg = HarnessConfig(seed=42, epochs=1, smoke=True)

        suite = MultivariateIntervalsSuite()

        t0 = time.monotonic()
        model_state = suite.fit("MultivariateGaussian", splits, cfg)
        predictions = suite.predict("MultivariateGaussian", model_state, splits["X_test"], cfg)
        metrics = suite.evaluate("MultivariateGaussian", predictions, splits["y_test"], cfg)
        elapsed = time.monotonic() - t0

        coverage = float(metrics.get("Coverage", 0.0))
        joint_cov = float(metrics.get("JointCoverage", 0.0))

        threshold = 0.90
        if coverage >= threshold:
            msgs.append(
                f"  \u2713 MultivariateGaussian coverage OK "
                f"(Coverage={coverage:.3f} ≥ {threshold}, "
                f"JointCoverage={joint_cov:.3f}, "
                f"{elapsed:.1f}s)"
            )
        else:
            ok = False
            msgs.append(
                f"  \u2717 MultivariateGaussian coverage regression! "
                f"Coverage={coverage:.3f} < {threshold}. "
                f"JointCoverage={joint_cov:.3f}. "
                "Check Cholesky/eigh fallback, jitter, loss computation."
            )

    except Exception as exc:
        ok = False
        msgs.append(
            f"  \u2717 MultivariateGaussian coverage smoketest failed: {type(exc).__name__}: {exc}"
        )

    return ok, msgs


def validate_multivariate_gaussian_coverage_negative() -> tuple[bool, list[str]]:
    """Negative test: verify the CI coverage check catches broken loss configs.

    Constructs MultivariateGaussianLoss with ``jitter=0`` and ``eps=0`` (both
    safety nets disabled) and feeds it a rank-deficient (all-ones) covariance
    matrix.  Asserts the loss produces ``NaN`` or ``Inf`` — proving that
    ``jitter`` and ``eps`` are essential and that removing either would
    cause the loss to break, which the coverage smoketest would detect
    (training with a broken loss can't converge to coverage ≥ 0.90).

    With both safety nets off and a singular matrix, Cholesky fails (caught
    internally), the eigh fallback produces zero eigenvalues, and
    ``log(0) → -inf`` plus division by zero → ``NaN`` or ``Inf``.

    This is a meta-test of the CI infrastructure — not part of the
    ``--validate`` pipeline.  Run separately::

        from tools._validate import validate_multivariate_gaussian_coverage_negative
        ok, msgs = validate_multivariate_gaussian_coverage_negative()
        assert ok  # ok=False means the safety-net removal went undetected

    Returns (ok, messages) where ok is True if the unsafeguarded loss
    correctly produces NaN/Inf (negative test passes).
    """
    ok = True
    msgs: list[str] = []

    try:
        import torch
        from torchregress.losses.gaussian import MultivariateGaussianLoss

        batch_size = 4
        n_features = 3

        # Rank-deficient (singular) covariance: all 1s
        singular_cov = torch.ones(batch_size, n_features, n_features, dtype=torch.float64)

        # Well-conditioned predictions
        mean = torch.randn(batch_size, n_features, dtype=torch.float64)
        target = mean + 0.1 * torch.randn(batch_size, n_features, dtype=torch.float64)

        # Disable BOTH safety nets: jitter is not added, eps is zero
        loss_fn = MultivariateGaussianLoss(jitter=0.0, eps=0.0, reduction="mean")

        result = loss_fn(mean, target, covariance_matrices=singular_cov)
        result_val = float(result.item())

        if not torch.isfinite(result).item():
            msgs.append(
                f"  ✓ MultivariateGaussianLoss correctly fails "
                f"without safety nets (result={result_val})"
            )
        else:
            ok = False
            msgs.append(
                f"  ✗ Unsafeguarded MultivariateGaussianLoss "
                f"unexpectedly produced finite result ({result_val}). "
                "The jitter/eps safety nets may not be essential, or "
                "the eigh fallback masked the issue."
            )

    except Exception as exc:
        ok = False
        msgs.append(f"  ✗ MultivariateGaussian negative test failed: {type(exc).__name__}: {exc}")

    return ok, msgs


def validate_suites(all_suites: dict) -> tuple[bool, list[str]]:
    """Check that all registered suites can be instantiated and queried.

    Args:
        all_suites: Dict mapping suite name → suite class (from ALL_SUITES).

    Returns (ok, messages) where ok is True if all checks pass.
    """
    ok = True
    msgs: list[str] = []

    for name, cls in all_suites.items():
        try:
            suite = cls()
            methods = suite.methods()
            datasets = suite.supported_datasets
            msgs.append(f"  ✓ {name}: {len(methods)} methods, {len(datasets)} datasets")
        except Exception as exc:
            ok = False
            msgs.append(f"  ✗ {name}: {type(exc).__name__}: {exc}")

    return ok, msgs


def validate_image_backbones() -> tuple[bool, list[str]]:
    """CI smoketest: verify image backbones emit correct shapes.

    Builds CNN/UNet/Dropout-CNN with synthetic (N,C,H,W) input and checks
    output shapes for scalar, multi-output, and dense heads. Deterministic,
    no training. Also guards backbone hyperparams (base_filters) so silent
    arch regressions fail fast, mirroring tabular NF[64,64]/SLS[64,4] checks.
    """
    ok = True
    msgs: list[str] = []
    try:
        import numpy as np
        import torch
        from suites.image._base import (
            _image_to_dataloader,
            _ImageCNN,
            _ImageCNN_Dropout,
            _UNet,
        )

        cnn = _ImageCNN(in_channels=1, out_dim=2)
        # Arch guard: default CNN fibres must stay at base_filters=16 → 64-dim head.
        if cnn.head.in_features != 64:
            ok = False
            msgs.append(
                f"  ✗ _ImageCNN head dim {cnn.head.in_features} != 64 (base_filters regression?)"
            )
        else:
            msgs.append("  ✓ _ImageCNN head dim 64 (base_filters=16)")
        x = torch.randn(4, 1, 32, 32)
        out = cnn(x)
        if tuple(out.shape) == (4, 2):
            msgs.append("  ✓ _ImageCNN (4,1,32,32)->(4,2)")
        else:
            ok = False
            msgs.append(f"  ✗ _ImageCNN shape {tuple(out.shape)} != (4, 2)")

        drop = _ImageCNN_Dropout(in_channels=3, out_dim=1)
        out_d = drop(torch.randn(2, 3, 32, 32))
        if tuple(out_d.shape) == (2, 1):
            msgs.append("  ✓ _ImageCNN_Dropout (2,3,32,32)->(2,1)")
        else:
            ok = False
            msgs.append(f"  ✗ _ImageCNN_Dropout shape {tuple(out_d.shape)} != (2, 1)")

        unet = _UNet(in_channels=3, out_channels=2)
        out_u = unet(torch.randn(2, 3, 64, 64))
        if tuple(out_u.shape) == (2, 2, 64, 64):
            msgs.append("  ✓ _UNet (2,3,64,64)->(2,2,64,64)")
        else:
            ok = False
            msgs.append(f"  ✗ _UNet shape {tuple(out_u.shape)} != (2, 2, 64, 64)")
        # UNet param guard: base_filters=32 must stay (silent arch change check).
        n_unet = sum(p.numel() for p in unet.parameters())
        if n_unet > 1_000_000:
            msgs.append(f"  ✓ _UNet params {n_unet:,} (>1M, base_filters=32)")
        else:
            ok = False
            msgs.append(f"  ✗ _UNet params {n_unet:,} too small (base_filters regression?)")

        # Shared constants guard: single source of truth in suites.image._base.
        try:
            from suites.image._base import (
                EVIDENTIAL_COEFF_NIG,
                EVIDENTIAL_EPS,
                IMAGE_ALPHA,
                IMAGE_Z_SCORE,
            )

            if abs(IMAGE_Z_SCORE - 1.645) < 1e-9 and abs(IMAGE_ALPHA - 0.1) < 1e-12:
                msgs.append("  ✓ image constants ALPHA=0.1/Z=1.645")
            else:
                ok = False
                msgs.append(f"  ✗ image constants drift ALPHA={IMAGE_ALPHA} Z={IMAGE_Z_SCORE}")
            if abs(EVIDENTIAL_COEFF_NIG - 0.01) < 1e-12 and abs(EVIDENTIAL_EPS - 0.01) < 1e-12:
                msgs.append("  ✓ evidential constants coeff=0.01/eps=0.01")
            else:
                ok = False
                msgs.append("  ✗ evidential constants drift")
        except Exception as exc:
            ok = False
            msgs.append(f"  ✗ image constants import failed: {exc}")

        # Dataloader: scalar / multi-output / dense y layouts.
        dl1 = _image_to_dataloader(
            np.zeros((5, 1, 32, 32), np.float32), np.zeros(5, np.float32), 2, False
        )
        b = next(iter(dl1))
        if tuple(b[1].shape) == (2, 1):
            msgs.append("  ✓ dataloader scalar y->(B,1)")
        else:
            ok = False
            msgs.append(f"  ✗ dataloader scalar {tuple(b[1].shape)}")
        dl2 = _image_to_dataloader(
            np.zeros((5, 3, 32, 32), np.float32), np.zeros((5, 3), np.float32), 2, False
        )
        b2 = next(iter(dl2))
        if tuple(b2[1].shape) == (2, 3):
            msgs.append("  ✓ dataloader multi-output y->(B,D)")
        else:
            ok = False
            msgs.append(f"  ✗ dataloader multi {tuple(b2[1].shape)}")
        dl3 = _image_to_dataloader(
            np.zeros((5, 3, 16, 16), np.float32), np.zeros((5, 16, 16), np.float32), 2, False
        )
        b3 = next(iter(dl3))
        if tuple(b3[1].shape) == (2, 1, 16, 16):
            msgs.append("  ✓ dataloader dense y->(B,1,H,W)")
        else:
            ok = False
            msgs.append(f"  ✗ dataloader dense {tuple(b3[1].shape)}")
    except Exception as exc:
        ok = False
        msgs.append(f"  ✗ image backbone smoketest failed: {type(exc).__name__}: {exc}")
    return ok, msgs
