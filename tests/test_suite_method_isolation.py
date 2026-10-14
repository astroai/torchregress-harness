"""One failing method must yield a failed row, not abort the whole suite run.

Regression test: the tabular suites caught a fixed tuple of exception types,
so a third-party error outside it (``httpcore.ProxyError`` when TabICL could
not download its checkpoint) escaped ``run()`` and dropped every other
method's row for that dataset.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from harness.runners import HarnessConfig


class _ThirdPartyError(Exception):
    """Stands in for an exception type from an external library."""


def _splits() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3)).astype(np.float32)
    y = (X[:, 0] + 0.1 * rng.normal(size=120)).astype(np.float32)
    return {"X_train": X[:80], "y_train": y[:80], "X_test": X[80:], "y_test": y[80:]}


@pytest.mark.parametrize(
    ("module", "cls", "fit_attr"),
    [
        ("suites.tabular.conformal", "ConformalSuite", "fit"),
        ("suites.tabular.calibration", "CalibrationSuite", "_fit_calibrator"),
        ("suites.tabular.distributional_bins", "DistributionalBinsSuite", "fit"),
    ],
)
def test_failing_method_becomes_failed_row(
    monkeypatch, module: str, cls: str, fit_attr: str
) -> None:
    import importlib

    suite_cls = getattr(importlib.import_module(module), cls)
    suite = suite_cls()
    good, bad = suite.methods()[:1][0], "boom"
    monkeypatch.setattr(suite_cls, "methods", lambda self: [bad, good])
    real_fit = getattr(suite_cls, fit_attr)

    def fit(self, method, *args, **kwargs):
        if method == bad:
            raise _ThirdPartyError("checkpoint download refused")
        return real_fit(self, method, *args, **kwargs)

    monkeypatch.setattr(suite_cls, fit_attr, fit)
    rows = suite.run("unit", config=HarnessConfig(seed=0, smoke=True, epochs=2), splits=_splits())
    # Some suites emit several rows per method (``Split_CP/bin_0`` ...).
    names = [r["Method"].split("/")[1] for r in rows]
    bad_rows = [r for r, n in zip(rows, names, strict=True) if n == bad]
    good_rows = [r for r, n in zip(rows, names, strict=True) if n == good]
    assert bad_rows and all("_ThirdPartyError" in str(r["Notes"]) for r in bad_rows)
    assert good_rows and not any("_ThirdPartyError" in str(r.get("Notes", "")) for r in good_rows)
