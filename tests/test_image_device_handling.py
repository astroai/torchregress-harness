"""Device-handling regressions from release run #6 (CUDA node).

* age_regression ``CNN+RankNContrast``: the lstsq linear probe mixed a CPU
  design matrix with targets left on ``current_device()`` (cuda).
* keypoint_regression ``*+GaussianNLL``: ``np.exp`` was applied to a device
  tensor, which implicitly calls ``Tensor.__array__`` and fails on cuda.

No GPU here, so the tests guard the mechanism instead: lstsq operands must
share one (CPU) device, and ``predict`` must never implicitly convert a tensor
to numpy (only explicit ``.detach().cpu().numpy()``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from suites.image.age_regression import AgeRegressionSuite
from suites.image.keypoint_regression import KeypointRegressionSuite

from harness.datasets import load_dataset
from harness.runners import HarnessConfig

_CFG = HarnessConfig(seed=0, smoke=True, epochs=1)


def test_rank_n_contrast_lstsq_operands_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    real = torch.linalg.lstsq
    seen: list[tuple[torch.device, torch.device]] = []

    def checked(A: torch.Tensor, B: torch.Tensor, *a, **k):
        seen.append((A.device, B.device))
        assert A.device == B.device == torch.device("cpu")
        return real(A, B, *a, **k)

    monkeypatch.setattr(torch.linalg, "lstsq", checked)
    suite = AgeRegressionSuite()
    splits = load_dataset("synthetic_image_regression", n_samples=120)
    state = suite.fit("CNN+RankNContrast", splits, _CFG)
    assert seen, "lstsq probe not exercised"
    pred = suite.predict("CNN+RankNContrast", state, splits["X_test"], _CFG)
    assert np.isfinite(np.asarray(pred["y_pred"])).all()


@pytest.mark.parametrize("method", ["CNN+GaussianNLL", "CNN+MSE"])
def test_keypoint_predict_has_no_implicit_tensor_to_numpy(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = KeypointRegressionSuite()
    splits = load_dataset("synthetic_image_keypoint", n_samples=60)
    state = suite.fit(method, splits, _CFG)

    def no_implicit(self, *a, **k):  # what cuda tensors effectively do
        raise TypeError("implicit tensor -> numpy conversion (fails on cuda)")

    monkeypatch.setattr(torch.Tensor, "__array__", no_implicit)
    out = suite.predict(method, state, splits["X_test"], _CFG)
    for v in out.values():
        assert isinstance(v, np.ndarray) and np.isfinite(v).all()
