"""Scale paths of the likelihoods suite: chunked NSF sampling, native units, CUDA graphs."""

from __future__ import annotations

import numpy as np
import pytest
import suites.tabular.likelihoods as lik
import torch
from suites._backbones import _MLP, _train_one
from suites.tabular.likelihoods import LikelihoodsSuite

from harness.devices import set_default_device
from harness.runners import HarnessConfig


def _skewed(n: int = 1500, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    # Raw dollar-like scale far from the z scale: mean ~ 60, sd ~ 15.
    y = (50.0 + 10.0 * np.exp(0.6 * X[:, 0] + 0.2 * rng.normal(size=n))).astype(np.float32)
    k = int(0.8 * n)
    return {"X_train": X[:k], "y_train": y[:k], "X_test": X[k:], "y_test": y[k:]}


def test_nsf_samples_are_native_units_and_row_aligned(monkeypatch: pytest.MonkeyPatch) -> None:
    set_default_device("cpu")
    # Several sampling chunks over the 300 test rows.
    monkeypatch.setattr(lik, "_SAMPLE_CHUNK_DRAWS", 64 * lik._N_SAMPLES)
    suite = LikelihoodsSuite()
    splits = _skewed()
    cfg = HarnessConfig(seed=0, epochs=15)
    state = suite.fit("NSF", splits, cfg)
    pred = suite.predict("NSF", state, splits["X_test"], cfg)
    samples = pred["samples"]
    y = splits["y_test"].astype(np.float64)
    assert samples.shape == (lik._N_SAMPLES, len(y))
    # Native units: the predictive draws live on the target's scale.
    assert abs(samples.mean() - y.mean()) < 0.5 * y.std()
    # Row-aligned: each test row's predictive median tracks its own target.
    assert np.corrcoef(np.median(samples, axis=0), y)[0, 1] > 0.6
    nll = suite._test_nll("NSF", state, pred, splits["X_test"], splits["y_test"])
    assert np.isfinite(nll) and nll > np.log(y.std()) - 1.0  # native-unit density


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@pytest.mark.parametrize("loss_name", ["GaussianNLLLoss", "SkewNormalNLLLoss"])
def test_cuda_graph_training_matches_eager(loss_name: str) -> None:
    import torchregress.losses as losses

    dev = set_default_device("cuda")
    g = torch.Generator().manual_seed(0)
    X = torch.randn(5000, 6, generator=g)
    y = (X[:, :1] + 0.3 * torch.randn(5000, 1, generator=g)).exp()
    out_dim = 2 if loss_name == "GaussianNLLLoss" else 3
    trained = []
    for graph in (False, True):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        model = _MLP(6, 16, out_dim).to(dev)
        used = _train_one(
            model,
            getattr(losses, loss_name)(),
            X,
            y,
            epochs=3,
            batch_size=1024,
            lr=1e-3,
            cuda_graph=graph,
        )
        assert used is graph
        trained.append(torch.cat([p.detach().flatten() for p in model.parameters()]).cpu())
    # Same batches, same Adam update (on-device step counters): float rounding only.
    torch.testing.assert_close(trained[1], trained[0], rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_cuda_graph_falls_back_for_host_synchronising_loss() -> None:
    from torchregress.losses import SkewTLoss

    dev = set_default_device("cuda")
    X = torch.randn(3000, 4)
    y = torch.randn(3000, 1)
    torch.manual_seed(0)
    model = _MLP(4, 16, 4).to(dev)
    # The skew-t CDF's continued fraction checks convergence on the host.
    assert _train_one(model, SkewTLoss(), X, y, epochs=1, batch_size=1024, cuda_graph=True) is False
    assert all(torch.isfinite(p).all() for p in model.parameters())
