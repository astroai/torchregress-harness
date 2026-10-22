"""Known-label-noise and multi-target conditional-density comparison.

This suite is the modern-density companion to ``density_parity``.  It uses a
controlled synthetic problem with a known heteroscedastic label-noise model so
that the clean and noise-aware objectives can be compared on a **clean test
set**:

- clean / noise-aware NSF (Gaussian-Hermite marginal likelihood),
- clean / analytic-convolution MDN,
- clean / Gaussian-smoothed binned-CDF head,
- conditional flow matching (sampling-only, no exact likelihood).

Targets are scalar (``synthetic_na_scalar``) or two-dimensional
(``synthetic_na_multi``); the latter exercises joint density and correlation
metrics.  All methods share an MLP context encoder and the harness training
budget.
"""

from __future__ import annotations

import os
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner, SkipMethod
from suites._backbones import _MLP, _hidden_size

_METHODS = [
    "NSF",
    "NSF_noise",
    "MDN",
    "MDN_noise",
    "Binned",
    "Binned_noise",
    "Binned_joint",
    "Binned_joint_noise",
    "CFM",
    "NFM",
]
_N_TRAIN, _N_TEST = 5000, 2000
_SMOKE_TRAIN, _SMOKE_TEST = 512, 128


def _make_splits(name: str, config: HarnessConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(config.seed))
    n_train, n_test = (_SMOKE_TRAIN, _SMOKE_TEST) if config.smoke else (_N_TRAIN, _N_TEST)
    multi = name.endswith("multi")

    def draw(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = rng.normal(size=(n, 2)).astype(np.float32)
        mean = np.stack(
            [np.sin(1.2 * x[:, 0]) + 0.5 * x[:, 1], np.cos(1.1 * x[:, 1]) - 0.4 * x[:, 0]],
            axis=1,
        )
        mode = rng.choice([-1.0, 1.0], size=(n, 1) if multi else (n,))
        if multi:
            y = (
                mean
                + mode * np.array([0.8, 0.6], dtype=np.float32)
                + rng.normal(scale=0.35, size=(n, 2))
            )
            sigma = np.stack([0.10 + 0.18 * np.abs(x[:, 0]), 0.12 + 0.14 * np.abs(x[:, 1])], axis=1)
        else:
            y = mean[:, 0] + mode * 0.8 + rng.normal(scale=0.35, size=n)
            sigma = 0.10 + 0.22 * np.abs(x[:, 0])
        return x, y.astype(np.float32), sigma.astype(np.float32)

    x_tr, y_clean, sigma = draw(n_train)
    x_te, y_te, _ = draw(n_test)
    z = rng.normal(size=y_clean.shape if multi else y_clean.shape)
    y_obs = (y_clean + sigma * z).astype(np.float32)
    return {
        "X_train": x_tr,
        "y_train": y_obs,
        "sigma_train": sigma,
        "X_test": x_te,
        "y_test": y_te,
        "_source_url": "synthetic known-noise design in suites/tabular/noise_aware_density.py",
    }


def _train(
    modules: list[nn.Module],
    loss_fn: Any,
    X: np.ndarray,
    Y: np.ndarray,
    sigma: np.ndarray | None,
    cfg: HarnessConfig,
    *,
    epochs: int | None = None,
) -> float:
    dev = current_device()
    xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(Y, dtype=torch.float32, device=dev)
    st = torch.as_tensor(sigma, dtype=torch.float32, device=dev) if sigma is not None else None
    params = [p for m in modules for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=cfg.lr)
    n = len(xt)
    bs = min(int(cfg.batch_size), n)
    epochs = 3 if cfg.smoke else int(epochs if epochs is not None else cfg.epochs)
    for _ in range(epochs):
        for m in modules:
            m.train()
        perm = torch.randperm(n, device=dev)
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(xt[idx], yt[idx], None if st is None else st[idx])
            loss.backward()
            opt.step()
    for m in modules:
        m.eval()
    return float("nan")


def _eval(
    model: nn.Module,
    base_loss: Any,
    kind: str,
    X: np.ndarray,
    y: np.ndarray,
    cfg: HarnessConfig,
) -> dict[str, float | None]:
    dev = current_device()
    xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    from torchregress.losses import NoiseAwareDensityLoss

    if isinstance(base_loss, NoiseAwareDensityLoss):
        base_loss = base_loss.base_loss
    with torch.no_grad():
        ctx = model(xt)
        n_s = 64 if cfg.smoke else 256
        if kind == "nfm":
            nfm = base_loss
            nll = float(-nfm.log_prob(ctx, yt).mean())
            samples = nfm.sample(ctx, n_samples=n_s).cpu().numpy()
        elif kind == "cfm":
            from torchregress.models import sample_cfm

            samples = sample_cfm(base_loss, ctx, n_samples=n_s, steps=32).cpu().numpy()
            nll = None
        elif kind == "flow":
            nll = float(-base_loss.log_prob(ctx, yt).mean())
            samples = base_loss.sample(ctx, n_samples=n_s).cpu().numpy()
        elif kind == "mdn":
            nll = float(base_loss(ctx, yt))
            samples = base_loss.sample(ctx, n_samples=n_s).transpose(0, 1).cpu().numpy()
        elif kind == "binned_joint":
            logits = ctx
            nll = float(-base_loss.log_prob(logits, yt).mean())
            samples = base_loss.sample(logits, n_samples=n_s)
            if samples.shape[0] == n_s:
                samples = samples.transpose(0, 1)
            samples = samples.cpu().numpy()
        elif kind == "binned":
            logits = ctx
            y_idx = (
                torch.searchsorted(base_loss.bin_edges, yt.reshape(-1)).clamp(1, base_loss.n_bins)
                - 1
            )
            nll = float(-torch.log_softmax(logits, -1)[torch.arange(len(yt)), y_idx].mean())
            samples = base_loss.sample(logits, n_samples=n_s)
            if samples.shape[0] == n_s:
                samples = samples.T
            samples = samples.cpu().numpy()
        else:
            raise ValueError(kind)
    y_np = np.asarray(y, dtype=np.float64)
    if y_np.ndim == 1:
        y_np = y_np[:, None]
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim == 2:
        s = s[:, :, None]
    mean = s.mean(axis=1)
    rmse = float(np.sqrt(np.mean((mean - y_np) ** 2)))
    # Marginal CRPS (mean over target dimensions).
    from torchregress.metrics import crps_from_samples

    crps = float(
        np.mean([crps_from_samples(s[:, :, d].T, y_np[:, d]) for d in range(y_np.shape[1])])
    )
    lo = np.quantile(s, 0.05, axis=1)
    hi = np.quantile(s, 0.95, axis=1)
    cov = float(np.mean((y_np >= lo) & (y_np <= hi)))
    pit = np.mean(s <= y_np[:, None, :], axis=1)
    # PIT CvM on the first marginal (paper-style aggregate).
    from torchregress.metrics import pit_cvm

    pit_cvm_val = float(np.mean([pit_cvm(pit[:, d]) for d in range(pit.shape[1])]))
    rare = np.asarray(X)[:, 0] > 1.5
    rare_rmse = (
        float(np.sqrt(np.mean((mean[rare, 0] - y_np[rare, 0]) ** 2))) if rare.any() else None
    )
    return {
        "NLL": nll,
        "RMSE": rmse,
        "CRPS": crps,
        "Coverage90": cov,
        "PITCVM": pit_cvm_val,
        "RareRMSE": rare_rmse,
    }


class NoiseAwareDensitySuite(HarnessRunner):
    @property
    def name(self) -> str:
        return "noise_aware_density"

    @property
    def description(self) -> str:
        return (
            "Known-label-noise conditional densities on clean test labels: "
            "NSF/MDN/binned clean vs noise-aware; CFM sampler; scalar and multi-target"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return ["synthetic_na_scalar", "synthetic_na_multi"]

    def methods(self) -> list[str]:
        requested = os.environ.get("TORCHREGRESS_NOISE_METHODS", "").strip()
        if not requested:
            return list(_METHODS)
        wanted = {part.strip() for part in requested.split(",") if part.strip()}
        return [method for method in _METHODS if method in wanted]

    def library_name(self, method: str) -> str:
        return "torchregress"

    def library_version(self, method: str) -> str:
        import torchregress

        return str(torchregress.__version__)

    def method_notes(self, method: str) -> str:
        return {
            "NSF": "clean NSF on noisy observed labels; no measurement model",
            "NSF_noise": "NSF with Gaussian-Hermite label-noise marginal likelihood",
            "MDN": "clean diagonal MDN on noisy observed labels",
            "MDN_noise": "MDN with analytic Gaussian convolution (sigma^2 added to variance)",
            "Binned": "smoothed binned-CDF head on noisy labels",
            "Binned_noise": "binned-CDF head with Gaussian-smoothed soft targets",
            "CFM": "OT-CFM (32 Euler steps) on noisy labels; sample-based metrics only",
            "Binned_joint": "joint multi-target binned histogram on noisy labels; no deconvolution",
            "Binned_joint_noise": "joint multi-target binned histogram with Gaussian-smoothed soft targets",
            "NFM": "tabular Normalizing Flow Map on noisy labels; exact density, no noise model",
        }[method]

    def fit(self, method: str, splits: dict[str, Any], config: HarnessConfig) -> dict[str, Any]:
        del method, splits, config
        raise NotImplementedError("noise_aware_density uses custom run()")

    def predict(self, method: str, model_state: Any, X_test: np.ndarray, config: HarnessConfig):
        raise NotImplementedError

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ):
        raise NotImplementedError

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: dict[str, np.ndarray] | None = None,
        *,
        scenario_name: str | None = None,
    ):
        del scenario_name
        cfg = config or HarnessConfig()
        data = splits or _make_splits(dataset_name, cfg)
        X = np.asarray(data["X_train"], dtype=np.float32)
        Y = np.asarray(data["y_train"], dtype=np.float32)
        sigma = np.asarray(data["sigma_train"], dtype=np.float32)
        Xte = np.asarray(data["X_test"], dtype=np.float32)
        Yte = np.asarray(data["y_test"], dtype=np.float32)
        if Y.ndim == 1:
            Y, Yte = Y[:, None], Yte[:, None]
        if sigma.ndim == 1:
            sigma = sigma[:, None]
        mu, sd = Y.mean(0), Y.std(0) + 1e-8
        Y = (Y - mu) / sd
        Yte = (Yte - mu) / sd
        # Noise sigma must be in standardized target units.
        sigma = sigma / sd
        multi = Y.shape[1] > 1
        methods = [m for m in self.methods() if not (multi and m in {"Binned", "Binned_noise"})]
        rows: list[dict[str, Any]] = []
        for method in methods:
            set_harness_seed(cfg.seed)
            base = {
                "Method": f"{self.name}/{method}",
                "Library": "torchregress",
                "LibraryVersion": self.library_version(method),
                "Dataset": dataset_name,
                "Suite": self.name,
                "NTargets": Y.shape[1],
                "Device": str(current_device()),
            }
            try:
                t0 = perf_counter()
                modules, loss_fn, kind = self._build(method, X, Y, sigma, cfg)
                _train(
                    modules,
                    loss_fn,
                    X,
                    Y,
                    sigma if method.endswith("_noise") else None,
                    cfg,
                    epochs=None,
                )
                fit_s = perf_counter() - t0
                t1 = perf_counter()
                model, base_loss = modules[0], (modules[1] if len(modules) > 1 else modules[0])
                metrics = _eval(model, base_loss, kind, Xte, Yte, cfg)
                rows.append(
                    {
                        **base,
                        **metrics,
                        "train_s": fit_s,
                        "eval_s": perf_counter() - t1,
                        "Notes": self.method_notes(method),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        **base,
                        "NLL": None,
                        "RMSE": None,
                        "CRPS": None,
                        "Coverage90": None,
                        "PITCVM": None,
                        "RareRMSE": None,
                        "train_s": None,
                        "eval_s": None,
                        "Notes": f"failed: {type(exc).__name__}: {exc}",
                    }
                )
        return rows

    def _build(
        self, method: str, X: np.ndarray, Y: np.ndarray, sigma: np.ndarray, cfg: HarnessConfig
    ):
        from torchregress.losses import (
            BinnedDensityLoss,
            MixtureDensityLoss,
            NoiseAwareDensityLoss,
            NormalizingFlowLoss,
            create_flow_model,
        )
        from torchregress.models import FlowMatchingMLP, NormalizingFlowMap

        dev = current_device()
        hidden = _hidden_size(X.shape[1])
        d = Y.shape[1]
        encoder = _MLP(X.shape[1], hidden, hidden).to(dev)
        if method.startswith("NSF"):
            flow = create_flow_model(
                n_features=d,
                context_dim=hidden,
                flow_type="nsf",
                n_transforms=3,
                hidden_features=[64, 64],
            ).to(dev)
            loss_fn = NormalizingFlowLoss(flow=flow).to(dev)
            if method.endswith("_noise"):
                loss_fn = NoiseAwareDensityLoss(loss_fn, mode="quadrature" if d == 1 else "mc").to(
                    dev
                )

            def train_loss(xb, yb, sig):
                ctx = encoder(xb)
                return loss_fn(ctx, yb, sigma=sig) if sig is not None else loss_fn(ctx, yb)

            return [encoder, loss_fn], train_loss, "flow"
        if method.startswith("MDN"):
            mdn = MixtureDensityLoss(n_components=5, n_features=d).to(dev)
            head = nn.Linear(hidden, mdn.expected_output_size).to(dev)
            loss_fn = (
                NoiseAwareDensityLoss(mdn, mode="analytic") if method.endswith("_noise") else mdn
            )
            if method.endswith("_noise"):
                loss_fn = loss_fn.to(dev)

            class _MDNModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = encoder
                    self.head = head

                def forward(self, x):
                    return self.head(self.encoder(x))

            model = _MDNModel().to(dev)

            def train_loss(xb, yb, sig):
                out = model(xb)
                return loss_fn(out, yb, sigma=sig) if sig is not None else loss_fn(out, yb)

            return [model, mdn], train_loss, "mdn"
        if method.startswith("Binned_joint"):
            from torchregress.losses import JointBinnedDensityLoss

            if d < 2:
                raise SkipMethod("JointBinnedDensityLoss requires a multi-target dataset")
            k = 25 if d == 2 else 15
            edges = torch.linspace(-6.0, 6.0, k + 1)
            binned_joint = JointBinnedDensityLoss([edges] * d, max_joint_bins=4096).to(dev)
            head_joint = nn.Linear(hidden, binned_joint.joint_bins).to(dev)

            class _JointBinnedModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = encoder
                    self.head = head_joint

                def forward(self, x):
                    return self.head(self.encoder(x))

            model_joint = _JointBinnedModel().to(dev)
            if method.endswith("_noise"):

                def train_loss(xb, yb, sig):
                    return binned_joint(model_joint(xb), yb, sigma=sig)
            else:

                def train_loss(xb, yb, sig):
                    del sig
                    return binned_joint(model_joint(xb), yb)

            return [model_joint, binned_joint], train_loss, "binned_joint"
        if method.startswith("Binned"):
            edges = torch.linspace(-6.0, 6.0, 65)
            binned = BinnedDensityLoss(edges).to(dev)
            head = nn.Linear(hidden, binned.n_bins).to(dev)

            class _BinnedModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = encoder
                    self.head = head

                def forward(self, x):
                    return self.head(self.encoder(x))

            model = _BinnedModel().to(dev)
            if method.endswith("_noise"):

                def train_loss(xb, yb, sig):
                    return binned(model(xb), yb, sigma=sig)
            else:

                def train_loss(xb, yb, sig):
                    del sig
                    return binned(model(xb), yb)

            # For evaluation, return the binned loss and the logits model.
            return [model, binned], train_loss, "binned"
        if method == "NFM":
            nfm = NormalizingFlowMap(
                d,
                hidden,
                n_layers=3,
                hidden=(64, 64),
                arch="coupling",
                nll_weight=1.0,
            ).to(dev)

            class _NFMModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = encoder
                    self.nfm = nfm

                def forward(self, x):
                    return self.encoder(x)

            model = _NFMModel().to(dev)

            def train_loss(xb, yb, sig):
                del sig
                # Endpoint NLL is the fair density objective and fits the
                # tabular map faster than the mixed map-matching loss at the
                # harness budget.
                return -nfm.log_prob(model(xb), yb).mean()

            return [model, nfm], train_loss, "nfm"
        if method == "CFM":
            velocity = FlowMatchingMLP(d, hidden, hidden=(64, 64)).to(dev)

            class _CFMModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = encoder
                    self.velocity = velocity

                def forward(self, x):
                    return self.encoder(x)

            cfm = _CFMModel().to(dev)

            def train_loss(xb, yb, sig):
                del sig
                from torchregress.models import cfm_loss

                return cfm_loss(velocity, cfm(xb), yb)

            return [cfm, velocity], train_loss, "cfm"
        raise ValueError(method)
