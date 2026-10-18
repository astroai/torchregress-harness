"""Conditional invertible-network (cINN / FrEIA) parity on real tabular data.

Compares conditional density estimators for scalar regression:

* ``GaussianNLL`` — MLP mean/log-variance head trained with Gaussian NLL.
* ``NSF`` — conditional neural spline flow (zuko) trained with NLL.
* ``FrEIA_cINN`` — conditional invertible neural network built with
  `FrEIA <https://github.com/vislearn/FrEIA>`_ (Ardizzone et al.), i.e. the
  framework behind the original cINN papers, trained with exact NLL.

All three methods model ``p(y | x)``; predictions are posterior samples, so the
harness reports point accuracy, NLL, CRPS, interval coverage and PIT
calibration on identical train/test splits.  Real datasets only (sklearn
diabetes, California housing, and OpenML YearPredictionMSD where available);
no optional ``sbi``/``sbibm`` dependency is required — only ``FrEIA``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _MLP, _hidden_size

_METHODS = ["GaussianNLL", "NSF", "FrEIA_cINN"]
_N_SAMPLES = 256


class FreiaParitySuite(HarnessRunner):
    """Conditional density estimation with FrEIA/cINN vs torchregress flows."""

    @property
    def name(self) -> str:
        return "freia_parity"

    @property
    def description(self) -> str:
        return "FrEIA conditional INN vs torchregress NSF/Gaussian NLL on real tabular data"

    @property
    def supported_datasets(self) -> list[str]:
        return ["diabetes", "california_housing", "openml_year"]

    def methods(self) -> list[str]:
        return list(_METHODS)

    def library_name(self, method: str) -> str:
        return "FrEIA" if method == "FrEIA_cINN" else "torchregress"

    def library_version(self, method: str) -> str:
        if method == "FrEIA_cINN":
            try:
                import importlib.metadata as md

                return md.version("FrEIA")
            except Exception:
                return "unknown"
        import torchregress

        return str(torchregress.__version__)

    def method_notes(self, method: str) -> str:
        notes = {
            "GaussianNLL": "MLP mean/log-var head, Gaussian NLL (heteroscedastic baseline)",
            "NSF": "conditional neural spline flow, exact NLL (torchregress/zuko)",
            "FrEIA_cINN": "conditional INN from FrEIA (AllInOneBlock), exact NLL",
        }
        return notes.get(method, "")

    def _epochs(self, config: HarnessConfig, cap: int = 5) -> int:
        return min(int(config.epochs), cap) if config.smoke else int(config.epochs)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _fit_gaussian(
        self, X: torch.Tensor, y: torch.Tensor, config: HarnessConfig
    ) -> dict[str, Any]:
        from torchregress.losses import GaussianNLLLoss

        device = current_device()
        hidden = 32 if config.smoke else _hidden_size(int(X.shape[1]))
        model = _MLP(in_dim=int(X.shape[1]), hidden=hidden, out_dim=2).to(device)
        loss_fn = GaussianNLLLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        generator = torch.Generator().manual_seed(int(config.seed) + 1)
        batch_size = min(int(config.batch_size), len(X))
        for _ in range(self._epochs(config)):
            order = torch.randperm(len(X), generator=generator).to(device)
            for start in range(0, len(X), batch_size):
                idx = order[start : start + batch_size]
                out = model(X[idx])
                # Concatenated [mean, log_var] output is the documented
                # GaussianNLLLoss contract (target shaped [B, 1]).
                loss = loss_fn(out, y[idx].reshape(-1, 1))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        model.eval()
        return {"type": "gaussian", "model": model}

    def _fit_nsf(self, X: torch.Tensor, y: torch.Tensor, config: HarnessConfig) -> dict[str, Any]:
        from torchregress.losses import NormalizingFlowLoss, create_flow_model

        device = current_device()
        hidden = 32 if config.smoke else 64
        encoder = _MLP(in_dim=int(X.shape[1]), hidden=hidden, out_dim=hidden).to(device)
        flow = create_flow_model(
            n_features=1,
            context_dim=hidden,
            flow_type="nsf",
            n_transforms=2 if config.smoke else 3,
            hidden_features=(hidden, hidden),
            bins=8 if config.smoke else 16,
        ).to(device)
        loss_fn = NormalizingFlowLoss(flow=flow, reduction="mean").to(device)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(loss_fn.parameters()), lr=config.lr
        )
        generator = torch.Generator().manual_seed(int(config.seed) + 3)
        batch_size = min(int(config.batch_size), len(X))
        for _ in range(self._epochs(config)):
            order = torch.randperm(len(X), generator=generator).to(device)
            for start in range(0, len(X), batch_size):
                idx = order[start : start + batch_size]
                loss = loss_fn(encoder(X[idx]), y[idx].reshape(-1, 1))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        encoder.eval()
        loss_fn.eval()
        return {"type": "nsf", "encoder": encoder, "loss_fn": loss_fn}

    @staticmethod
    def _subnet(hidden: int):
        def subnet(ch_in: int, ch_out: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(ch_in, hidden),
                nn.ReLU(),
                nn.Linear(hidden, ch_out),
            )

        return subnet

    def _fit_cinn(self, X: torch.Tensor, y: torch.Tensor, config: HarnessConfig) -> dict[str, Any]:
        try:
            import FrEIA.framework as Ff
            import FrEIA.modules as Fm
        except ImportError as exc:  # reported as a skipped row by HarnessRunner.run
            raise ImportError(
                "FrEIA is not installed; install with `pip install FrEIA` (optional extra 'invertible')"
            ) from exc

        device = current_device()
        hidden = 32 if config.smoke else 64
        context_dim = hidden
        encoder = _MLP(in_dim=int(X.shape[1]), hidden=hidden, out_dim=context_dim).to(device)
        inn = Ff.SequenceINN(1)
        n_blocks = 2 if config.smoke else 4
        for _ in range(n_blocks):
            inn.append(
                Fm.AllInOneBlock,
                cond=0,
                cond_shape=(context_dim,),
                subnet_constructor=self._subnet(hidden),
                permute_soft=True,
            )
        inn = inn.to(device)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(inn.parameters()), lr=config.lr
        )
        generator = torch.Generator().manual_seed(int(config.seed) + 5)
        batch_size = min(int(config.batch_size), len(X))
        for _ in range(self._epochs(config, cap=8)):
            order = torch.randperm(len(X), generator=generator).to(device)
            for start in range(0, len(X), batch_size):
                idx = order[start : start + batch_size]
                context = encoder(X[idx])
                z, log_jac = inn(y[idx].reshape(-1, 1), c=[context])
                nll = (0.5 * z.pow(2).sum(dim=1) - log_jac).mean()
                optimizer.zero_grad(set_to_none=True)
                nll.backward()
                optimizer.step()
        encoder.eval()
        inn.eval()
        return {"type": "cinn", "encoder": encoder, "inn": inn, "context_dim": context_dim}

    def fit(
        self,
        method: str,
        splits: dict[str, Any],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)
        device = current_device()
        X = torch.as_tensor(splits["X_train"], dtype=torch.float32, device=device)
        y = torch.as_tensor(splits["y_train"], dtype=torch.float32, device=device).reshape(-1)
        if method == "GaussianNLL":
            return self._fit_gaussian(X, y, config)
        if method == "NSF":
            return self._fit_nsf(X, y, config)
        if method == "FrEIA_cINN":
            return self._fit_cinn(X, y, config)
        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: "Any",
        config: HarnessConfig,
    ) -> dict[str, Any]:
        device = current_device()
        import numpy as np

        X = torch.as_tensor(np.asarray(X_test), dtype=torch.float32, device=device)
        n_samples = 64 if config.smoke else _N_SAMPLES
        kind = model_state["type"]
        with torch.no_grad():
            if kind == "gaussian":
                out = model_state["model"](X)
                mean = out[:, 0]
                std = torch.exp(0.5 * out[:, 1]).clamp_min(1e-8)
                noise = torch.randn(
                    len(X),
                    n_samples,
                    generator=torch.Generator().manual_seed(int(config.seed)),
                ).to(device)
                samples = mean.unsqueeze(-1) + std.unsqueeze(-1) * noise
                return {
                    "y_pred": mean.detach().cpu().numpy(),
                    "mean": mean.detach().cpu().numpy(),
                    "std": std.detach().cpu().numpy(),
                }
            if kind == "nsf":
                context = model_state["encoder"](X)
                samples = model_state["loss_fn"].sample(context, n_samples=n_samples)
                samples = samples.reshape(len(X), n_samples, 1)
            elif kind == "cinn":
                encoder = model_state["encoder"]
                inn = model_state["inn"]
                context = encoder(X)
                z = torch.randn(
                    len(X),
                    n_samples,
                    1,
                    generator=torch.Generator().manual_seed(int(config.seed)),
                ).to(device)
                context_tiled = context.unsqueeze(1).expand(-1, n_samples, -1)
                flat = inn(
                    z.reshape(-1, 1),
                    c=[context_tiled.reshape(-1, context.shape[-1])],
                    rev=True,
                )[0]
                samples = flat.reshape(len(X), n_samples, 1)
            else:
                raise ValueError(f"Unknown method kind: {kind}")
        samples_np = samples.detach().cpu().numpy()
        if samples_np.ndim == 3 and samples_np.shape[-1] == 1:
            samples_np = samples_np[..., 0]  # harness metrics expect [N, S]
        return {
            "y_pred": samples_np.mean(axis=1).reshape(-1),
            "mean": samples_np.mean(axis=1).reshape(-1),
            "std": samples_np.std(axis=1).reshape(-1),
            "samples": samples_np,
        }


__all__ = ["FreiaParitySuite"]
