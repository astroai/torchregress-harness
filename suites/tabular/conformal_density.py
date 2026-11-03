"""Conformalized conditional densities: raw samples vs CTI vs Mondrian vs WALDO.

Two base density estimators share an MLP encoder: an NSF trained by NLL
(``NormalizingFlowLoss``) and a flow trained with ``ContrastiveFlowLoss`` in its
likelihood-ratio regime (``temperature=1``, ``margin=0``; in-batch negatives).
Each base model is calibrated on a held-out split and evaluated on a clean test
split by:

* raw sample 5/95% intervals (marginal, per output dimension);
* split-conformal density level sets ``{y : -log p(y|x) <= q}``:
  ``ConformalizedDensity`` for the NSF, ``ContrastiveFlowConformal`` (the
  library's wrapper around the contrastive flow's positive density) for the
  contrastive flow, marginal and Mondrian (target-mean terciles, oracle groups);
* WALDO joint regions from the density's own samples (``WaldoDensityConformal``,
  full covariance), with a global critical value or a simulator-free
  critical-value regressor (``WaldoCriticalValueRegressor``).

For multi-target data the level-set and WALDO rows report *joint* region
coverage, the raw rows marginal per-dimension coverage (``CoverageKind``).

Datasets: ``cd_scalar`` / ``cd_multi`` (2-D bimodal synthetic designs from
``tools/conformal_density_bench.py``) and ``synthetic_multitarget_big`` (1M
training rows, 6 targets).  Training streams mini-batches from host memory to
:func:`harness.devices.current_device`; calibration/test sets are bounded
(50k / 20k rows on the big dataset) and scored in chunks, so nothing holds the
full dataset on the GPU.  Rows report ``train_s`` (base model fit),
``eval_s`` (that calibrator's calibrate + predict) and, on CUDA, the peak GPU
memory of that phase.
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any

import numpy as np
import torch
from tools.conformal_density_bench import _data, _Encoder

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites.tabular._scale import (
    BIG_BATCH,
    iter_chunks,
    peak_memory_mb,
    reset_peak_memory,
    to_device,
)

_METHODS = (
    "NSF",
    "NSF_CTI",
    "NSF_CTI_Mondrian",
    "NSF_WALDO",
    "NSF_WALDO_QR",
    "ContrastiveFlow",
    "ContrastiveFlow_CTI",
    "ContrastiveFlow_CTI_Mondrian",
    "ContrastiveFlow_WALDO",
    "ContrastiveFlow_WALDO_QR",
)
_ALPHA = 0.1
_EPOCHS = 80
_BATCH = 256
_HIDDEN = 64
_N_FIT, _N_CAL, _N_TEST = 3000, 1200, 2000
# synthetic_multitarget_big: calibration / test rows and epoch cap.
_BIG_N_CAL, _BIG_N_TEST, _BIG_EPOCHS = 50_000, 20_000, 10
_N_RAW_SAMPLES = 256
_N_WALDO_SAMPLES = 128
_N_INTERVAL_ROWS = 100  # scalar interval widths are grid searches, one row at a time
_SAMPLE_CHUNK = 1024


def _streamed_train(
    enc: torch.nn.Module,
    loss_fn: Any,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    epochs: int,
    batch: int,
    contrastive: bool,
    seed: int,
) -> None:
    """Adam(1e-3) on host-resident data, one mini-batch on the device at a time."""
    params = list(enc.parameters()) + list(loss_fn.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    rng = np.random.default_rng(seed)
    enc.train()
    loss_fn.train()
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for start in range(0, len(order), batch):
            idx = np.sort(order[start : start + batch])
            ctx = enc(to_device(X[idx]))
            yb = to_device(Y[idx])
            if contrastive:
                loss = loss_fn(ctx, yb, negative_context=ctx.roll(1, dims=0).unsqueeze(1))
            else:
                loss = loss_fn(ctx, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    enc.eval()
    loss_fn.eval()


@torch.no_grad()
def _encode(enc: torch.nn.Module, X: np.ndarray) -> torch.Tensor:
    """Encoder contexts of a bounded (calibration / test) split, chunked."""
    return torch.cat([enc(to_device(X[s])) for s in iter_chunks(len(X))])


def _chunked_sampler(loss_fn: Any) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """``sample_fn(context, n) -> [N, n, D]`` drawing in row chunks."""

    @torch.no_grad()
    def sample(context: torch.Tensor, n: int) -> torch.Tensor:
        return torch.cat(
            [
                loss_fn.sample(context[s], n_samples=n)
                for s in iter_chunks(len(context), _SAMPLE_CHUNK)
            ]
        )

    return sample


def _chunked_neg_log_density(
    loss_fn: Any,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    @torch.no_grad()
    def neg_log_density(context: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [-loss_fn.log_prob(context[s], target[s]) for s in iter_chunks(len(context))]
        )

    return neg_log_density


def _q_gap(a: Any, b: Any) -> float:
    """Max |difference| between two (possibly per-group dict) conformal thresholds."""
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)) or set(a) != set(b):
            return float("inf")
        return max(_q_gap(a[k], b[k]) for k in a)
    return float((torch.as_tensor(a).double() - torch.as_tensor(b).double()).abs().max())


@torch.no_grad()
def _raw_metrics(loss_fn: Any, ctx: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """NLL, mean marginal CRPS and raw 5/95% sample intervals, in sample chunks."""
    from torchregress.metrics import crps_from_samples

    nll, crps, cover, width = [], [], [], []
    for s in iter_chunks(len(ctx), _SAMPLE_CHUNK):
        c, yb = ctx[s], y[s]
        nll.append(-loss_fn.log_prob(c, yb).reshape(-1))
        samples = loss_fn.sample(c, n_samples=_N_RAW_SAMPLES)  # [B, S, D]
        lo = samples.quantile(0.05, dim=1)
        hi = samples.quantile(0.95, dim=1)
        cover.append(((yb >= lo) & (yb <= hi)).double().reshape(-1))
        width.append((hi - lo).double().reshape(-1))
        crps.append(
            torch.as_tensor(
                crps_from_samples(samples.transpose(0, 1), yb, reduction="none")
            ).reshape(-1)
        )
    return {
        "NLL": float(torch.cat(nll).double().mean()),
        "CRPS": float(torch.cat(crps).double().mean()),
        "Coverage90": float(torch.cat(cover).mean()),
        "Width90": float(torch.cat(width).mean()),
    }


class ConformalDensitySuite(HarnessRunner):
    @property
    def name(self) -> str:
        return "conformal_density"

    @property
    def description(self) -> str:
        return (
            "Conditional densities (scalar, multi-target, 1M-row multi-target): raw samples vs "
            "ConformalizedDensity / ContrastiveFlowConformal level sets (marginal, Mondrian) vs "
            "WaldoDensityConformal joint regions"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return ["cd_scalar", "cd_multi", "synthetic_multitarget_big"]

    def methods(self) -> list[str]:
        return list(_METHODS)

    def library_name(self, method: str) -> str:
        return "torchregress"

    def library_version(self, method: str) -> str:
        import torchregress

        return str(torchregress.__version__)

    def method_notes(self, method: str) -> str:
        return {
            "NSF": "raw sample 5/95% intervals (marginal per dimension) from an NSF density",
            "NSF_CTI": "ConformalizedDensity split-conformal level set (-log p), marginal",
            "NSF_CTI_Mondrian": "ConformalizedDensity with target-mean tercile Mondrian groups",
            "NSF_WALDO": "WaldoDensityConformal joint region from NSF samples, global threshold",
            "NSF_WALDO_QR": "WaldoDensityConformal + WaldoCriticalValueRegressor (6 bins)",
            "ContrastiveFlow": "raw sample intervals from the ContrastiveFlowLoss positive density",
            "ContrastiveFlow_CTI": (
                "ContrastiveFlowConformal level set, marginal; CPParityGap = |q_hat - "
                "ConformalizedDensity q_hat on the same log_prob| (0 expected)"
            ),
            "ContrastiveFlow_CTI_Mondrian": (
                "ContrastiveFlowConformal with target-mean tercile Mondrian groups; CPParityGap "
                "as for ContrastiveFlow_CTI, per group"
            ),
            "ContrastiveFlow_WALDO": (
                "WaldoDensityConformal joint region from ContrastiveFlow samples"
            ),
            "ContrastiveFlow_WALDO_QR": "ContrastiveFlow WALDO + critical-value regressor",
        }[method]

    def fit(self, method: str, splits: dict[str, Any], config: HarnessConfig) -> dict[str, Any]:
        del method, splits, config
        raise NotImplementedError("conformal_density uses custom run()")

    def predict(self, method: str, model_state: Any, X_test: np.ndarray, config: HarnessConfig):
        raise NotImplementedError("conformal_density uses custom run()")

    def evaluate(
        self, method: str, model_state: Any, splits: dict[str, Any], config: HarnessConfig
    ) -> dict[str, Any]:
        raise NotImplementedError("conformal_density uses custom run()")

    # ------------------------------------------------------------------

    @staticmethod
    def _splits(dataset_name: str, splits: Any, cfg: HarnessConfig) -> tuple[np.ndarray, ...]:
        """(x_fit, y_fit, x_cal, y_cal, x_test, y_test), y always [N, D] float32.

        Registry datasets are standardized per target dimension on the
        training split; calibration rows are a seeded carve of the training
        split (never used for fitting).
        """
        seed = int(cfg.seed)
        if dataset_name in ("cd_scalar", "cd_multi"):
            multi = dataset_name == "cd_multi"
            parts: list[np.ndarray] = []
            for n, offset in ((_N_FIT, 0), (_N_CAL, 1000), (_N_TEST, 2000)):
                x, y = _data(n, multi, seed + offset)
                parts += [x, y.reshape(len(y), -1)]
            return tuple(parts)
        if splits is None:
            from harness.datasets import load_dataset

            splits = load_dataset(dataset_name)
        X = np.asarray(splits["X_train"], dtype=np.float32)
        Y = np.asarray(splits["y_train"], dtype=np.float32)
        Xte = np.asarray(splits["X_test"], dtype=np.float32)
        Yte = np.asarray(splits["y_test"], dtype=np.float32)
        if Y.ndim == 1:
            Y, Yte = Y[:, None], Yte[:, None]
        mu, sd = Y.mean(0), Y.std(0) + 1e-8
        Y, Yte = (Y - mu) / sd, (Yte - mu) / sd
        rng = np.random.default_rng(seed)
        n_cal = min(_BIG_N_CAL, len(X) // 4)
        n_test = min(_BIG_N_TEST, len(Xte))
        if cfg.smoke:
            n_cal, n_test = min(n_cal, 256), min(n_test, 128)
        perm = rng.permutation(len(X))
        cal, fit = np.sort(perm[:n_cal]), np.sort(perm[n_cal:])
        if cfg.smoke:
            fit = fit[:1024]
        test = np.sort(rng.permutation(len(Xte))[:n_test])
        return X[fit], Y[fit], X[cal], Y[cal], Xte[test], Yte[test]

    def _row(self, common: dict[str, Any], method: str, **metrics: Any) -> dict[str, Any]:
        return {**common, "Method": f"{self.name}/{method}", **metrics}

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: dict[str, np.ndarray] | None = None,
        *,
        scenario_name: str | None = None,
    ) -> list[dict[str, Any]]:
        del scenario_name
        cfg = config or HarnessConfig()
        seed = int(cfg.seed)
        x_fit, y_fit, x_cal, y_cal, x_test, y_test = self._splits(dataset_name, splits, cfg)
        big = dataset_name == "synthetic_multitarget_big"
        y_dim = y_fit.shape[1]
        if cfg.smoke:
            epochs = 1 if big else 3
        else:
            epochs = min(int(cfg.epochs or _EPOCHS), _BIG_EPOCHS if big else _EPOCHS)
        batch = max(_BATCH, BIG_BATCH) if big else _BATCH
        n_iv = min(_N_INTERVAL_ROWS, len(x_test))
        # Mondrian groups: target-mean terciles (oracle groups, as in the bench tool).
        key_cal = y_cal.mean(1)
        cuts = np.quantile(key_cal, [0.33, 0.67])
        groups_cal = torch.as_tensor(np.digitize(key_cal, cuts), device=current_device())
        groups_test = torch.as_tensor(np.digitize(y_test.mean(1), cuts), device=current_device())

        rows: list[dict[str, Any]] = []
        for base_name, contrastive in (("NSF", False), ("ContrastiveFlow", True)):
            names = [m for m in _METHODS if m.split("_", 1)[0] == base_name]
            common: dict[str, Any] = {
                "Library": "torchregress",
                "LibraryVersion": self.library_version(base_name),
                "Dataset": dataset_name,
                "Suite": self.name,
                "NTargets": y_dim,
                "NTrain": int(len(x_fit)),
                "NCal": int(len(x_cal)),
                "NTest": int(len(x_test)),
                "Epochs": epochs,
                "Device": str(current_device()),
                "RMSE": None,
            }
            set_harness_seed(seed)
            try:
                from torchregress.losses import (
                    ContrastiveFlowLoss,
                    NormalizingFlowLoss,
                    create_flow_model,
                )

                reset_peak_memory()
                t0 = perf_counter()
                enc = _Encoder(x_fit.shape[1], _HIDDEN).to(current_device())
                flow = create_flow_model(
                    n_features=y_dim,
                    context_dim=_HIDDEN,
                    flow_type="nsf",
                    n_transforms=3,
                    hidden_features=[_HIDDEN, _HIDDEN],
                ).to(current_device())
                loss_fn: Any = (
                    ContrastiveFlowLoss(flow=flow, temperature=1.0, margin=0.0)
                    if contrastive
                    else NormalizingFlowLoss(flow=flow)
                )
                _streamed_train(
                    enc,
                    loss_fn,
                    x_fit,
                    y_fit,
                    epochs=epochs,
                    batch=batch,
                    contrastive=contrastive,
                    seed=seed,
                )
                train_s = perf_counter() - t0
                train_peak = peak_memory_mb()
                ctx_cal, ctx_test = _encode(enc, x_cal), _encode(enc, x_test)
                y_cal_t, y_test_t = to_device(y_cal), to_device(y_test)
                t1 = perf_counter()
                raw = _raw_metrics(loss_fn, ctx_test, y_test_t)
                raw_s = perf_counter() - t1
            except Exception as exc:  # noqa: BLE001 — row-level resilience
                note = f"failed: {type(exc).__name__}: {exc}"
                rows += [
                    self._row(common, m, Coverage90=None, train_s=None, eval_s=None, Notes=note)
                    for m in names
                ]
                continue
            common.update({"NLL": raw["NLL"], "CRPS": raw["CRPS"], "train_s": train_s})
            rows.append(
                self._row(
                    common,
                    base_name,
                    Coverage90=raw["Coverage90"],
                    CoverageKind="marginal",
                    Width90=raw["Width90"],
                    eval_s=raw_s,
                    PeakGPUMemMB=train_peak,
                    Notes=self.method_notes(base_name),
                )
            )
            for method in names[1:]:
                set_harness_seed(seed)
                try:
                    reset_peak_memory()
                    t1 = perf_counter()
                    if "_CTI" in method:
                        mondrian = method.endswith("_Mondrian")
                        metrics = self._level_set(
                            loss_fn,
                            contrastive,
                            ctx_cal,
                            y_cal_t,
                            ctx_test,
                            y_test_t,
                            groups_cal if mondrian else None,
                            groups_test if mondrian else None,
                            n_iv,
                        )
                    else:
                        metrics = self._waldo(
                            loss_fn,
                            ctx_cal,
                            y_cal_t,
                            ctx_test,
                            y_test_t,
                            n_iv,
                            regressor=method.endswith("_QR"),
                        )
                    rows.append(
                        self._row(
                            common,
                            method,
                            **metrics,
                            CoverageKind="joint" if y_dim > 1 else "scalar",
                            eval_s=perf_counter() - t1,
                            PeakGPUMemMB=peak_memory_mb(),
                            Notes=self.method_notes(method),
                        )
                    )
                except Exception as exc:  # noqa: BLE001 — row-level resilience
                    rows.append(
                        self._row(
                            common,
                            method,
                            Coverage90=None,
                            eval_s=None,
                            Notes=f"failed: {type(exc).__name__}: {exc}",
                        )
                    )
        return rows

    @staticmethod
    def _level_set(
        loss_fn: Any,
        contrastive: bool,
        ctx_cal: torch.Tensor,
        y_cal: torch.Tensor,
        ctx_test: torch.Tensor,
        y_test: torch.Tensor,
        groups_cal: torch.Tensor | None,
        groups_test: torch.Tensor | None,
        n_iv: int,
    ) -> dict[str, Any]:
        """Split-conformal density level set; ContrastiveFlowConformal for the contrastive flow."""
        from torchregress.losses import ConformalizedDensity, ContrastiveFlowConformal

        neg_log_density = _chunked_neg_log_density(loss_fn)
        out: dict[str, Any] = {}
        lo_y, hi_y = float(y_test.min()) - 4.0, float(y_test.max()) + 4.0
        g_iv = None if groups_test is None else groups_test[:n_iv]
        with torch.no_grad():
            if contrastive:
                cf = ContrastiveFlowConformal(loss_fn, alpha=_ALPHA, grid_size=100)
                cf.calibrate(ctx_cal, y_cal, groups=groups_cal)
                inside = cf.contains(ctx_test, y_test, groups=groups_test)
                # The same score through the generic class: the wrapper must not move q_hat.
                ref = ConformalizedDensity(alpha=_ALPHA, grid_size=100)
                ref.calibrate(neg_log_density, ctx_cal, y_cal, groups=groups_cal)
                out["CPParityGap"] = _q_gap(cf.cti.q_hat, ref.cti.q_hat)

                def interval() -> tuple[torch.Tensor, torch.Tensor]:
                    return cf.predict_interval(ctx_test[:n_iv], lo_y, hi_y, groups=g_iv)
            else:
                cd = ConformalizedDensity(alpha=_ALPHA, grid_size=100)
                cd.calibrate(neg_log_density, ctx_cal, y_cal, groups=groups_cal)
                inside = cd.contains(neg_log_density, ctx_test, y_test, groups=groups_test)

                def interval() -> tuple[torch.Tensor, torch.Tensor]:
                    return cd.predict_interval(
                        neg_log_density, ctx_test[:n_iv], lo_y, hi_y, groups=g_iv
                    )

        out["Coverage90"] = float(inside.double().mean())
        if y_test.shape[1] == 1:
            with torch.no_grad():
                lo, hi = interval()
            out["Width90"] = float((hi - lo).mean())
        else:
            out["Width90"] = None
        return out

    @staticmethod
    def _waldo(
        loss_fn: Any,
        ctx_cal: torch.Tensor,
        y_cal: torch.Tensor,
        ctx_test: torch.Tensor,
        y_test: torch.Tensor,
        n_iv: int,
        *,
        regressor: bool,
    ) -> dict[str, Any]:
        """WaldoDensityConformal joint regions from the density's own samples."""
        from torchregress.losses import WaldoCriticalValueRegressor, WaldoDensityConformal

        reg = (
            WaldoCriticalValueRegressor(alpha=_ALPHA, n_bins=6, min_samples_per_bin=10)
            if regressor
            else None
        )
        cp = WaldoDensityConformal(
            alpha=_ALPHA,
            covariance_type="full",
            n_samples=_N_WALDO_SAMPLES,
            critical_value_regressor=reg,
        )
        sample_fn = _chunked_sampler(loss_fn)
        with torch.no_grad():
            cp.calibrate(sample_fn, ctx_cal, y_cal)
            inside = torch.cat(
                [cp.contains(sample_fn, ctx_test[s], y_test[s]) for s in iter_chunks(len(ctx_test))]
            )
        threshold = cp.threshold
        out: dict[str, Any] = {
            "Coverage90": float(inside.double().mean()),
            "WaldoThreshold": None if threshold is None else float(threshold),
        }
        if y_test.shape[1] == 1:
            with torch.no_grad():
                lo, hi = cp.predict_interval(
                    sample_fn,
                    ctx_test[:n_iv],
                    y_min=float(y_test.min()) - 4.0,
                    y_max=float(y_test.max()) + 4.0,
                )
            out["Width90"] = float((hi - lo).mean())
        else:
            out["Width90"] = None
        return out
