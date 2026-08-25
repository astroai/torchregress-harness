"""Regression-as-Classification (R2C) Conformal Prediction suite.

Benchmarks classification-based conformal prediction methods where the
continuous regression target is discretised into ordered bins, a classifier
predicts bin probabilities, and conformal intervals are constructed from
the probability distribution.

Compares:
- torchregress R2CConformal with three backbones: Softmax, CumulativeLink, CORAL
- torchcp's R2CCP (if available)
- Standard Split/MLP and CQR/MLP baselines (for reference)

Methods
-------
- R2C+Softmax           : binned classifier (Softmax CE) + R2CConformal
- R2C+CumulativeLink    : ordinal cumulative logits + R2CConformal
- R2C+CORAL             : CORAL architecture + R2CConformal
- torchcp_R2C           : torchcp R2CCP score + R2ccpLoss + SplitPredictor
- Split+MLP             : Split conformal baseline (MLP + MSE + post-hoc)
- CQR+MLP               : CQR baseline (quantile MLP + post-hoc)

References
----------
.. [1] Guha, E., Natarajan, S., Möllenhoff, T., Khan, M. E., & Ndiaye, E. (2024).
   Conformal Prediction via Regression-as-Classification.
   ICLR 2024. https://arxiv.org/abs/2404.08168
.. [2] Cao, W., Mirjalili, V., & Raschka, S. (2020).
   Rank consistent ordinal regression for neural networks.
   Pattern Recognition Letters, 140, 325-331. https://arxiv.org/abs/1901.07884
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from harness.core import set_harness_seed
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import (
    _MLP,
    _carve_calibration_split,
    _hidden_size,
    _train_one,
)

# ---------------------------------------------------------------------------
# Optional external R2C library (torchcp)
# ---------------------------------------------------------------------------

try:
    from torchcp.regression.score.r2ccp import R2CCP, R2ccpLoss, build_regression_model

    _TORCHCP_R2C_AVAILABLE = True
except ImportError:
    _TORCHCP_R2C_AVAILABLE = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bin_edges(
    y_train: np.ndarray,
    n_bins: int = 50,
    *,
    margin: float = 0.1,
) -> np.ndarray:
    """Create uniform bin edges covering the training target range.

    Args:
        y_train: 1-D array of training targets.
        n_bins: Number of bins.
        margin: Fractional padding beyond min/max.

    Returns:
        1-D array of length ``n_bins + 1``.
    """
    y_min = float(np.min(y_train))
    y_max = float(np.max(y_train))
    span = y_max - y_min
    if span < 1e-8:
        span = 1.0
    lo = y_min - margin * span
    hi = y_max + margin * span
    return np.linspace(lo, hi, n_bins + 1, dtype=np.float32)


def _discretise_targets(y: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Map continuous targets to bin indices 0..n_bins-1."""
    bins = np.digitize(y.reshape(-1), bin_edges[1:-1])
    return bins.astype(np.int64)


def _cumulative_logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Convert K-1 cumulative logits to K bin probabilities.

    Given z₀, z₁, ..., z_{K-2} where zₖ = logit(P(y > k)):

        P(y = 0)   = 1 - σ(z₀)
        P(y = k)   = σ(z_{k-1}) - σ(zₖ)   for k = 1..K-2
        P(y = K-1) = σ(z_{K-2})

    Args:
        logits: (batch, K-1) tensor of cumulative logits.

    Returns:
        (batch, K) tensor of bin probabilities summing to 1.
    """
    sig = torch.sigmoid(logits)  # (batch, K-1)
    one = torch.ones_like(sig[..., :1])
    zero = torch.zeros_like(sig[..., :1])
    padded = torch.cat([one, sig, zero], dim=-1)  # (batch, K+1)
    probs = padded[..., :-1] - padded[..., 1:]  # 1-sig[0], sig[0]-sig[1], ..., sig[K-2]-0
    return probs.clamp(min=1e-8)


def _torch_split_intervals(
    model: nn.Module,
    X_cal: torch.Tensor,
    y_cal: torch.Tensor,
    X_test: torch.Tensor,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Standard split-conformal: absolute residual scores."""
    model.eval()
    with torch.no_grad():
        pred_cal = model(X_cal).squeeze(-1)
        pred_test = model(X_test).squeeze(-1)
    residuals = (y_cal - pred_cal).abs()
    n = residuals.numel()
    q_level = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    q = torch.quantile(residuals, q_level, interpolation="higher").item()
    return (pred_test - q).numpy(), (pred_test + q).numpy()


def _torch_cqr_intervals(
    model: nn.Module,
    X_cal: torch.Tensor,
    y_cal: torch.Tensor,
    X_test: torch.Tensor,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """CQR intervals from quantile predictions."""
    model.eval()
    with torch.no_grad():
        pred_cal = model(X_cal)
        pred_test = model(X_test)
    q_lo_cal, q_hi_cal = pred_cal[:, 0], pred_cal[:, 1]
    q_lo_test, q_hi_test = pred_test[:, 0], pred_test[:, 1]
    score = torch.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    n = score.numel()
    q_level = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    q = torch.quantile(score, q_level, interpolation="higher").item()
    return (q_lo_test - q).numpy(), (q_hi_test + q).numpy()


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class R2CConformalSuite(HarnessRunner):
    """R2C conformal prediction benchmark suite."""

    CALIBRATION_FRAC = 0.25
    N_BINS = 50  # number of target bins for discretisation
    ALPHA = 0.1  # miscoverage rate

    @property
    def name(self) -> str:
        return "r2c_conformal"

    @property
    def description(self) -> str:
        return (
            "Regression-as-Classification conformal prediction: "
            "torchregress R2C (Softmax / CumulativeLink / CORAL) vs torchcp R2CCP"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "california_housing",
            "diabetes",
            "synthetic_hetero",
            "house_sales",
            "openml_year",
        ]

    @property
    def canonical_scenarios(self) -> list[str]:
        return []

    # ------------------------------------------------------------------
    # Method registry
    # ------------------------------------------------------------------

    def methods(self) -> list[str]:
        methods = [
            "R2C+Softmax",
            "R2C+CumulativeLink",
            "R2C+CORAL",
            "Split+MLP",
            "CQR+MLP",
        ]
        if _TORCHCP_R2C_AVAILABLE:
            methods.append("torchcp_R2C")
        return methods

    def library_name(self, method: str) -> str:
        if method.startswith("torchcp"):
            return "torchcp"
        return "torchregress"

    def method_notes(self, method: str) -> str:
        notes: dict[str, str] = {
            "R2C+Softmax": (
                f"MLP({self.N_BINS}-class softmax) + CE → R2CConformal.  "
                "Discretises target into uniform bins, trains classifier, "
                "builds APS-style conformal intervals from softmax probabilities."
            ),
            "R2C+CumulativeLink": (
                f"MLP({self.N_BINS - 1} cumulative logits) + CumulativeLinkLoss "
                "→ R2CConformal.  Ordinal regression backbone with sigmoid-based "
                "bin probabilities."
            ),
            "R2C+CORAL": (
                "MLP backbone + CORALHead(shared-weight, monotonic biases) "
                "+ CORALLoss → R2CConformal.  Rank-consistent ordinal "
                "architecture with R2C conformal on top."
            ),
            "torchcp_R2C": (
                "torchcp R2CCP score + R2ccpLoss + SplitPredictor.  "
                "Uses torchcp's build_regression_model and ordinal-aware "
                "binned loss per Guha et al. (2024)."
            ),
            "Split+MLP": (
                "MLP + MSE → post-hoc split conformal baseline.  "
                "Standard approach — no classification discretisation."
            ),
            "CQR+MLP": (
                "MLP quantile head + MultiQuantileLoss → post-hoc CQR baseline.  "
                "Conformalized quantile regression for comparison."
            ),
        }
        return notes.get(method, "")

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)

        X_train = torch.from_numpy(splits["X_train"]).float()
        y_train_np = splits["y_train"]
        X_cal = torch.from_numpy(splits["X_cal"]).float()
        y_cal = torch.from_numpy(splits["y_cal"]).float()
        in_dim = X_train.shape[1]
        hidden = _hidden_size(in_dim)
        alpha = self.ALPHA
        n_bins = self.N_BINS

        # Shared bin edges (computed from training targets)
        bin_edges_np = _make_bin_edges(y_train_np, n_bins=n_bins)
        bin_edges = torch.from_numpy(bin_edges_np).float()

        # Discretise training and calibration targets
        y_train_bins = _discretise_targets(y_train_np, bin_edges_np)
        y_train_idx = torch.from_numpy(y_train_bins).long()
        y_cal_bins = _discretise_targets(splits["y_cal"], bin_edges_np)

        # --- R2C+Softmax -------------------------------------------------
        if method == "R2C+Softmax":
            model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=n_bins)
            loss_fn = nn.CrossEntropyLoss()
            _train_one(model, loss_fn, X_train, y_train_idx, config)

            return {
                "type": "r2c",
                "model": model,
                "X_cal": X_cal,
                "y_cal": y_cal,
                "y_cal_bins": y_cal_bins,
                "bin_edges": bin_edges_np,
                "alpha": alpha,
                "predict_fn": "softmax",
            }

        # --- R2C+CumulativeLink ------------------------------------------
        elif method == "R2C+CumulativeLink":
            from torchregress.losses import CumulativeLinkLoss

            model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=n_bins - 1)
            loss_fn = CumulativeLinkLoss()
            _train_one(model, loss_fn, X_train, y_train_idx, config, epochs=150)

            return {
                "type": "r2c",
                "model": model,
                "X_cal": X_cal,
                "y_cal": y_cal,
                "y_cal_bins": y_cal_bins,
                "bin_edges": bin_edges_np,
                "alpha": alpha,
                "predict_fn": "cumulative_logits",
            }

        # --- R2C+CORAL ---------------------------------------------------
        elif method == "R2C+CORAL":
            from torchregress.losses import CORALLoss
            from torchregress.utils.ordinal import CORALHead

            backbone = _MLP(in_dim=in_dim, hidden=hidden, out_dim=hidden)
            head = CORALHead(in_features=hidden, num_classes=n_bins)
            model = nn.Sequential(backbone, head)
            loss_fn = CORALLoss()
            _train_one(model, loss_fn, X_train, y_train_idx, config, epochs=150)

            return {
                "type": "r2c",
                "model": model,
                "X_cal": X_cal,
                "y_cal": y_cal,
                "y_cal_bins": y_cal_bins,
                "bin_edges": bin_edges_np,
                "alpha": alpha,
                "predict_fn": "cumulative_logits",
            }

        # --- torchcp R2C -------------------------------------------------
        elif method == "torchcp_R2C":
            if not _TORCHCP_R2C_AVAILABLE:
                raise ImportError("torchcp R2CCP not available")

            # torchcp's R2CCP works differently:
            # - build_regression_model creates the model
            # - R2ccpLoss is the training loss
            # - R2CCP is the score function used with SplitPredictor
            device = torch.device(config.device if torch.cuda.is_available() else "cpu")

            # Build model using torchcp's factory
            model = build_regression_model(model_name="NonLinearNet")

            # Override input dimension (torchcp default might differ)
            # The model's first layer needs to match in_dim
            # Build a wrapped model with correct input dim
            class _R2CWrapper(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(in_dim, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, n_bins),
                    )

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.net(x)

            model = _R2CWrapper().to(device)

            # Train with R2ccpLoss (ordinal-aware binned loss)
            bin_edges_t = bin_edges.to(device)
            midpoints = (bin_edges_t[:-1] + bin_edges_t[1:]) / 2
            loss_fn = R2ccpLoss(p=2, tau=0.1, midpoints=midpoints)
            X_train_dev = X_train.to(device)
            y_train_dev = y_train_idx.to(device)

            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=config.lr)
            n = X_train_dev.shape[0]
            batch_size = config.batch_size
            for _ in range(config.epochs):
                perm = torch.randperm(n)
                for i in range(0, n, batch_size):
                    idx = perm[i : i + batch_size]
                    opt.zero_grad(set_to_none=True)
                    out = model(X_train_dev[idx])
                    loss = loss_fn(out, y_train_dev[idx])
                    loss.backward()
                    opt.step()

            model.eval()

            return {
                "type": "torchcp_r2c",
                "model": model,
                "X_cal": X_cal,
                "y_cal_bins": y_cal_bins,
                "bin_edges": bin_edges_np,
                "alpha": alpha,
                "device": device,
            }

        # --- Split+MLP (standard split conformal baseline) -----------------
        elif method == "Split+MLP":
            model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=1)
            loss_fn = nn.MSELoss()
            y_train = torch.from_numpy(y_train_np).float().unsqueeze(1)
            _train_one(model, loss_fn, X_train, y_train, config)
            model.eval()

            return {
                "type": "split",
                "model": model,
                "X_cal": X_cal,
                "y_cal": y_cal,
                "alpha": alpha,
            }

        # --- CQR+MLP (CQR baseline) ----------------------------------------
        elif method == "CQR+MLP":
            from torchregress.losses.quantile import MultiQuantileLoss

            model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=2)
            loss_fn = MultiQuantileLoss(quantiles=[alpha / 2, 1.0 - alpha / 2])
            y_train = torch.from_numpy(y_train_np).float().unsqueeze(1)
            _train_one(model, loss_fn, X_train, y_train, config)
            model.eval()

            return {
                "type": "cqr",
                "model": model,
                "X_cal": X_cal,
                "y_cal": y_cal,
                "alpha": alpha,
            }

        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        del config
        alpha = model_state.get("alpha", self.ALPHA)
        model_type = model_state["type"]

        X_test_t = torch.from_numpy(X_test).float()

        # --- R2C methods --------------------------------------------------
        if model_type == "r2c":
            from torchregress.losses.conformal import R2CConformal

            model = model_state["model"]
            X_cal = model_state["X_cal"]
            bin_edges_np = model_state["bin_edges"]
            predict_fn = model_state["predict_fn"]

            model.eval()
            with torch.no_grad():
                logits_cal = model(X_cal)
                logits_test = model(X_test_t)

                # Convert to softmax probabilities
                if predict_fn == "softmax":
                    probs_cal = F.softmax(logits_cal, dim=-1)
                    probs_test = F.softmax(logits_test, dim=-1)
                else:  # cumulative_logits
                    probs_cal = _cumulative_logits_to_probs(logits_cal)
                    probs_test = _cumulative_logits_to_probs(logits_test)

            # Calibrate R2CConformal with *continuous* targets (not bin indices).
            # R2CConformal internally discretises continuous targets via searchsorted,
            # so passing pre-discretised bin indices produces garbage scores.
            bin_edges_t = torch.from_numpy(bin_edges_np).float()
            r2c = R2CConformal(alpha=alpha, bin_edges=bin_edges_t)
            y_cal_cont = model_state["y_cal"]  # stored as torch float tensor
            r2c.calibrate(probs_cal, y_cal_cont)

            # Predict intervals
            lo, hi = r2c.predict_interval(probs_test)
            lo_np = lo.detach().cpu().numpy().reshape(-1)
            hi_np = hi.detach().cpu().numpy().reshape(-1)

            # Also return point predictions (bin with highest probability)
            y_pred_idx = probs_test.argmax(dim=-1).numpy()
            centers = (bin_edges_np[:-1] + bin_edges_np[1:]) / 2
            y_pred = centers[y_pred_idx]

            return {"lower": lo_np, "upper": hi_np, "y_pred": y_pred}

        # --- torchcp R2C --------------------------------------------------
        elif model_type == "torchcp_r2c":
            if not _TORCHCP_R2C_AVAILABLE:
                raise ImportError("torchcp R2CCP not available")

            model = model_state["model"]
            X_cal = model_state["X_cal"]
            y_cal_bins = model_state["y_cal_bins"]
            bin_edges_np = model_state["bin_edges"]
            device = model_state["device"]

            bin_edges_t = torch.from_numpy(bin_edges_np).float()
            midpoints = (bin_edges_t[:-1] + bin_edges_t[1:]) / 2

            # R2CCP has its own generate_intervals() method — use it directly
            # rather than SplitPredictor which expects regression-style intervals.
            model.eval()
            model.to(device)

            X_cal_dev = X_cal.to(device)
            y_cal_dev = torch.from_numpy(np.asarray(y_cal_bins)).long().to(device)

            # Compute calibration scores via R2CCP.__call__
            score_fn = R2CCP(midpoints=midpoints, device=device)
            with torch.no_grad():
                probs_cal = F.softmax(model(X_cal_dev), dim=-1)
            scores = score_fn(probs_cal, y_cal_dev)

            # Conformal quantile
            n_cal = scores.numel()
            q_level = min(np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal, 1.0)
            q_hat = torch.quantile(scores, q_level, interpolation="higher")

            # Generate intervals for test set
            X_test_dev = X_test_t.to(device)
            with torch.no_grad():
                probs_test = F.softmax(model(X_test_dev), dim=-1)
            result = score_fn.generate_intervals(probs_test, q_hat)
            # result shape: (batch, 2*(n_bins-1)) — first half = per-pair lower,
            # second half = per-pair upper.  Take extremes for overall interval.
            mid = result.shape[1] // 2
            lo = result[:, :mid].min(dim=-1).values.detach().cpu().numpy().reshape(-1)
            hi = result[:, mid:].max(dim=-1).values.detach().cpu().numpy().reshape(-1)

            # Point prediction from already-computed probabilities
            y_pred_idx = probs_test.argmax(dim=-1).cpu().numpy()
            centers = (bin_edges_np[:-1] + bin_edges_np[1:]) / 2
            y_pred = centers[y_pred_idx]

            return {"lower": lo, "upper": hi, "y_pred": y_pred}

        # --- Split conformal ----------------------------------------------
        elif model_type == "split":
            model = model_state["model"]
            X_cal = model_state["X_cal"]
            y_cal = model_state["y_cal"]
            lo, hi = _torch_split_intervals(model, X_cal, y_cal, X_test_t, alpha=alpha)
            # Point prediction
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test_t).squeeze(-1).numpy()
            return {"lower": lo, "upper": hi, "y_pred": y_pred}

        # --- CQR ----------------------------------------------------------
        elif model_type == "cqr":
            model = model_state["model"]
            X_cal = model_state["X_cal"]
            y_cal = model_state["y_cal"]
            lo, hi = _torch_cqr_intervals(model, X_cal, y_cal, X_test_t, alpha=alpha)
            # Point prediction (mean of quantiles)
            model.eval()
            with torch.no_grad():
                preds = model(X_test_t)
                y_pred = preds.mean(dim=-1).numpy()
            return {"lower": lo, "upper": hi, "y_pred": y_pred}

        raise ValueError(f"Unknown model type: {model_type}")

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        """Compute both interval and point metrics."""
        from harness.metrics import compute_interval_metrics, compute_point_metrics
        from harness.runners import _inverse_transform_predictions, _inverse_transform_y

        y_scaler = predictions.get("_y_scaler")
        if y_scaler is not None:
            predictions = _inverse_transform_predictions(predictions, y_scaler)
            y_test = _inverse_transform_y(y_test, y_scaler)

        lo = np.asarray(predictions["lower"]).reshape(-1)
        hi = np.asarray(predictions["upper"]).reshape(-1)
        y_true = np.asarray(y_test).reshape(-1)

        metrics: dict[str, float] = {}
        metrics.update(compute_interval_metrics(lo, hi, y_true))

        if "y_pred" in predictions:
            y_pred = np.asarray(predictions["y_pred"]).reshape(-1)
            metrics.update(compute_point_metrics(y_pred, y_true))

        return metrics

    # ------------------------------------------------------------------
    # Override run() to create calibration split
    # ------------------------------------------------------------------

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: dict[str, np.ndarray] | None = None,
        *,
        scenario_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run the R2C conformal suite, carving a calibration split."""
        from time import perf_counter

        cfg = config or HarnessConfig()

        # Resolve canonical scenario
        dataset_name, splits = self._prepare_scenario_run(
            scenario_name,
            dataset_name,
            splits,
            standardize_targets=cfg.standardize_targets,
        )

        if splits is None:
            from harness.datasets import load_dataset

            splits = load_dataset(dataset_name, standardize_targets=cfg.standardize_targets)

        # Carve calibration split
        augmented_splits = _carve_calibration_split(
            splits,
            cal_frac=self.CALIBRATION_FRAC,
            seed=cfg.seed,
            smoke=cfg.smoke,
            suite_name="r2c_conformal",
            forward_keys=("_alpha", "_scenario", "_source_url"),
        )

        X_test = augmented_splits["X_test"]
        y_test = augmented_splits["y_test"]

        rows: list[dict[str, Any]] = []

        for method in self.methods():
            set_harness_seed(cfg.seed)

            try:
                t0 = perf_counter()
                model_state = self.fit(method, augmented_splits, cfg)
                fit_time = perf_counter() - t0

                t0 = perf_counter()
                predictions = self.predict(method, model_state, X_test, cfg)
                pred_time = perf_counter() - t0

                if "_y_scaler" in augmented_splits:
                    predictions["_y_scaler"] = augmented_splits["_y_scaler"]

                metrics = self.evaluate(method, predictions, y_test, cfg)

                row: dict[str, Any] = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    **metrics,
                    "train_s": fit_time,
                    "eval_s": pred_time,
                    "SourceURL": splits.get("_source_url", ""),
                    "SyntheticFallback": bool(splits.get("_syntheticfallback", False)),
                    "Notes": self.method_notes(method),
                }

            except ImportError as exc:
                row = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    "Coverage": None,
                    "Width": None,
                    "IntervalScore": None,
                    "train_s": None,
                    "eval_s": None,
                    "SourceURL": splits.get("_source_url", ""),
                    "SyntheticFallback": bool(splits.get("_syntheticfallback", False)),
                    "Notes": f"skipped: {type(exc).__name__}: {exc}",
                }

            except (ValueError, RuntimeError, KeyError, TypeError, OSError) as exc:
                row = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    "Coverage": None,
                    "Width": None,
                    "IntervalScore": None,
                    "train_s": None,
                    "eval_s": None,
                    "SourceURL": splits.get("_source_url", ""),
                    "SyntheticFallback": bool(splits.get("_syntheticfallback", False)),
                    "Notes": f"failed: {type(exc).__name__}: {exc}",
                }

            rows.append(row)

        return rows
