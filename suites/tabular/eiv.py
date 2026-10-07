"""Errors-in-Variables (EIV) regression suite.

Compares torchregress EIV losses against a plain MSE baseline on synthetic
data with known feature-measurement noise.  The dataset provides both noisy
observed features (X) and clean true features (X_true); the suite trains on
the noisy data and evaluates point-prediction quality.

Methods
------
- MSE              : MLP + WeightedMSELoss (no EIV correction — baseline)
- FunctionalEIV    : MLP + FunctionalEIVLoss (analytical gradient-based)
- FunctionalEIV_MC : MLP + FunctionalEIVLoss (Monte-Carlo variance — no Jacobian)
- StructuralEIV    : MLP + StructuralEIVLoss (full cross-covariance model)
- ODR              : MLP + OrthogonalDistanceRegressionLoss (latent-x opt)
- EnsembleEIV      : MLP + EnsembleEIVLoss (Monte-Carlo input perturbation)
- InputNoiseMC     : MLP + InputNoiseMarginalizationLoss (MC marginalisation at train time)
- MSE_NoisyPredict : MSE-trained MLP + NoisyInputPredictor (MC averaging at inference only)
- LatentNN         : LatentNN algorithm — jointly optimises model + latent clean inputs (with test-time marginalization fix)
- NoiseAware       : NoiseAwareRegressor — error-aware feature encoder + backbone
- SIMEX            : SIMEX algorithm — simulation-extrapolation with matched-noise predict fix
- RC               : RegressionCalibration — pre-processing input calibration via reliability
- TLS              : noise-weighted Total Least Squares via SVD (numpy-only)
- SciPyODR         : scipy.odr.ODR — classical orthogonal distance regression (external)
- MSE_WarmupMC     : MSE pre-training (20 epochs) → FunctionalEIV_MC — warmup for linear data
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchregress.algorithms import (
    SIMEX,
    LatentNN,
    NoiseAwareRegressor,
    RegressionCalibration,
    WarmupMCTrainer,
)
from torchregress.losses import (
    EnsembleEIVLoss,
    FunctionalEIVLoss,
    GaussianNLLLoss,
    InputNoiseAugmentationLoss,
    LatentMarginalizationLoss,
    OrthogonalDistanceRegressionLoss,
    StructuralEIVLoss,
    WeightedMSELoss,
)
from torchregress.losses.eiv import (
    InputNoiseBinnedPDFLoss,
    InputNoiseMDNLoss,
    NoisyInputPredictor,
)

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _MLP, _carve_calibration_split, _hidden_size, _Linear, _train_one

# --- scipy.odr (deprecated in scipy >= 1.17; check gracefully) ---
# ODRPACK expects x with shape (n_vars, n_obs); the suite therefore passes
# X.T. The probe below uses the same orientation as the SciPyODR fit path.
try:
    from scipy.odr import ODR as _ODR
    from scipy.odr import Model as _ODRModel
    from scipy.odr import RealData

    # Verify it actually works with sx/sy parameters (it may be broken
    # in the deprecation window, e.g. scipy >= 1.17 on macOS).
    _x_test = np.random.randn(3, 10).astype(np.float64)
    _y_test = np.random.randn(10).astype(np.float64)
    _model_test = _ODRModel(lambda beta, x: beta[0] + beta[1:] @ x)
    _odr_test = _ODR(RealData(_x_test, _y_test, sx=0.2, sy=0.1), _model_test, beta0=np.zeros(4))
    _odr_test.run()
    HAS_SCIPY_ODR = True
except Exception:
    HAS_SCIPY_ODR = False


# --- Custom training helper for EIV losses
#
# EIV losses are called with (x_obs, y_obs), NOT (model(x), y).  The model
# is passed at construction time and the loss calls it internally.
# ---------------------------------------------------------------------------


def _train_eiv(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    config: HarnessConfig | None = None,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    sigma_rows: torch.Tensor | None = None,
    forward_sigma: bool = False,
) -> None:
    """Mini-batch training loop for EIV losses (loss_fn takes x_obs, y_obs).

    ``sigma_rows`` ([N, D] per-observation noise, aligned with ``X`` rows)
    is sliced per batch and forwarded as ``sigma_x=`` when ``forward_sigma``
    is set — the per-obs path for losses with per-call sigma support
    (Functional/Structural/ODR/Ensemble/InputNoise families). Otherwise the
    construction-time sigma applies (per-band fallback).
    """
    if config is not None:
        n_epochs = epochs if epochs is not None else config.epochs
        bs = batch_size if batch_size is not None else config.batch_size
        learning_rate = lr if lr is not None else config.lr
        # Throughput-motivated uplift at scale (documented): mini-batch 64
        # on >100k rows is infeasible for slow EIV methods (ODR latent-opt,
        # SIMEX ensembles, input-noise MC). Dynamics change is confined to
        # big-data runs; small-data protocol is untouched.
        if batch_size is None and X.shape[0] > 100_000:
            bs = max(bs, 1024)
    else:
        n_epochs = epochs or 50
        bs = batch_size or 64
        learning_rate = lr or 1e-3

    dataset = TensorDataset(X, y)
    use_rows = forward_sigma and sigma_rows is not None and sigma_rows.shape[0] == X.shape[0]
    if use_rows:
        dataset = TensorDataset(X, y, torch.arange(X.shape[0]), sigma_rows.detach().cpu())
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=False)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model.train()
    for _epoch in range(n_epochs):
        if use_rows:
            for X_batch, y_batch, _idx, sig_batch in loader:
                optimizer.zero_grad()
                # EIV losses: forward(x_obs, target) — the loss internally calls model(x_obs)
                loss = loss_fn(X_batch, y_batch, sigma_x=sig_batch.to(X_batch.device))
                loss.backward()
                optimizer.step()
            continue
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            # EIV losses: forward(x_obs, target) — the loss internally calls model(x_obs)
            loss = loss_fn(X_batch, y_batch)
            loss.backward()
            optimizer.step()
    model.eval()


# --- Shared prediction helper (noisy and clean X paths) ---


class TLSIllConditionedError(RuntimeError):
    """Total least squares has no stable finite solution for this design."""


def tls_solve(
    X: np.ndarray,
    y: np.ndarray,
    sigma_x: Any = 1.0,
    sigma_y: float = 1.0,
    *,
    min_v_last: float = 0.05,
    max_sv_ratio: float = 0.98,
) -> tuple[np.ndarray, float]:
    """Noise-weighted total least squares via SVD (Deming for one feature).

    Columns of X are divided by ``sigma_x`` and y by ``sigma_y`` before the
    SVD of ``[X | y]`` so that errors are unit-variance in every coordinate
    (the TLS assumption).  Without this, unequal feature/target scales make
    the orthogonal fit meaningless (the "smallest singular vector" simply
    tracks the lowest-variance direction).  For one feature this is exactly
    Deming regression with ``delta = sigma_y**2 / sigma_x**2``.

    Raises :class:`TLSIllConditionedError` when the solution is not stable:
    the smallest right singular vector has (almost) no y-component
    (``|V[-1]| < min_v_last``) so the coefficients blow up like
    ``1 / V[-1]``, or ``sigma_min([X|y])`` is within ``max_sv_ratio`` of
    ``sigma_min(X)`` (the classical TLS existence condition is barely met,
    typical for collinear features).

    Returns ``(coef, intercept)`` in the units of the inputs.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    sx = np.broadcast_to(
        np.asarray(
            sigma_x.detach().cpu().numpy() if isinstance(sigma_x, torch.Tensor) else sigma_x,
            dtype=np.float64,
        ).reshape(-1),
        (X.shape[1],),
    )
    sy = float(sigma_y)
    if not (np.all(sx > 0) and sy > 0):
        raise ValueError("TLS noise scales must be positive")
    x_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    Zx = (X - x_mean) / sx
    Z = np.column_stack([Zx, (y - y_mean) / sy])
    _, sv, Vt = np.linalg.svd(Z, full_matrices=False)
    v = Vt[-1]
    sv_x_min = np.linalg.svd(Zx, compute_uv=False)[-1]
    if abs(v[-1]) < min_v_last or sv[-1] > max_sv_ratio * sv_x_min:
        raise TLSIllConditionedError(
            f"TLS ill-conditioned: |V[-1]|={abs(v[-1]):.3g}, "
            f"sigma_min([X|y])/sigma_min(X)={sv[-1] / max(sv_x_min, 1e-300):.4f} "
            "(collinear features / noise-dominated target; no stable finite solution)"
        )
    beta_scaled = -v[:-1] / v[-1]
    coef = beta_scaled * sy / sx
    intercept = float(y_mean - x_mean @ coef)
    return coef, intercept


def _eiv_sigma(value: Any) -> Any:
    """Pass-through measurement-noise scale: scalars stay scalars (fast
    path for classical algorithms), arrays become CPU float tensors for
    the tensor-ready losses/predictors (Functional/Structural/ODR/Ensemble/
    InputNoise/LatentNN/NoisyInputPredictor/NoiseAware all accept them).

    Stays on CPU deliberately: consumers move internally as needed, while
    suite-side mixing (CPU model + CUDA sigma) breaks predict paths.
    Enables heteroscedastic datasets (per-band GalaxiesML sigmas) without
    changing any method's math on scalar datasets.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, (float, int)):
        return float(value)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return float(arr)
    return torch.as_tensor(arr)


# Methods whose losses accept per-call sigma_x (per-obs threading).
_PER_OBS_METHODS = frozenset(
    {
        "FunctionalEIV",
        "FunctionalEIV_MC",
        "StructuralEIV",
        "ODR",
        "EnsembleEIV",
        "InputNoiseMC",
        "InputNoiseMDN",
        "InputNoiseBinnedPDF",
        "GaussianNLL+InputNoiseMC",
        "TLS_Init+FunctionalEIV",
        "MSE_NoisyPredict",
    }
)


def _test_sigma(model_state: dict[str, Any], n_rows: int) -> Any:
    """Per-obs test sigma truncated to the presented rows (or scalar)."""
    arr = model_state.get("sigma_x_test")
    if arr is None:
        return _eiv_sigma(model_state.get("sigma_x", 0.2))
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2 and a.shape[0] >= n_rows:
        return torch.from_numpy(a[:n_rows])
    return _eiv_sigma(model_state.get("sigma_x", 0.2))


def _eiv_predict(
    method: str,
    model_state: dict[str, Any],
    m: Any,
    X_np: np.ndarray,
    *,
    noisy: bool,
) -> np.ndarray:
    """Predict on X_np for an EIV method, returning squeezed numpy array."""
    model_type = model_state.get("type", "torch")

    if model_type in ("scipy_odr", "tls"):
        beta = model_state["coef"]
        if model_type == "scipy_odr":
            return beta[0] + X_np @ beta[1:]
        intercept = model_state["intercept"]
        return X_np @ beta + intercept

    if model_type == "latentnn":
        sigma_x_val = _eiv_sigma(model_state.get("sigma_x", 0.2))
        with torch.no_grad():
            X = torch.from_numpy(X_np).float().to(current_device())
            return m.predict(X, sigma_x=sigma_x_val).squeeze(-1).detach().cpu().numpy()

    if model_type == "simex":
        with torch.no_grad():
            X = torch.from_numpy(X_np).float().to(current_device())
            return m.predict(X).squeeze(-1).detach().cpu().numpy()

    if model_type == "rc":
        if noisy:
            X_cal = model_state["X_cal_test"]
            with torch.no_grad():
                return m(X_cal).squeeze(-1).detach().cpu().numpy()
        # Clean X doesn't need RC calibration — use model directly
        with torch.no_grad():
            X = torch.from_numpy(X_np).float().to(current_device())
            return m(X).squeeze(-1).detach().cpu().numpy()

    if method == "MSE_NoisyPredict" and noisy:
        n_rows = np.asarray(X_np).shape[0]
        sigma_x_val = _test_sigma(model_state, n_rows)
        noisy_pred = NoisyInputPredictor(m, sigma_x=sigma_x_val, n_samples=16)
        with torch.no_grad():
            X = torch.from_numpy(X_np).float().to(current_device())
            out = noisy_pred(X)
            if not torch.is_tensor(out):
                out = torch.as_tensor(np.asarray(out))
            return out.squeeze(-1).detach().cpu().numpy()

    if model_type == "mdn_eiv":
        loss_fn = model_state["loss_fn"]
        with torch.no_grad():
            X = torch.from_numpy(X_np).float().to(current_device())
            mean, _ = loss_fn.predict_mean_std(m(X))
            return mean.squeeze(-1).detach().cpu().numpy()

    if model_type == "binned_eiv":
        centers = model_state["centers"].to(current_device())
        with torch.no_grad():
            X = torch.from_numpy(X_np).float().to(current_device())
            probs = torch.softmax(m(X), dim=-1)
            return (probs @ centers).squeeze(-1).detach().cpu().numpy()

    # Torch models
    with torch.no_grad():
        X = torch.from_numpy(X_np).float().to(current_device())
        if method == "NoiseAware":
            # NoiseAware always needs sigma_x (architectural requirement)
            sigma_x_val = _eiv_sigma(model_state.get("sigma_x", 0.2))
            return m(X, sigma_x_val).squeeze(-1).detach().cpu().numpy()
        return m(X).squeeze(-1).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class EIVSuite(HarnessRunner):
    """Errors-in-Variables regression benchmark suite."""

    @property
    def name(self) -> str:
        return "eiv"

    @property
    def description(self) -> str:
        return (
            "Errors-in-Variables: torchregress EIV losses vs plain MSE baseline "
            "on synthetic data with known feature noise"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "synthetic_eiv",
            "synthetic_eiv_highnoise",
            "synthetic_eiv_nonlinear",
            "synthetic_eiv_nonlinear_highnoise",
            "synthetic_eiv_veryhighnoise",
            "synthetic_eiv_nonlinear_veryhighnoise",
            "photoz_eiv",
            "galaxiesml_eiv",
            "folktables_eiv",
        ]

    @property
    def canonical_scenarios(self) -> list[str]:
        return []

    def methods(self) -> list[str]:
        return [
            "MSE",
            "FunctionalEIV",
            "FunctionalEIV_MC",
            "StructuralEIV",
            "ODR",
            "EnsembleEIV",
            "InputNoiseMC",
            "InputNoiseMDN",
            "InputNoiseBinnedPDF",
            "LatentMarginalization",
            "MSE_NoisyPredict",
            "LatentNN",
            "NoiseAware",
            "SIMEX",
            "RC",
            "TLS",
            "SciPyODR",
            # --- Composability / distributional / conformal ---
            "GaussianNLL+InputNoiseMC",
            "RC+GaussianNLL",
            "RC+FunctionalEIV",
            "TLS_Init+FunctionalEIV",
            "MSE_WarmupMC",
            "SplitCP+FunctionalEIV",
            "SplitCP+InputNoiseMC",
            "SplitCP+MSE_WarmupMC",
        ]

    def library_name(self, method: str) -> str:
        if method.startswith("SplitCP+"):
            return self.library_name(method.split("+", 1)[1]) + "+CP"
        if method == "TLS":
            return "numpy"
        if method == "SciPyODR":
            return "scipy"
        return "torchregress"

    # ------------------------------------------------------------------
    # run — carve calibration split for SplitCP methods
    # ------------------------------------------------------------------

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: dict[str, np.ndarray] | None = None,
        *,
        scenario_name: str = "",
    ) -> list[dict[str, Any]]:
        import time

        # Skip scenario resolution for EIV (no canonical scenarios)
        cfg = config or HarnessConfig()
        if splits is None:
            from harness.datasets import load_dataset

            splits = load_dataset(
                dataset_name,
                standardize_targets=cfg.standardize_targets,
            )

        rows: list[dict[str, Any]] = []
        for method in self.methods():
            set_harness_seed(cfg.seed)
            try:
                # Time training
                t0 = time.perf_counter()
                state = self.fit(method, splits, cfg)
                train_s = time.perf_counter() - t0
                # Time prediction + evaluation
                t0 = time.perf_counter()
                preds = self.predict(method, state, splits["X_test"], cfg)
                if "_y_scaler" in splits:
                    preds["_y_scaler"] = splits["_y_scaler"]
                # Per-row target std for the uncertain-ground-truth report
                # (evaluate consumes it when present; absent otherwise).
                if "y_std_test" in splits:
                    try:
                        preds["y_std_test"] = np.asarray(
                            splits["y_std_test"], dtype=np.float64
                        ).reshape(-1)
                    except Exception:
                        pass
                metrics = self.evaluate(method, preds, splits["y_test"], cfg)
                eval_s = time.perf_counter() - t0
                rows.append(
                    {
                        "Method": f"{self.name}/{method}",
                        "Library": self.library_name(method),
                        "Dataset": dataset_name,
                        "Suite": self.name,
                        "train_s": train_s,
                        "eval_s": eval_s,
                        "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                        "Device": str(current_device()),
                        "Notes": self.method_notes(method),
                        **metrics,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "Method": f"{self.name}/{method}",
                        "Library": self.library_name(method),
                        "Dataset": dataset_name,
                        "Suite": self.name,
                        "RMSE": None,
                        "MAE": None,
                        "R2": None,
                        "train_s": None,
                        "eval_s": None,
                        "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                        "Device": str(current_device()),
                        "Notes": f"skipped: {exc}",
                    }
                )
        return rows

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

        X_train_np = splits["X_train"]
        y_train_np = splits["y_train"]

        X = torch.from_numpy(X_train_np).float().to(current_device())
        y = torch.from_numpy(y_train_np).float().to(current_device())
        if y.dim() == 1:
            y = y.unsqueeze(1)

        # Known noise parameters from the dataset generator (fallback matches
        # the defaults in _load_synthetic_eiv: sigma_x=0.2, sigma_y=0.1).
        # Scalar or per-feature array (heteroscedastic datasets) —
        # losses/algorithms accept both; see _eiv_sigma.
        sigma_x = _eiv_sigma(splits.get("sigma_x", 0.2))
        sigma_y = float(splits.get("sigma_y", 0.1))

        # Per-observation noise [N, D] (heteroscedastic datasets only):
        # threaded per-batch into losses with per-call sigma_x support.
        sigma_rows: torch.Tensor | None = None
        _raw_rows = splits.get("sigma_x_train")
        if _raw_rows is not None:
            _arr = np.asarray(_raw_rows, dtype=np.float32)
            if _arr.ndim == 2 and _arr.shape[0] == np.asarray(X_train_np).shape[0]:
                sigma_rows = torch.from_numpy(_arr)
        # Per-obs test sigma for predict-time marginalization (truncated
        # to the presented rows in predict()).
        _raw_te = splits.get("sigma_x_test")
        sigma_test: np.ndarray | None = None
        if _raw_te is not None:
            _te = np.asarray(_raw_te, dtype=np.float32)
            if _te.ndim == 2:
                sigma_test = _te

        in_dim = X.shape[1]
        out_dim = y.shape[1]
        hidden = _hidden_size(in_dim)

        # Store clean test data for attenuation-bias evaluation
        _X_test_true = splits.get("X_test_true")

        if method == "MSE":
            model = _MLP(in_dim, hidden, out_dim)
            loss_fn = WeightedMSELoss()
            _train_one(model, loss_fn, X, y, config)
            result = {"type": "torch", "model": model}
            if _X_test_true is not None:
                result["_X_test_true"] = _X_test_true
            return result

        elif method == "FunctionalEIV":
            model = _MLP(in_dim, hidden, out_dim)
            loss_fn = FunctionalEIVLoss(model=model, sigma_x=sigma_x, sigma_y=sigma_y)
            _train_eiv(model, loss_fn, X, y, config, sigma_rows=sigma_rows, forward_sigma=True)
            result = {"type": "torch", "model": model}
            if _X_test_true is not None:
                result["_X_test_true"] = _X_test_true
            return result

        elif method == "FunctionalEIV_MC":
            # Monte-Carlo variance propagation — avoids noisy Jacobian entirely.
            # Sampled perturbations → empirical covariance → Gaussian NLL.
            # Same loss function as analytical EIV but without the Jacobian path.
            model = _MLP(in_dim, hidden, out_dim)
            loss_fn = FunctionalEIVLoss(
                model=model,
                sigma_x=sigma_x,
                sigma_y=sigma_y,
                monte_carlo=True,
                n_samples=20,
            )
            _train_eiv(model, loss_fn, X, y, config, sigma_rows=sigma_rows, forward_sigma=True)
            result = {"type": "torch", "model": model}
            if _X_test_true is not None:
                result["_X_test_true"] = _X_test_true
            return result

        elif method == "StructuralEIV":
            model = _MLP(in_dim, hidden, out_dim)
            # Uncorrelated feature/target noise → zero cross-covariance
            sigma_xy = torch.zeros(out_dim, in_dim)
            loss_fn = StructuralEIVLoss(
                model=model, sigma_x=sigma_x, sigma_y=sigma_y, sigma_xy=sigma_xy
            )
            _train_eiv(model, loss_fn, X, y, config, sigma_rows=sigma_rows, forward_sigma=True)
            result = {"type": "torch", "model": model}
            if _X_test_true is not None:
                result["_X_test_true"] = _X_test_true
            return result

        elif method == "ODR":
            model = _MLP(in_dim, hidden, out_dim)
            loss_fn = OrthogonalDistanceRegressionLoss(
                model=model, sigma_x=sigma_x, sigma_y=sigma_y
            )
            _train_eiv(model, loss_fn, X, y, config, sigma_rows=sigma_rows, forward_sigma=True)
            result = {"type": "torch", "model": model}
            if _X_test_true is not None:
                result["_X_test_true"] = _X_test_true
            return result

        elif method == "EnsembleEIV":
            model = _MLP(in_dim, hidden, out_dim)
            loss_fn = EnsembleEIVLoss(model=model, sigma_x=sigma_x)
            _train_eiv(model, loss_fn, X, y, config, sigma_rows=sigma_rows, forward_sigma=True)
            result = {"type": "torch", "model": model}
            if _X_test_true is not None:
                result["_X_test_true"] = _X_test_true
            return result

        elif method == "InputNoiseMC":
            # Input-noise marginalization: perturb inputs → run model → avg MSE.
            # More principled than EnsembleEIV — proper MC marginalisation over
            # input noise with configurable base loss.
            model = _MLP(in_dim, hidden, out_dim)
            loss_fn = InputNoiseAugmentationLoss(
                model=model,
                base_loss=WeightedMSELoss(),
                sigma_x=sigma_x,
                n_samples=20,
            )
            _train_eiv(model, loss_fn, X, y, config, sigma_rows=sigma_rows, forward_sigma=True)
            result = {"type": "torch", "model": model}
            if _X_test_true is not None:
                result["_X_test_true"] = _X_test_true
            return result

        elif method == "InputNoiseMDN":
            # MDN head under input-noise marginalization (multimodal EIV).
            from torchregress.losses import MixtureDensityLoss

            mdn = MixtureDensityLoss(n_components=3, n_features=out_dim)
            model = _MLP(in_dim, hidden, out_dim=mdn.expected_output_size)
            loss_fn = InputNoiseMDNLoss(
                model=model,
                n_components=3,
                n_features=out_dim,
                sigma_x=sigma_x,
            )
            _train_eiv(model, loss_fn, X, y, config, sigma_rows=sigma_rows, forward_sigma=True)
            result = {"type": "mdn_eiv", "model": model, "loss_fn": mdn}
            if _X_test_true is not None:
                result["_X_test_true"] = _X_test_true
            return result

        elif method == "InputNoiseBinnedPDF":
            # Binned-PDF marginalization over an ordinal head: discretize
            # targets into quantile bins, train class logits under noise.
            # (No local torch import: module-global torch must win.)
            n_bins = 10
            y_np = np.asarray(splits["y_train"]).reshape(-1)
            edges = np.quantile(y_np, np.linspace(0, 1, n_bins + 1)).astype(np.float32)
            edges[0], edges[-1] = -np.inf, np.inf
            centers = (
                (
                    np.quantile(y_np, np.linspace(0, 1, n_bins + 1))[:-1]
                    + np.quantile(y_np, np.linspace(0, 1, n_bins + 1))[1:]
                )
                / 2.0
            ).astype(np.float32)
            model = _MLP(in_dim, hidden, n_bins)
            loss_fn = InputNoiseBinnedPDFLoss(model=model, sigma_x=sigma_x)
            dev = current_device()
            y_idx = torch.from_numpy(np.clip(np.digitize(y_np, edges) - 1, 0, n_bins - 1)).long()
            X_all = torch.from_numpy(np.asarray(splits["X_train"])).float()
            use_rows = sigma_rows is not None and sigma_rows.shape[0] == X_all.shape[0]
            if use_rows:
                loader = DataLoader(
                    TensorDataset(
                        X_all,
                        y_idx,
                        torch.arange(X_all.shape[0]),
                        sigma_rows.detach().cpu(),
                    ),
                    batch_size=config.batch_size,
                    shuffle=True,
                )
            else:
                loader = DataLoader(
                    TensorDataset(X_all, y_idx),
                    batch_size=config.batch_size,
                    shuffle=True,
                )
            opt = torch.optim.Adam(model.parameters(), lr=config.lr)
            model.train()
            for _ in range(config.epochs):
                if use_rows:
                    for Xb, yb, _idx, sb in loader:
                        opt.zero_grad(set_to_none=True)
                        loss = loss_fn(Xb.to(dev), yb.to(dev), sigma_x=sb.to(dev))
                        loss.backward()
                        opt.step()
                    continue
                for Xb, yb in loader:
                    opt.zero_grad(set_to_none=True)
                    loss = loss_fn(Xb.to(dev), yb.to(dev))
                    loss.backward()
                    opt.step()
            model.eval()
            result = {
                "type": "binned_eiv",
                "model": model,
                "centers": torch.from_numpy(centers).float(),
            }
            if _X_test_true is not None:
                result["_X_test_true"] = _X_test_true
            return result

        elif method == "LatentMarginalization":
            # Analytic marginalization over latent clean inputs (Gaussian).
            # Prior moments from the noisy train inputs (empirical Bayes).
            model = _MLP(in_dim, hidden, out_dim)
            prior_mean = X.mean(dim=0)
            prior_cov = torch.diag(X.var(dim=0, unbiased=False).clamp_min(1e-6))
            loss_fn = LatentMarginalizationLoss(
                model=model,
                base_loss=WeightedMSELoss(),
                prior_mean=prior_mean,
                prior_cov=prior_cov,
                sigma_u=sigma_x,
            )
            _train_eiv(model, loss_fn, X, y, config)
            result = {"type": "torch", "model": model}
            if _X_test_true is not None:
                result["_X_test_true"] = _X_test_true
            return result

        elif method == "MSE_NoisyPredict":
            # Train with plain MSE; at inference time, wrap with NoisyInputPredictor
            # for MC input-noise averaging.  Tests whether EIV correction at
            # inference time alone is sufficient.
            model = _MLP(in_dim, hidden, out_dim)
            _train_one(model, WeightedMSELoss(), X, y, config)
            result = {"type": "torch", "model": model, "sigma_x": sigma_x}
            if sigma_test is not None:
                result["sigma_x_test"] = sigma_test
            if _X_test_true is not None:
                result["_X_test_true"] = _X_test_true
            return result

        elif method == "LatentNN":
            # LatentNN: jointly optimises model parameters AND per-sample latent
            # clean inputs under a Gaussian quadratic penalty.  This is the most
            # principled algorithm-level EIV approach — it explicitly learns the
            # true (unobserved) inputs during training.
            latent = LatentNN(
                model_factory=lambda: _MLP(in_dim, hidden, out_dim),
                loss_fn=torch.nn.MSELoss(),  # nn.MSELoss activates sigma_y scaling
                sigma_x=sigma_x,
                sigma_y=sigma_y,
                epochs=200,
                lr=config.lr,
                batch_size=config.batch_size,
            )
            latent.fit(X, y)
            result = {"type": "latentnn", "model": latent, "sigma_x": sigma_x}
            if _X_test_true is not None:
                result["_X_test_true"] = _X_test_true
            return result

        elif method == "NoiseAware":
            # NoiseAwareRegressor: encodes features alongside derived quality
            # signals (log-uncertainty, SNR, quality-gated features).  The
            # encoder learns to downweight low-SNR inputs via a learned gate.
            # forward(x, sigma_x) — wrapper bakes sigma_x for _train_one compat.
            nar = NoiseAwareRegressor(
                in_dim,
                out_dim,
                encoder_hidden_dim=hidden,
                backbone_hidden_dims=(hidden, hidden),
            )
            # Explicit device placement (the regressor does not self-place
            # like _MLP): without this, _train_one trains on CPU while
            # predict() moves X to the session device → device mismatch.
            nar.to(current_device())

            # Wrapper: makes forward(x) call nar(x, sigma_x) internally
            class _NARWrapper(torch.nn.Module):
                def __init__(self, nar):
                    super().__init__()
                    self.nar = nar

                def forward(self, x):
                    return self.nar(x, sigma_x)

            wrapped = _NARWrapper(nar)
            _train_one(wrapped, WeightedMSELoss(), X, y, config)
            result = {"type": "torch", "model": nar, "sigma_x": sigma_x}
            if _X_test_true is not None:
                result["_X_test_true"] = _X_test_true
            return result

        elif method == "SIMEX":
            # SIMEX: adds extra noise at multiple levels, retrains models,
            # extrapolates back to zero-error via polynomial fit.
            # predict() now matched-noise-correct: each λ model receives
            # test inputs perturbed by √λ·N(0, Σ_u).
            def _simex_train(m, X_data, y_data):
                m.train()
                opt = Adam(m.parameters(), lr=config.lr)
                dataset = TensorDataset(X_data, y_data)
                loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
                for _ in range(config.epochs):
                    for Xb, yb in loader:
                        opt.zero_grad()
                        loss = ((m(Xb) - yb) ** 2).mean()
                        loss.backward()
                        opt.step()
                m.eval()
                return m

            simex = SIMEX(
                model_factory=lambda: _MLP(in_dim, hidden, out_dim),
                train_func=_simex_train,
                sigma_u=sigma_x,
                n_simulations=5,
            )
            simex.fit(X, y)
            result = {"type": "simex", "model": simex}
            if _X_test_true is not None:
                result["_X_test_true"] = _X_test_true
            return result

        elif method == "RC":
            # RegressionCalibration: pre-processing step that estimates true
            # inputs via E[X|W] = mu_w + reliability @ (W - mu_w), then trains
            # an MLP on the calibrated inputs.
            rc = RegressionCalibration(sigma_u=sigma_x)
            rc.fit(X)
            X_cal_train = rc.transform(X)
            X_cal_test = rc.transform(
                torch.from_numpy(splits["X_test"]).float().to(current_device())
            )
            model = _MLP(in_dim, hidden, out_dim)
            _train_one(model, WeightedMSELoss(), X_cal_train, y, config)
            result = {"type": "rc", "model": model, "X_cal_test": X_cal_test}
            if _X_test_true is not None:
                result["_X_test_true"] = _X_test_true
            return result

        elif method == "SciPyODR":
            if not HAS_SCIPY_ODR:
                raise ImportError("scipy.odr unavailable (deprecated in scipy >= 1.17)")

            X_np = X_train_np.astype(np.float64)
            y_np = y_train_np.ravel().astype(np.float64)

            # Linear model: y = intercept + X @ coef. ODRPACK takes x as
            # (n_vars, n_obs), so the design matrix is transposed here; the
            # stored beta convention is unchanged (beta[0] + X @ beta[1:]).
            def _linear_fcn(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
                return beta[0] + np.dot(beta[1:], x)

            model = _ODRModel(_linear_fcn)
            data = RealData(x=X_np.T, y=y_np, sx=sigma_x, sy=sigma_y)

            # Initial guess from OLS
            X_with_bias = np.column_stack([np.ones(len(X_np), dtype=np.float64), X_np])
            beta0, _, _, _ = np.linalg.lstsq(X_with_bias, y_np, rcond=None)

            odr = _ODR(data, model, beta0=beta0)
            output = odr.run()
            return {"type": "scipy_odr", "coef": output.beta}

        elif method == "TLS":
            # Noise-weighted TLS (see tls_solve): same sigma_x / sigma_y
            # assumptions as the other EIV rows; raises if ill-conditioned.
            beta, intercept = tls_solve(X_train_np, y_train_np, sigma_x, sigma_y)
            return {"type": "tls", "coef": beta, "intercept": intercept}

        # --- Composability methods ---

        elif method == "GaussianNLL+InputNoiseMC":
            # Probabilistic EIV: GaussianNLL wrapped in input-noise MC.
            # Model outputs (mean, logvar); perturbations average the NLL.
            model = _MLP(in_dim, hidden, out_dim=2)
            loss_fn = InputNoiseAugmentationLoss(
                model=model,
                base_loss=GaussianNLLLoss(),
                sigma_x=sigma_x,
                n_samples=20,
            )
            _train_eiv(model, loss_fn, X, y, config, sigma_rows=sigma_rows, forward_sigma=True)
            state = {"type": "gaussian_mc", "model": model, "sigma_x": sigma_x}
            if sigma_test is not None:
                state["sigma_x_test"] = sigma_test
            return state

        elif method == "RC+GaussianNLL":
            # Pipeline: calibrate inputs via RC, then fit probabilistic model.
            rc = RegressionCalibration(sigma_u=sigma_x)
            rc.fit(X)
            X_cal_train = rc.transform(X)
            X_cal_test = rc.transform(
                torch.from_numpy(splits["X_test"]).float().to(current_device())
            )
            model = _MLP(in_dim, hidden, out_dim=2)
            _train_one(model, GaussianNLLLoss(), X_cal_train, y, config)
            return {"type": "rc_gaussian", "model": model, "X_cal_test": X_cal_test}

        elif method == "RC+FunctionalEIV":
            # Algorithmic chaining: does FunctionalEIV help if inputs are
            # already RC-debiased?
            rc = RegressionCalibration(sigma_u=sigma_x)
            rc.fit(X)
            X_cal_train = rc.transform(X)
            X_cal_test = rc.transform(
                torch.from_numpy(splits["X_test"]).float().to(current_device())
            )
            model = _MLP(in_dim, hidden, out_dim)
            # Small residual sigma_x after RC
            residual_sigma = sigma_x * 0.3
            loss_fn = FunctionalEIVLoss(
                model=model,
                sigma_x=residual_sigma,
                sigma_y=sigma_y,
            )
            _train_eiv(model, loss_fn, X_cal_train, y, config)
            result = {"type": "rc", "model": model, "X_cal_test": X_cal_test}
            if _X_test_true is not None:
                result["_X_test_true"] = _X_test_true
            return result

        elif method == "TLS_Init+FunctionalEIV":
            # Address attenuation bias: initialize with unbiased TLS β,
            # then fine-tune with FunctionalEIV. Tests whether the EIV
            # loss can preserve the TLS solution (vs sliding back to MSE β).
            beta, intercept = tls_solve(X_train_np, y_train_np, sigma_x, sigma_y)

            model = _Linear(in_dim, out_dim=1)
            with torch.no_grad():
                model.fc.weight.copy_(
                    torch.from_numpy(beta).float().to(current_device()).unsqueeze(0)
                )
                model.fc.bias.copy_(torch.tensor([intercept]).float())
            loss_fn = FunctionalEIVLoss(model=model, sigma_x=sigma_x, sigma_y=sigma_y)
            _train_eiv(
                model,
                loss_fn,
                X,
                y,
                config,
                lr=config.lr * 0.1,
                sigma_rows=sigma_rows,
                forward_sigma=True,
            )
            result = {"type": "torch", "model": model}
            if _X_test_true is not None:
                result["_X_test_true"] = _X_test_true
            return result

        elif method == "MSE_WarmupMC":
            # Pre-train with MSE for warmup_epochs to get into a reasonable basin,
            # then switch to FunctionalEIV_MC.  Uses torchregress's WarmupMCTrainer
            # (shared Adam optimizer across phases — momentum carries over).
            trainer = WarmupMCTrainer(
                model_factory=lambda: _MLP(in_dim, hidden, out_dim),
                sigma_x=sigma_x,
                sigma_y=sigma_y,
                total_epochs=config.epochs,
                warmup_epochs=config.warmup_epochs,
                lr=config.lr,
                batch_size=config.batch_size,
                n_mc_samples=20,
            )
            trainer.fit(X, y)
            model = trainer.model
            result = {"type": "torch", "model": model}
            if _X_test_true is not None:
                result["_X_test_true"] = _X_test_true
            return result

        elif method.startswith("SplitCP+"):
            # Split conformal prediction on top of an EIV model.
            # Carves a 25% calibration split from training data.
            sub_method = method.split("+", 1)[1]
            aug_splits = _carve_calibration_split(
                splits,
                cal_frac=0.25,
                seed=config.seed,
                smoke=False,
                suite_name="eiv",
            )
            for k in ("sigma_x", "sigma_y", "X_test_true", "_y_scaler"):
                if k in splits:
                    aug_splits[k] = splits[k]
            X_cal_np = aug_splits["X_cal"]
            y_cal_np = aug_splits["y_cal"]

            # Fit base model on reduced training set
            base_state = self.fit(sub_method, aug_splits, config)
            m = base_state.get("model")
            y_pred_cal = _eiv_predict(sub_method, base_state, m, X_cal_np, noisy=True)
            residuals = np.abs(y_cal_np.ravel() - y_pred_cal)
            q_90 = float(np.quantile(residuals, 0.90))
            base_state["type"] = f"conformal_{base_state.get('type', 'torch')}"
            base_state["q_90"] = q_90
            base_state["sub_method"] = sub_method
            return base_state

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
        m = model_state.get("model")
        model_type = model_state.get("type", "torch")
        result: dict[str, np.ndarray] = {}

        # --- Conformal prediction path ---
        if model_type.startswith("conformal_"):
            sub_method = model_state["sub_method"]
            y_pred = _eiv_predict(sub_method, model_state, m, X_test, noisy=True)
            q_90 = model_state["q_90"]
            result["y_pred"] = y_pred
            result["lower"] = y_pred - q_90
            result["upper"] = y_pred + q_90
            # Clean-X: CP intervals don't apply (calibrated on noisy residuals),
            # but point predictions on clean X measure attenuation bias
            X_test_true = model_state.get("_X_test_true")
            if X_test_true is not None:
                result["y_pred_clean"] = _eiv_predict(
                    sub_method,
                    model_state,
                    m,
                    X_test_true,
                    noisy=False,
                )
            return result

        # --- Distributional path: GaussianNLL+InputNoiseMC ---
        if model_type == "gaussian_mc":
            n_rows = np.asarray(X_test).shape[0]
            sigma_x_val = _test_sigma(model_state, n_rows)
            predictor = NoisyInputPredictor(m, sigma_x=sigma_x_val, n_samples=32)

            def _moment_match(X_np):
                with torch.no_grad():
                    X_t = torch.from_numpy(X_np).float().to(current_device())
                    raw = predictor.sample_predictions(X_t)
                    means = raw[..., 0]
                    logvars = raw[..., 1]
                    vars_ = torch.exp(logvars)
                    mean_final = means.mean(dim=0)
                    var_final = (vars_ + means**2).mean(dim=0) - mean_final**2
                return (
                    mean_final.squeeze(-1).detach().cpu().numpy(),
                    torch.sqrt(var_final.clamp_min(1e-8)).squeeze(-1).detach().cpu().numpy(),
                )

            mean_np, std_np = _moment_match(X_test)
            result["mean"] = mean_np
            result["std"] = std_np
            result["y_pred"] = mean_np
            # Clean-X evaluation for distributional methods
            X_test_true = model_state.get("_X_test_true")
            if X_test_true is not None:
                result["y_pred_clean"] = _moment_match(X_test_true)[0]
            return result

        # --- Distributional path: RC+GaussianNLL ---
        if model_type == "rc_gaussian":
            X_cal = model_state["X_cal_test"]
            with torch.no_grad():
                out = m(X_cal)
                mean = out[:, 0]
                std = torch.exp(out[:, 1] * 0.5)
            result["mean"] = mean.squeeze(-1).detach().cpu().numpy()
            result["std"] = std.squeeze(-1).detach().cpu().numpy()
            result["y_pred"] = result["mean"]
            # Clean-X: run model directly on clean X (no RC calibration needed)
            X_test_true = model_state.get("_X_test_true")
            if X_test_true is not None:
                with torch.no_grad():
                    X_t = torch.from_numpy(X_test_true).float().to(current_device())
                    out_clean = m(X_t)
                result["y_pred_clean"] = out_clean[:, 0].squeeze(-1).detach().cpu().numpy()
            return result

        # --- Standard point-prediction path ---
        y_pred = _eiv_predict(method, model_state, m, X_test, noisy=True)
        result["y_pred"] = y_pred

        # Clean-X evaluation: shows attenuation bias correction
        X_test_true = model_state.get("_X_test_true")
        if X_test_true is not None:
            result["y_pred_clean"] = _eiv_predict(
                method,
                model_state,
                m,
                X_test_true,
                noisy=False,
            )

        return result

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
        del method, config
        from harness.metrics import compute_all_metrics
        from harness.runners import _inverse_transform_predictions, _inverse_transform_y

        y_scaler = predictions.get("_y_scaler")
        if y_scaler is not None:
            predictions = _inverse_transform_predictions(predictions, y_scaler)
            y_test = _inverse_transform_y(y_test, y_scaler)
            # Also inverse-transform clean predictions if present
            if "y_pred_clean" in predictions:
                predictions["y_pred_clean"] = _inverse_transform_y(
                    predictions.pop("y_pred_clean"),
                    y_scaler,
                )

        # compute_all_metrics handles y_pred, mean/std, lower/upper, samples
        # and produces RMSE, MAE, R2, NLL, Coverage, IntervalWidth, PIT, etc.
        result = compute_all_metrics(predictions, y_test)

        # Clean-X metrics: evaluate on X_test_true to measure attenuation bias
        if "y_pred_clean" in predictions:
            y_pred_clean = np.asarray(predictions["y_pred_clean"]).reshape(-1)
            y_true_np = np.asarray(y_test).reshape(-1)
            if len(y_pred_clean) != len(y_true_np):
                # Smoke-subsampled targets vs full clean predictions (or vice
                # versa): align on the overlapping prefix deterministically
                # rather than failing the row.
                n = min(len(y_pred_clean), len(y_true_np))
                y_pred_clean, y_true_np = y_pred_clean[:n], y_true_np[:n]
            result["RMSE_clean"] = float(np.sqrt(np.mean((y_pred_clean - y_true_np) ** 2)))
            ss_res = float(np.sum((y_true_np - y_pred_clean) ** 2))
            ss_tot = float(np.sum((y_true_np - y_true_np.mean()) ** 2))
            result["R2_clean"] = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

        # Attenuation diagnostics (library): slope of prediction on truth.
        # Values near 1 = no attenuation; < 1 = compressed dynamic range
        # (the textbook EIV failure mode); reported on noisy and clean
        # inputs alike.
        try:
            from torchregress.metrics import attenuation_factor as _att

            y_pred_all = np.asarray(predictions["y_pred"]).reshape(-1)
            y_true_all = np.asarray(y_test).reshape(-1)
            n = min(len(y_pred_all), len(y_true_all))
            result["Attenuation"] = float(_att(y_pred_all[:n], y_true_all[:n]))
            if "y_pred_clean" in predictions:
                yc = np.asarray(predictions["y_pred_clean"]).reshape(-1)
                n = min(len(yc), len(y_true_all))
                result["Attenuation_clean"] = float(_att(yc[:n], y_true_all[:n]))
        except Exception:
            pass

        # Uncertain-ground-truth report (library): fires only when the
        # dataset supplies per-row target std (stashed as y_std_test by
        # run()) AND the row emits a predictive std.
        try:
            if "y_std_test" in predictions and "std" in predictions:
                from torchregress.metrics import (
                    uncertain_gt_metrics_report as _ugr,
                )

                officiated = np.asarray(predictions["y_std_test"], dtype=np.float64).reshape(-1)
                pm = np.asarray(predictions["y_pred"], dtype=np.float64).reshape(-1)
                ps = np.asarray(predictions["std"], dtype=np.float64).reshape(-1)
                yt = np.asarray(y_test, dtype=np.float64).reshape(-1)
                n = min(len(pm), len(ps), len(yt), len(officiated))
                rep = _ugr(
                    pred_mean=torch.from_numpy(pm[:n]),
                    pred_variance=torch.from_numpy((ps[:n] ** 2).clip(min=1e-12)),
                    target=torch.from_numpy(yt[:n]),
                    target_variance=torch.from_numpy((officiated[:n] ** 2).clip(min=0.0)),
                )
                for _k, _v in rep.items():
                    try:
                        result[f"UGT_{_k}"] = float(
                            np.asarray(_v, dtype=np.float64).reshape(-1)[0]
                            if np.asarray(_v).size > 1
                            else float(_v)
                        )
                    except Exception:
                        continue
        except Exception:
            pass

        return result

    # ------------------------------------------------------------------
    # notes
    # ------------------------------------------------------------------

    def method_notes(self, method: str) -> str:
        notes: dict[str, str] = {
            "MSE": "MLP + WeightedMSELoss — plain baseline (no EIV correction)",
            "FunctionalEIV": (
                "MLP + FunctionalEIVLoss — analytical gradient-based variance propagation"
            ),
            "FunctionalEIV_MC": (
                "MLP + FunctionalEIVLoss(monte_carlo=True) — MC sample covariance, no Jacobian"
            ),
            "StructuralEIV": ("MLP + StructuralEIVLoss — full cross-covariance structural model"),
            "ODR": ("MLP + OrthogonalDistanceRegressionLoss — iterative latent-x optimisation"),
            "EnsembleEIV": ("MLP + EnsembleEIVLoss — Monte-Carlo input perturbation averaging"),
            "TLS": (
                "Noise-weighted Total Least Squares via SVD (columns scaled by sigma_x / sigma_y; "
                "Deming for one feature) — self-contained, numpy-only; reported as skipped "
                "when ill-conditioned (collinear features)"
            ),
            "SciPyODR": (
                "scipy.odr.ODR — classical orthogonal distance regression (external, linear model)"
            ),
            "InputNoiseMC": (
                "MLP + InputNoiseMarginalizationLoss(MSE) — proper MC marginalisation over input noise"
            ),
            "InputNoiseMDN": ("MDN head + InputNoiseMDNLoss — multimodal EIV marginalization"),
            "InputNoiseBinnedPDF": (
                "MLP + InputNoiseBinnedPDFLoss — nonparametric binned-PDF marginalization"
            ),
            "LatentMarginalization": (
                "MLP + LatentMarginalizationLoss — analytic Gaussian marginalization"
            ),
            "MSE_NoisyPredict": (
                "MSE-trained MLP + NoisyInputPredictor — MC averaging at inference time only"
            ),
            "LatentNN": (
                "LatentNN — jointly optimises model + per-sample latent clean inputs (Gaussian penalty); test-time MC marginalization"
            ),
            "NoiseAware": (
                "NoiseAwareRegressor — error-aware feature encoder (SNR, quality gates) + MLP backbone"
            ),
            "SIMEX": (
                "SIMEX — simulation-extrapolation with matched-noise prediction fix (perturbs test inputs per λ level)"
            ),
            "RC": (
                "RegressionCalibration — pre-processing input calibration via reliability matrix"
            ),
        }
        notes.update(
            {
                "GaussianNLL+InputNoiseMC": (
                    "MLP(mean+logvar) + InputNoiseMC→GaussianNLL — MC perturbation "
                    "of probabilistic model; test-time moment-matching"
                ),
                "RC+GaussianNLL": (
                    "RC pre-processing + GaussianNLL on calibrated inputs — "
                    "pipeline approach to probabilistic EIV"
                ),
                "RC+FunctionalEIV": (
                    "RC pre-processing + FunctionalEIVLoss on calibrated inputs — "
                    "algorithmic chaining: does EIV loss help after RC?"
                ),
                "TLS_Init+FunctionalEIV": (
                    "Linear model initialised to TLS solution → FunctionalEIV "
                    "fine-tune — ⚠️ INVESTIGATED: TLS β is NOT preserved; "
                    "β drifts completely to OLS/MSE within ~50 epochs. "
                    "Prefer RC+FunctionalEIV instead."
                ),
                "MSE_WarmupMC": (
                    "MSE pre-training (config.warmup_epochs, default 20) → FunctionalEIV_MC — "
                    "warmup avoids early-phase MC gradient noise; designed for "
                    "linear data where MC is competitive"
                ),
                "SplitCP+FunctionalEIV": (
                    "FunctionalEIV + split-conformal intervals (90% q_quantile "
                    "on 25% held-out calibration residuals)"
                ),
                "SplitCP+InputNoiseMC": (
                    "InputNoiseMC + split-conformal intervals (90% q_quantile "
                    "on 25% held-out calibration residuals)"
                ),
                "SplitCP+MSE_WarmupMC": (
                    "MSE_WarmupMC + split-conformal intervals (90% q_quantile "
                    "on 25% held-out calibration residuals) — tests whether "
                    "warmup improves conformal interval efficiency on linear data"
                ),
            }
        )
        base = notes.get(method, "")
        if method in _PER_OBS_METHODS:
            base += " [per-obs σ on heteroscedastic sets, per-band fallback]"
        return base
