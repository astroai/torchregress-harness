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
- TLS              : Total Least Squares via SVD (numpy-only)
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
    OrthogonalDistanceRegressionLoss,
    StructuralEIVLoss,
    WeightedMSELoss,
)
from torchregress.losses.eiv import NoisyInputPredictor

from harness.core import set_harness_seed
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _MLP, _carve_calibration_split, _hidden_size, _Linear, _train_one

# --- scipy.odr (deprecated in scipy >= 1.17; check gracefully) ---
try:
    from scipy.odr import ODR as _ODR
    from scipy.odr import Model as _ODRModel
    from scipy.odr import RealData

    # Verify it actually works with sx/sy parameters (it may be broken
    # in the deprecation window, e.g. scipy >= 1.17 on macOS).
    _x_test = np.random.randn(10, 3).astype(np.float64)
    _y_test = np.random.randn(10).astype(np.float64)
    _model_test = _ODRModel(lambda beta, x: beta[0] + x @ beta[1:])
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
) -> None:
    """Mini-batch training loop for EIV losses (loss_fn takes x_obs, y_obs)."""
    if config is not None:
        n_epochs = epochs if epochs is not None else config.epochs
        bs = batch_size if batch_size is not None else config.batch_size
        learning_rate = lr if lr is not None else config.lr
    else:
        n_epochs = epochs or 50
        bs = batch_size or 64
        learning_rate = lr or 1e-3

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=False)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model.train()
    for _epoch in range(n_epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            # EIV losses: forward(x_obs, target) — the loss internally calls model(x_obs)
            loss = loss_fn(X_batch, y_batch)
            loss.backward()
            optimizer.step()
    model.eval()


# --- Shared prediction helper (noisy and clean X paths) ---


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
        sigma_x_val = float(model_state.get("sigma_x", 0.2))
        with torch.no_grad():
            X = torch.from_numpy(X_np).float()
            return m.predict(X, sigma_x=sigma_x_val).squeeze(-1).numpy()

    if model_type == "simex":
        with torch.no_grad():
            X = torch.from_numpy(X_np).float()
            return m.predict(X).squeeze(-1).numpy()

    if model_type == "rc":
        if noisy:
            X_cal = model_state["X_cal_test"]
            with torch.no_grad():
                return m(X_cal).squeeze(-1).numpy()
        # Clean X doesn't need RC calibration — use model directly
        with torch.no_grad():
            X = torch.from_numpy(X_np).float()
            return m(X).squeeze(-1).numpy()

    if method == "MSE_NoisyPredict" and noisy:
        sigma_x_val = float(model_state.get("sigma_x", 0.2))
        noisy_pred = NoisyInputPredictor(m, sigma_x=sigma_x_val, n_samples=16)
        with torch.no_grad():
            X = torch.from_numpy(X_np).float()
            return noisy_pred(X).squeeze(-1).numpy()

    # Torch models
    with torch.no_grad():
        X = torch.from_numpy(X_np).float()
        if method == "NoiseAware":
            # NoiseAware always needs sigma_x (architectural requirement)
            sigma_x_val = float(model_state.get("sigma_x", 0.2))
            return m(X, sigma_x_val).squeeze(-1).numpy()
        return m(X).squeeze(-1).numpy()


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

        X = torch.from_numpy(X_train_np).float()
        y = torch.from_numpy(y_train_np).float()
        if y.dim() == 1:
            y = y.unsqueeze(1)

        # Known noise parameters from the dataset generator (fallback matches
        # the defaults in _load_synthetic_eiv: sigma_x=0.2, sigma_y=0.1).
        sigma_x = float(splits.get("sigma_x", 0.2))
        sigma_y = float(splits.get("sigma_y", 0.1))

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
            _train_eiv(model, loss_fn, X, y, config)
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
            _train_eiv(model, loss_fn, X, y, config)
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
            _train_eiv(model, loss_fn, X, y, config)
            result = {"type": "torch", "model": model}
            if _X_test_true is not None:
                result["_X_test_true"] = _X_test_true
            return result

        elif method == "ODR":
            model = _MLP(in_dim, hidden, out_dim)
            loss_fn = OrthogonalDistanceRegressionLoss(
                model=model, sigma_x=sigma_x, sigma_y=sigma_y
            )
            _train_eiv(model, loss_fn, X, y, config)
            result = {"type": "torch", "model": model}
            if _X_test_true is not None:
                result["_X_test_true"] = _X_test_true
            return result

        elif method == "EnsembleEIV":
            model = _MLP(in_dim, hidden, out_dim)
            loss_fn = EnsembleEIVLoss(model=model, sigma_x=sigma_x)
            _train_eiv(model, loss_fn, X, y, config)
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
            X_cal_train = rc.fit_transform(X)
            X_cal_test = rc.transform(torch.from_numpy(splits["X_test"]).float())
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

            # Linear model: y = intercept + X @ coef
            def _linear_fcn(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
                return beta[0] + np.dot(x, beta[1:])

            model = _ODRModel(_linear_fcn)
            data = RealData(x=X_np, y=y_np, sx=sigma_x, sy=sigma_y)

            # Initial guess from OLS
            X_with_bias = np.column_stack([np.ones(len(X_np), dtype=np.float64), X_np])
            beta0, _, _, _ = np.linalg.lstsq(X_with_bias, y_np, rcond=None)

            odr = _ODR(data, model, beta0=beta0)
            output = odr.run()
            return {"type": "scipy_odr", "coef": output.beta}

        elif method == "TLS":
            # Total Least Squares via SVD — minimises orthogonal distances
            X_np = X_train_np.astype(np.float64)
            y_np = y_train_np.ravel().astype(np.float64)
            # Centre the data
            X_mean = X_np.mean(axis=0)
            y_mean = y_np.mean()
            X_c = X_np - X_mean
            y_c = y_np - y_mean
            # SVD of the augmented matrix [X_c | y_c]
            Z = np.column_stack([X_c, y_c])
            _, _, Vt = np.linalg.svd(Z, full_matrices=False)
            # Solution from the right singular vector of the smallest singular value
            V = Vt[-1]  # last row = smallest singular vector
            beta = -V[:-1] / V[-1]
            intercept = y_mean - X_mean @ beta
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
            _train_eiv(model, loss_fn, X, y, config)
            return {"type": "gaussian_mc", "model": model, "sigma_x": sigma_x}

        elif method == "RC+GaussianNLL":
            # Pipeline: calibrate inputs via RC, then fit probabilistic model.
            rc = RegressionCalibration(sigma_u=sigma_x)
            X_cal_train = rc.fit_transform(X)
            X_cal_test = rc.transform(torch.from_numpy(splits["X_test"]).float())
            model = _MLP(in_dim, hidden, out_dim=2)
            _train_one(model, GaussianNLLLoss(), X_cal_train, y, config)
            return {"type": "rc_gaussian", "model": model, "X_cal_test": X_cal_test}

        elif method == "RC+FunctionalEIV":
            # Algorithmic chaining: does FunctionalEIV help if inputs are
            # already RC-debiased?
            rc = RegressionCalibration(sigma_u=sigma_x)
            X_cal_train = rc.fit_transform(X)
            X_cal_test = rc.transform(torch.from_numpy(splits["X_test"]).float())
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
            X_np = X_train_np.astype(np.float64)
            y_np = y_train_np.ravel().astype(np.float64)
            X_mean = X_np.mean(axis=0)
            y_mean = y_np.mean()
            Z = np.column_stack([X_np - X_mean, y_np - y_mean])
            _, _, Vt = np.linalg.svd(Z, full_matrices=False)
            V = Vt[-1]
            beta = -V[:-1] / V[-1]
            intercept = y_mean - X_mean @ beta

            model = _Linear(in_dim, out_dim=1)
            with torch.no_grad():
                model.fc.weight.copy_(torch.from_numpy(beta).float().unsqueeze(0))
                model.fc.bias.copy_(torch.tensor([intercept]).float())
            loss_fn = FunctionalEIVLoss(model=model, sigma_x=sigma_x, sigma_y=sigma_y)
            _train_eiv(model, loss_fn, X, y, config, lr=config.lr * 0.1)
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
            sigma_x_val = float(model_state.get("sigma_x", 0.2))
            predictor = NoisyInputPredictor(m, sigma_x=sigma_x_val, n_samples=32)

            def _moment_match(X_np):
                with torch.no_grad():
                    X_t = torch.from_numpy(X_np).float()
                    raw = predictor.sample_predictions(X_t)
                    means = raw[..., 0]
                    logvars = raw[..., 1]
                    vars_ = torch.exp(logvars)
                    mean_final = means.mean(dim=0)
                    var_final = (vars_ + means**2).mean(dim=0) - mean_final**2
                return (
                    mean_final.squeeze(-1).numpy(),
                    torch.sqrt(var_final.clamp_min(1e-8)).squeeze(-1).numpy(),
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
            result["mean"] = mean.squeeze(-1).numpy()
            result["std"] = std.squeeze(-1).numpy()
            result["y_pred"] = result["mean"]
            # Clean-X: run model directly on clean X (no RC calibration needed)
            X_test_true = model_state.get("_X_test_true")
            if X_test_true is not None:
                with torch.no_grad():
                    X_t = torch.from_numpy(X_test_true).float()
                    out_clean = m(X_t)
                result["y_pred_clean"] = out_clean[:, 0].squeeze(-1).numpy()
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
            result["RMSE_clean"] = float(np.sqrt(np.mean((y_pred_clean - y_true_np) ** 2)))
            ss_res = float(np.sum((y_true_np - y_pred_clean) ** 2))
            ss_tot = float(np.sum((y_true_np - y_true_np.mean()) ** 2))
            result["R2_clean"] = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

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
                "Total Least Squares via SVD — minimises orthogonal distances (self-contained, numpy-only)"
            ),
            "SciPyODR": (
                "scipy.odr.ODR — classical orthogonal distance regression (external, linear model)"
            ),
            "InputNoiseMC": (
                "MLP + InputNoiseMarginalizationLoss(MSE) — proper MC marginalisation over input noise"
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
        return notes.get(method, "")
