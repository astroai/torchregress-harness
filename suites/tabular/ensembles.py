"""Ensembles suite: torchregress deep ensembles vs sklearn ensemble regressors.

Compares independently-trained deep ensembles (5 MLPs per method) against
sklearn's ``VotingRegressor``, ``StackingRegressor``, and ``BaggingRegressor``
on standard tabular datasets.  Deep ensembles provide both point predictions
and uncertainty estimates (via predictive variance across members).

Canonical scenarios
-------------------
* ``sklearn_diabetes_reference``
  https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
* ``sklearn_synthetic_reference``
  https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html

Architecture
------------
Each torchregress deep ensemble trains *n_ensemble* identical-architecture
MLPs independently (different random seeds), then combines their outputs:

* ``DeepEnsemble_MSE`` → mean of point predictions → ``{"y_pred"}``
* ``DeepEnsemble_GaussianNLL`` → mixture mean & total variance → ``{"mean", "std"}``
* ``DeepEnsemble_MultiQuantile`` → mean of quantile outputs → ``{"y_pred", "lower", "upper"}``

Sklearn ensembles use the native ``VotingRegressor`` / ``StackingRegressor`` /
``BaggingRegressor`` APIs with default hyperparameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Sklearn ensembles — hard dependency (sklearn is core)
# ---------------------------------------------------------------------------
from sklearn.ensemble import (
    BaggingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# ---------------------------------------------------------------------------
# Torchregress losses
# ---------------------------------------------------------------------------
from torchregress.losses import MultiQuantileLoss

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import (
    _MLP,
    _hidden_size,
    _train_one,
)

# ---------------------------------------------------------------------------
# Ensemble size
# ---------------------------------------------------------------------------

_N_ENSEMBLE = 5  # number of independent MLPs per deep ensemble


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class EnsemblesSuite(HarnessRunner):
    """Deep ensemble benchmark suite.

    Compares torchregress deep ensembles (5 MLPs each) against sklearn's
    VotingRegressor, StackingRegressor, and BaggingRegressor.
    """

    @property
    def name(self) -> str:
        return "ensembles"

    @property
    def description(self) -> str:
        return "Deep ensembles (torchregress) vs Voting / Stacking / Bagging (sklearn)"

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "california_housing",
            "diabetes",
            "make_regression_synthetic",
            "synthetic_hetero",
            "synthetic_outliers",
            "tabred_sberbank",
            "tabred_cooking",
            "tabred_delivery",
            "tabred_maps",
            "tabred_weather",
        ]

    @property
    def canonical_scenarios(self) -> list[str]:
        """sklearn canonical regression benchmarks."""
        return ["sklearn_diabetes_reference", "sklearn_synthetic_reference"]

    # ------------------------------------------------------------------
    # Method registry
    # ------------------------------------------------------------------

    _TR_METHODS: list[str] = [
        "DeepEnsemble_MSE",
        "DeepEnsemble_GaussianNLL",
        "DeepEnsemble_MultiQuantile",
        "MDNEnsemble",
        "LaplaceMAP",
        "Snapshot",
        "MultiSWAG",
        "BinnedPDFEnsemble",
        "HeteroscedasticLaplace",
        "VIDS",
        "IVON",
        "BatchEnsemble",
        "MeanOnlyBatchEnsemble",
        "Stacking",
    ]

    _SKLEARN_METHODS: list[str] = [
        "sklearn_VotingRegressor",
        "sklearn_StackingRegressor",
        "sklearn_BaggingRegressor",
    ]

    def methods(self) -> list[str]:
        return self._TR_METHODS + self._SKLEARN_METHODS

    def library_name(self, method: str) -> str:
        if method.startswith("sklearn_"):
            return "sklearn"
        return "torchregress"

    def method_notes(self, method: str) -> str:
        notes = {
            "DeepEnsemble_MSE": f"{_N_ENSEMBLE}× MLP + MSE → mean of point predictions",
            "DeepEnsemble_GaussianNLL": f"{_N_ENSEMBLE}× MLP + BetaNLL → mixture mean & std",
            "DeepEnsemble_MultiQuantile": f"{_N_ENSEMBLE}× MLP + MultiQuantile → mean of quantiles",
            "MDNEnsemble": "3× MDN heads (K=2) → mixture-mean point prediction",
            "LaplaceMAP": "MAP MLP + FullNetworkLaplace post-hoc (mean, var, intervals)",
            "Snapshot": "cyclical snapshots (3) → mean/var predictive intervals",
            "MultiSWAG": "3× heteroscedastic SWAGs → mean/epistemic/aleatoric split",
            "BinnedPDFEnsemble": "2× binned-PDF heads → ensemble mean/var over 10 bins",
            "HeteroscedasticLaplace": "last-layer Laplace on frozen MSE features (tuple head)",
            "VIDS": "adaptive-prior VI (bootstrap environments) → mean/std intervals",
            "IVON": "natural-gradient VI optimizer + posterior MC sampling",
            "BatchEnsemble": "BatchEnsembleRegressor over BatchEnsembleMLPBackbone (4 members, heteroscedastic) → epi+alea intervals",
            "MeanOnlyBatchEnsemble": "BatchEnsembleRegressor mean-only (4 members, epistemic variance only)",
            "Stacking": "torchregress StackingEnsemble: 3 MSE MLPs → Linear meta-learner",
            "sklearn_VotingRegressor": "Voting(LinearRegression, RandomForest, SVR, KNeighbors)",
            "sklearn_StackingRegressor": "Stacking(LinearRegression, RandomForest, SVR, KNeighbors) → RidgeCV",
            "sklearn_BaggingRegressor": "Bagging(DecisionTreeRegressor, n=100)",
        }
        return notes.get(method, "")

    # ------------------------------------------------------------------
    # fit — train n_ensemble models independently
    # ------------------------------------------------------------------

    def fit(
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)

        X_train_np = splits["X_train"]
        y_train_np = splits["y_train"].reshape(-1)
        in_dim = X_train_np.shape[1]
        hidden = _hidden_size(in_dim)

        # --- torchregress deep ensembles ---
        if method in self._TR_METHODS:
            from harness.wrappers import DeepEnsembleRegressor, HeteroscedasticEnsembleRegressor

            # Identify active device
            device = str(current_device())

            if method == "DeepEnsemble_MSE":
                reg = DeepEnsembleRegressor(
                    ensemble_size=_N_ENSEMBLE,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    lr=config.lr,
                    seed=config.seed,
                ).to(device)
                reg.fit(X_train_np, y_train_np)
                return {"type": "wrapper", "model": reg}

            elif method == "DeepEnsemble_GaussianNLL":
                reg = HeteroscedasticEnsembleRegressor(
                    ensemble_size=_N_ENSEMBLE,
                    beta=0.5,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    lr=config.lr,
                    seed=config.seed,
                ).to(device)
                reg.fit(X_train_np, y_train_np)
                return {"type": "wrapper", "model": reg}

            elif method == "DeepEnsemble_MultiQuantile":
                models: list[nn.Module] = []
                for i in range(_N_ENSEMBLE):
                    member_seed = config.seed + i * 1000
                    set_harness_seed(member_seed)
                    X_t = torch.from_numpy(X_train_np).float().to(current_device())
                    y_t = torch.from_numpy(y_train_np).float().to(current_device()).unsqueeze(1)
                    quantiles = [0.1, 0.5, 0.9]
                    model = _MLP(in_dim, hidden, len(quantiles))
                    loss_fn = MultiQuantileLoss(quantiles=quantiles)
                    _train_one(model, loss_fn, X_t, y_t, config)
                    model.eval()
                    models.append(model)
                return {"type": "multiquantile", "models": models, "quantiles": [0.1, 0.5, 0.9]}

            elif method == "MDNEnsemble":
                from torch.utils.data import DataLoader, TensorDataset
                from torchregress.ensemble import MDNEnsembleModel
                from torchregress.losses import MixtureDensityLoss

                mdn_loss = MixtureDensityLoss(n_components=2, n_features=1)
                ens = MDNEnsembleModel(
                    _MLP,
                    ensemble_size=3,
                    n_components=2,
                    n_features=1,
                    in_dim=in_dim,
                    hidden=hidden,
                    out_dim=mdn_loss.expected_output_size,
                )
                loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(X_train_np).float(),
                        torch.from_numpy(y_train_np).float().unsqueeze(1),
                    ),
                    batch_size=config.batch_size,
                    shuffle=True,
                )
                ens.fit(
                    loader,
                    mdn_loss,
                    epochs=config.epochs,
                    verbose=False,
                    device=str(current_device()),
                )
                return {"type": "mdn_ensemble", "model": ens, "loss_fn": mdn_loss}
            elif method == "LaplaceMAP":
                import torch.optim as optim
                from torch.utils.data import DataLoader, TensorDataset
                from torchregress.ensemble import FullNetworkLaplace

                model = _MLP(in_dim, hidden, out_dim=1)
                opt = optim.Adam(model.parameters(), lr=config.lr)
                loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(X_train_np).float(),
                        torch.from_numpy(y_train_np).float().unsqueeze(1),
                    ),
                    batch_size=config.batch_size,
                    shuffle=True,
                )
                model.train()
                for _ in range(config.epochs):
                    for Xb, yb in loader:
                        Xb, yb = Xb.to(current_device()), yb.to(current_device())
                        opt.zero_grad(set_to_none=True)
                        nn.MSELoss()(model(Xb), yb).backward()
                        opt.step()
                model.eval()
                lap = FullNetworkLaplace(model.to(current_device()))
                lap.fit(loader, nn.MSELoss(reduction="none"))
                return {"type": "laplace", "model": lap, "base": model}

            elif method == "Snapshot":
                from torch.utils.data import DataLoader, TensorDataset
                from torchregress.ensemble import SnapshotEnsemble

                loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(X_train_np).float(),
                        torch.from_numpy(y_train_np).float().unsqueeze(1),
                    ),
                    batch_size=config.batch_size,
                    shuffle=True,
                )
                sn = SnapshotEnsemble(
                    _MLP(in_dim, hidden, out_dim=1),
                    n_snapshots=3,
                    cycle_epochs=max(1, config.epochs // 3),
                )
                sn.fit(loader, epochs=config.epochs, device=str(current_device()))
                return {"type": "snapshot", "model": sn}

            elif method == "MultiSWAG":
                import torch.optim as optim
                from torch.utils.data import DataLoader, TensorDataset
                from torchregress.ensemble.swag import MultiSWAG
                from torchregress.losses import BetaNLLLoss

                loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(X_train_np).float(),
                        torch.from_numpy(y_train_np).float().unsqueeze(1),
                    ),
                    batch_size=config.batch_size,
                    shuffle=True,
                )
                mswag = MultiSWAG(_MLP(in_dim, hidden, out_dim=2), n_models=3, max_num_models=8)
                # Per-member seeds: re-init each SWAG base before training.
                for k, swag in enumerate(mswag.swag_models):
                    set_harness_seed(config.seed + k * 1000 + 7)
                    for mod in swag.base_model.modules() if hasattr(swag, "base_model") else []:
                        if isinstance(mod, nn.Linear):
                            mod.reset_parameters()
                for k, swag in enumerate(mswag.swag_models):
                    set_harness_seed(config.seed + k * 1000 + 7)
                    base = swag.base_model if hasattr(swag, "base_model") else None
                    trainable = base if base is not None else mswag
                    opt = optim.Adam(
                        (
                            trainable.parameters()
                            if hasattr(trainable, "parameters")
                            else mswag.parameters()
                        ),
                        lr=config.lr,
                    )
                    loss_fn = BetaNLLLoss(beta=0.5)
                    dev = current_device()
                    if base is not None:
                        base.to(dev)
                    else:
                        mswag.to(dev)
                    for _ in range(config.epochs):
                        for Xb, yb in loader:
                            Xb, yb = Xb.to(dev), yb.to(dev)
                            opt.zero_grad(set_to_none=True)
                            out = (base if base is not None else mswag)(Xb)
                            BetaNLLLoss(beta=0.5)(out, yb).backward()
                            opt.step()
                    # Collect the trained weights into this member's SWAG.
                    try:
                        swag.collect_model(base if base is not None else mswag)
                    except Exception:
                        pass
                mswag.eval()
                return {"type": "multiswag", "model": mswag}

            elif method == "BinnedPDFEnsemble":
                from torch.utils.data import DataLoader, TensorDataset
                from torchregress.ensemble import BinnedPDFEnsembleModel

                n_bins = 10
                y_np = np.asarray(y_train_np).reshape(-1)
                edges = np.quantile(y_np, np.linspace(0, 1, n_bins + 1)).astype(np.float32)
                be = BinnedPDFEnsembleModel(
                    _MLP, ensemble_size=2, in_dim=in_dim, hidden=hidden, out_dim=n_bins
                )
                # CrossEntropy needs class indices: reuse the quantile bins.
                y_idx = torch.from_numpy(
                    np.clip(np.digitize(y_np, edges) - 1, 0, n_bins - 1)
                ).long()
                loader = DataLoader(
                    TensorDataset(torch.from_numpy(X_train_np).float(), y_idx),
                    batch_size=config.batch_size,
                    shuffle=True,
                )
                be.fit(loader, nn.CrossEntropyLoss(), epochs=config.epochs, verbose=False)
                # Store binning for expected-value predict.
                qs = np.linspace(0, 1, n_bins + 1)
                centers = ((np.quantile(y_np, qs)[:-1] + np.quantile(y_np, qs)[1:]) / 2.0).astype(
                    np.float32
                )
                return {
                    "type": "binnedpdf",
                    "model": be,
                    "centers": torch.from_numpy(centers).float(),
                }

            elif method == "HeteroscedasticLaplace":
                from torch.utils.data import DataLoader, TensorDataset
                from torchregress.algorithms import HeteroscedasticLaplaceRegressor

                loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(X_train_np).float(),
                        torch.from_numpy(y_train_np).float().unsqueeze(1),
                    ),
                    batch_size=config.batch_size,
                    shuffle=True,
                )
                base = _MLP(in_dim, hidden, out_dim=hidden)

                class _TupleHead(nn.Module):
                    """Two-head with the .linear attribute HLAX requires."""

                    def __init__(self, d: int) -> None:
                        super().__init__()
                        self.linear = nn.Linear(d, 2)

                    def forward(self, x: torch.Tensor) -> Any:
                        o = self.linear(x)
                        return o[:, 0:1], o[:, 1:2]

                hl = HeteroscedasticLaplaceRegressor(base, _TupleHead(hidden))
                hl.fit(loader, lr=config.lr, epochs=config.epochs)
                return {"type": "hlaplace", "model": hl}

            elif method == "VIDS":
                from torchregress.algorithms.adaptive_prior_vi import VIDSRegressor

                vids = VIDSRegressor(in_dim, target_dim=1)
                vids.fit(
                    torch.from_numpy(X_train_np).float().to(current_device()),
                    torch.from_numpy(y_train_np).float().unsqueeze(1).to(current_device()),
                    epochs=config.epochs,
                )
                return {"type": "vids", "model": vids}

            elif method == "IVON":
                import torch.optim as optim
                from torch.utils.data import DataLoader, TensorDataset
                from torchregress.algorithms.ivon import IVON

                loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(X_train_np).float(),
                        torch.from_numpy(y_train_np).float().unsqueeze(1),
                    ),
                    batch_size=config.batch_size,
                    shuffle=True,
                )
                model = _MLP(in_dim, hidden, out_dim=1)
                opt = IVON(
                    model.parameters(),
                    lr=config.lr,
                    ess=float(len(X_train_np)),
                )
                dev = current_device()
                model.to(dev)
                model.train()
                for _ in range(config.epochs):
                    for Xb, yb in loader:
                        Xb, yb = Xb.to(dev), yb.to(dev)
                        opt.zero_grad(set_to_none=True)
                        nn.MSELoss()(model(Xb), yb).backward()
                        opt.step()
                model.eval()
                return {"type": "ivon", "model": model, "opt": opt}

            elif method in ("BatchEnsemble", "MeanOnlyBatchEnsemble"):
                import torch.optim as optim
                from torch.utils.data import DataLoader, TensorDataset
                from torchregress.ensemble import (
                    BatchEnsembleMLPBackbone,
                    BatchEnsembleRegressor,
                )
                from torchregress.losses import GaussianNLLLoss

                hetero = method == "BatchEnsemble"
                n_members = 4
                dev = current_device()
                bb = BatchEnsembleMLPBackbone(in_dim, hidden, ensemble_size=n_members)
                model = BatchEnsembleRegressor(
                    bb,
                    feature_dim=bb.feature_dim,
                    output_dim=1,
                    ensemble_size=n_members,
                    heteroscedastic=hetero,
                    device=str(dev),
                )
                model.to(dev)
                loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(X_train_np).float(),
                        torch.from_numpy(y_train_np).float().unsqueeze(1),
                    ),
                    batch_size=config.batch_size,
                    shuffle=True,
                )
                opt = optim.Adam(model.parameters(), lr=config.lr)
                loss_fn = GaussianNLLLoss()
                mse = nn.MSELoss()
                model.train()
                for _ in range(config.epochs):
                    for Xb, yb in loader:
                        Xb, yb = Xb.to(dev), yb.to(dev)
                        opt.zero_grad(set_to_none=True)
                        out = model(Xb)
                        if hetero:
                            B = Xb.shape[0]
                            flat = torch.cat(
                                [
                                    out["means"].reshape(B * n_members, 1),
                                    out["log_vars"].reshape(B * n_members, 1),
                                ],
                                dim=-1,
                            )
                            loss = loss_fn(flat, yb.repeat_interleave(n_members, dim=0))
                        else:
                            loss = mse(
                                out["means"],
                                yb.reshape(-1, 1, 1).expand(-1, n_members, -1),
                            )
                        loss.backward()
                        opt.step()
                model.eval()
                return {"type": "batchensemble", "model": model}

            elif method == "Stacking":
                from torchregress.ensemble import StackingEnsemble

                X_t = torch.from_numpy(X_train_np).float().to(current_device())
                y_t = torch.from_numpy(y_train_np).float().to(current_device()).unsqueeze(1)
                bases = []
                for k in range(3):
                    set_harness_seed(config.seed + 100 * k + 11)
                    m = _MLP(in_dim, hidden, out_dim=1)
                    _train_one(m, nn.MSELoss(), X_t, y_t, config)
                    m.eval()
                    bases.append(m)
                # Linear meta-learner on frozen base predictions (train-split).
                meta = nn.Linear(3, 1).to(current_device())
                _train_one(
                    meta,
                    nn.MSELoss(),
                    torch.cat([b(X_t).detach() for b in bases], dim=-1),
                    y_t,
                    config,
                )
                meta.eval()
                return {"type": "stacking", "model": StackingEnsemble(bases, meta)}

            else:
                raise ValueError(f"Unknown method: {method}")

        # --- sklearn ensembles ---
        elif method == "sklearn_VotingRegressor":
            est = VotingRegressor(
                [
                    ("lr", LinearRegression()),
                    ("rf", RandomForestRegressor(n_estimators=50, random_state=config.seed)),
                    ("svr", SVR()),
                    ("knn", KNeighborsRegressor(n_neighbors=5)),
                ]
            )
            est.fit(X_train_np, y_train_np)
            return {"type": "sklearn", "model": est}

        elif method == "sklearn_StackingRegressor":
            est = StackingRegressor(
                estimators=[
                    ("lr", LinearRegression()),
                    ("rf", RandomForestRegressor(n_estimators=50, random_state=config.seed)),
                    ("svr", SVR()),
                    ("knn", KNeighborsRegressor(n_neighbors=5)),
                ],
                final_estimator=RidgeCV(),
                cv=3,
            )
            est.fit(X_train_np, y_train_np)
            return {"type": "sklearn", "model": est}

        elif method == "sklearn_BaggingRegressor":
            from sklearn.tree import DecisionTreeRegressor

            est = BaggingRegressor(
                estimator=DecisionTreeRegressor(random_state=config.seed),
                n_estimators=100,
                random_state=config.seed,
                n_jobs=1,
            )
            est.fit(X_train_np, y_train_np)
            return {"type": "sklearn", "model": est}

        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # predict — aggregate ensemble members
    # ------------------------------------------------------------------

    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        model_type = model_state["type"]

        # --- wrapper ensembles (DeepEnsemble_MSE / DeepEnsemble_GaussianNLL) ---
        if model_type == "wrapper":
            reg = model_state["model"]
            from harness.wrappers import HeteroscedasticEnsembleRegressor

            if isinstance(reg, HeteroscedasticEnsembleRegressor):
                diag = reg._predict_diagnostics(X_test)
                return {"mean": diag["mean"], "std": diag["std"]}
            else:
                mean = reg.predict(X_test)
                return {"y_pred": mean}

        # --- torchregress: MultiQuantile → mean of quantiles ---
        elif model_type == "multiquantile":
            los = []
            meds = []
            his = []
            for m in model_state["models"]:
                with torch.no_grad():
                    out = m(torch.from_numpy(X_test).float().to(current_device()))
                    los.append(out[:, 0].detach().cpu().numpy().reshape(-1))
                    meds.append(out[:, 1].detach().cpu().numpy().reshape(-1))
                    his.append(out[:, 2].detach().cpu().numpy().reshape(-1))
            lo = np.stack(los, axis=0).mean(axis=0)
            med = np.stack(meds, axis=0).mean(axis=0)
            hi = np.stack(his, axis=0).mean(axis=0)
            return {"y_pred": med, "lower": lo, "upper": hi}

        # --- torchregress: MDNEnsemble → mixture-mean point prediction ---
        elif model_type == "mdn_ensemble":
            ens = model_state["model"]
            loss_fn = model_state["loss_fn"]
            ens.eval()
            with torch.no_grad():
                out = ens(torch.from_numpy(X_test).float().to(current_device()))
                # Forward returns (ensemble, N, out_dim) member params.
                if isinstance(out, torch.Tensor) and out.dim() == 3:
                    means = []
                    for k in range(out.shape[0]):
                        mk, _ = loss_fn.predict_mean_std(out[k])
                        means.append(mk.reshape(-1).detach().cpu().numpy())
                    y_pred = np.stack(means, axis=0).mean(axis=0)
                else:
                    y_pred = out.reshape(-1).detach().cpu().numpy()
            return {"y_pred": y_pred}

        # --- torchregress: LaplaceMAP → posterior mean/var + intervals ---
        # Point prediction is the MAP (base net); the Laplace posterior only
        # supplies the variance — its MC mean is unstable at smoke budgets.
        elif model_type == "laplace":
            lap = model_state["model"]
            base = model_state["base"]
            Xt = torch.from_numpy(X_test).float().to(current_device())
            with torch.no_grad():
                mean_np = base(Xt).reshape(-1).detach().cpu().numpy()
                _, var = lap.predict_with_uncertainty(Xt)
            std_np = np.sqrt(var.reshape(-1).detach().cpu().numpy().clip(min=1e-12))
            z = 1.645
            return {
                "lower": mean_np - z * std_np,
                "upper": mean_np + z * std_np,
                "y_pred": mean_np,
                "mean": mean_np,
                "std": std_np,
            }

        # --- torchregress: Snapshot → snapshot mean/var intervals ---
        elif model_type == "snapshot":
            sn = model_state["model"]
            with torch.no_grad():
                mean, var = sn.predict_with_uncertainty(
                    torch.from_numpy(X_test).float().to(current_device())
                )
            mean_np = np.asarray(mean).reshape(-1)
            std_np = np.sqrt(np.asarray(var).reshape(-1).clip(min=1e-12))
            z = 1.645
            return {
                "lower": mean_np - z * std_np,
                "upper": mean_np + z * std_np,
                "y_pred": mean_np,
                "mean": mean_np,
                "std": std_np,
            }

        # --- torchregress: MultiSWAG → mean/epistemic/aleatoric split ---
        elif model_type == "multiswag":
            mswag = model_state["model"]
            mswag.eval()
            with torch.no_grad():
                mean, epi, alea = mswag.predict_with_uncertainty(
                    torch.from_numpy(X_test).float().to(current_device()), n_samples=10
                )
            mean_np = np.asarray(mean.detach().cpu()).reshape(-1)
            std_np = np.sqrt(
                (
                    np.asarray(epi.detach().cpu()).reshape(-1)
                    + np.asarray(alea.detach().cpu()).reshape(-1)
                ).clip(min=1e-12)
            )
            z = 1.645
            return {
                "lower": mean_np - z * std_np,
                "upper": mean_np + z * std_np,
                "y_pred": mean_np,
                "mean": mean_np,
                "std": std_np,
            }

        # --- torchregress: BinnedPDFEnsemble → expected value over bins ---
        elif model_type == "binnedpdf":
            be = model_state["model"]
            centers = model_state["centers"].to(current_device())
            be.eval()
            with torch.no_grad():
                dist = be.predict(torch.from_numpy(X_test).float().to(current_device()))
            # predict() → {probabilities, mean, variance} over bin support.
            probs = dist["probabilities"]
            mean_np = (probs @ centers.reshape(-1)).reshape(-1).detach().cpu().numpy()
            if "variance" in dist and dist["variance"] is not None:
                std_np = np.sqrt(np.asarray(dist["variance"]).reshape(-1).clip(min=1e-12))
            else:
                var_np = (probs * (centers.reshape(-1) - mean_np.reshape(-1, 1)) ** 2).sum(-1)
                std_np = np.sqrt(np.asarray(var_np).clip(min=1e-12))
            z = 1.645
            return {
                "lower": mean_np - z * std_np,
                "upper": mean_np + z * std_np,
                "y_pred": mean_np,
                "mean": mean_np,
                "std": std_np,
            }

        # --- torchregress: HeteroscedasticLaplace → PredictiveBatch ---
        elif model_type == "hlaplace":
            hl = model_state["model"]
            with torch.no_grad():
                pb = hl.predict_distribution(torch.from_numpy(X_test).float().to(current_device()))

            def _field(name: str) -> Any:
                if hasattr(pb, "items"):
                    try:
                        return dict(pb.items()).get(name)
                    except Exception:
                        pass
                return getattr(pb, name, None)

            mean_t = _field("mean")
            if mean_t is None:
                mean_t = _field("loc")
            std_t = _field("std")
            if std_t is None and _field("variance") is not None:
                std_t = torch.sqrt(_field("variance").clamp_min(1e-12))
            if std_t is None and _field("scale") is not None:
                std_t = _field("scale") * np.sqrt(2.0)
            mean_np = np.asarray(mean_t).reshape(-1)
            std_np = (
                np.asarray(std_t).reshape(-1) if std_t is not None else np.full_like(mean_np, 0.1)
            )
            z = 1.645
            return {
                "lower": mean_np - z * std_np,
                "upper": mean_np + z * std_np,
                "y_pred": mean_np,
                "mean": mean_np,
                "std": std_np,
            }

        # --- torchregress: VIDS → PredictiveBatch mean/std ---
        elif model_type == "vids":
            vids = model_state["model"]
            with torch.no_grad():
                pb = vids.predict_distribution(
                    torch.from_numpy(X_test).float().to(current_device())
                )
            mean_np = np.asarray(pb.mean).reshape(-1)
            std_np = (
                np.asarray(pb.std).reshape(-1)
                if getattr(pb, "std", None) is not None
                else np.full_like(mean_np, 0.1)
            )
            z = 1.645
            return {
                "lower": mean_np - z * std_np,
                "upper": mean_np + z * std_np,
                "y_pred": mean_np,
                "mean": mean_np,
                "std": std_np,
            }

        # --- torchregress: IVON → posterior MC sampling ---
        # One-shot sampled_params context per draw (S=20); mean/var over draws.
        elif model_type == "ivon":
            ivon_model = model_state["model"]
            ivon_opt = model_state["opt"]
            ivon_model.eval()
            draws = []
            with torch.no_grad():
                Xt = torch.from_numpy(X_test).float().to(current_device())
                for _ in range(20):
                    with ivon_opt.sampled_params(train=False):
                        draws.append(ivon_model(Xt).detach().cpu().numpy())
            stack = np.stack(draws, axis=0)
            mean_np = stack.mean(axis=0).reshape(-1)
            std_np = stack.std(axis=0).reshape(-1).clip(min=1e-6)
            z = 1.645
            return {
                "lower": mean_np - z * std_np,
                "upper": mean_np + z * std_np,
                "y_pred": mean_np,
                "mean": mean_np,
                "std": std_np,
            }

        # --- sklearn ensembles ---
        elif model_type == "sklearn":
            est = model_state["model"]
            y_pred = est.predict(X_test).reshape(-1)
            return {"y_pred": y_pred}

        # --- torchregress: BatchEnsemble / MeanOnly → predict() mean/variance ---
        elif model_type == "batchensemble":
            be = model_state["model"]
            be.eval()
            dev = next(be.parameters()).device
            with torch.no_grad():
                d = be.predict(torch.from_numpy(X_test).float().to(dev))
            mean_np = d["mean"].reshape(-1).detach().cpu().numpy()
            std_np = np.sqrt(d["variance"].reshape(-1).detach().cpu().numpy().clip(min=1e-12))
            z = 1.645
            return {
                "lower": mean_np - z * std_np,
                "upper": mean_np + z * std_np,
                "y_pred": mean_np,
                "mean": mean_np,
                "std": std_np,
            }

        # --- torchregress: StackingEnsemble → meta-learner point prediction ---
        elif model_type == "stacking":
            st = model_state["model"]
            st.eval()
            with torch.no_grad():
                out = st(torch.from_numpy(X_test).float().to(next(st.parameters()).device))
            y_pred = out.reshape(-1).detach().cpu().numpy()
            return {"y_pred": y_pred}

        raise ValueError(f"Unknown model type: {model_type}")
