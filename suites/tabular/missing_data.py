"""Missing-data (MAR) suite — propensity-weighted regression.

Simulates Missing-At-Random targets on real tabular sets: observation
probability follows a logistic selection model in X, estimated back with
sklearn LogisticRegression. Compares complete-case MSE against IPW with
estimated vs oracle propensities, isolating the value of the weighting
(PropensityWeightedLoss) from the quality of the selection model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchregress.losses import PropensityWeightedLoss, WeightedMSELoss

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _MLP, _hidden_size, _train_one


def _mar_mask(
    X: np.ndarray, seed: int, strength: float = 1.5, base_rate: float = 0.7
) -> tuple[np.ndarray, np.ndarray]:
    """MAR observation mask + true propensities (logistic in X)."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1, X.shape[1]).astype(np.float64)
    w /= np.linalg.norm(w) + 1e-12
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    logit = strength * (Xs @ w) + np.log(base_rate / (1.0 - base_rate))
    propensity = 1.0 / (1.0 + np.exp(-logit))
    observed = (rng.random(len(X)) < propensity).astype(np.float32)
    return observed, propensity.astype(np.float32)


class MissingDataSuite(HarnessRunner):
    """IPW vs complete-case under MAR missingness."""

    @property
    def name(self) -> str:
        return "missing_data"

    @property
    def description(self) -> str:
        return "MAR targets: complete-case MSE vs estimated/oracle IPW"

    @property
    def supported_datasets(self) -> list[str]:
        return ["diabetes", "california_housing", "folktables_income"]

    def methods(self) -> list[str]:
        return ["CompleteCase", "PropensityWeighted", "OracleIPW", "sklearn_IPW"]

    def library_name(self, method: str) -> str:
        if method.startswith("sklearn_"):
            return "sklearn"
        return "torchregress"

    def method_notes(self, method: str) -> str:
        return {
            "CompleteCase": "MLP + MSE on observed targets only (selection-bias baseline)",
            "PropensityWeighted": "MLP + PropensityWeightedLoss with estimated propensities",
            "OracleIPW": "MLP + PropensityWeightedLoss with TRUE propensities (upper bound)",
            "sklearn_IPW": "sklearn LogisticRegression propensity + Ridge IPW (external)",
        }.get(method, "")

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
        from sklearn.linear_model import LogisticRegression

        X_np = np.asarray(splits["X_train"], dtype=np.float64)
        y_np = np.asarray(splits["y_train"], dtype=np.float32).reshape(-1)
        observed, true_prop = _mar_mask(X_np, config.seed)
        obs_idx = np.nonzero(observed > 0)[0]
        if len(obs_idx) < 16:
            raise ValueError("MAR mask left too few observed targets")
        Xo, yo = X_np[obs_idx], y_np[obs_idx]

        # Selection model back from the mask (the PropensityWeighted requirement).
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_np, observed)
        est_prop = clf.predict_proba(Xo)[:, 1].astype(np.float32)

        in_dim = Xo.shape[1]
        hidden = _hidden_size(in_dim)
        dev = current_device()

        if method == "CompleteCase":
            model = _MLP(in_dim, hidden, out_dim=1)
            _train_one(
                model,
                WeightedMSELoss(),
                torch.from_numpy(Xo).float().to(dev),
                torch.from_numpy(yo).float().to(dev).unsqueeze(1),
                config,
            )
            model.eval()
            return {"type": "point", "model": model}
        if method == "sklearn_IPW":
            from sklearn.linear_model import Ridge

            w = 1.0 / np.clip(est_prop.astype(np.float64), 0.01, None)
            reg = Ridge(alpha=1.0)
            reg.fit(Xo, yo, sample_weight=w)
            return {"type": "sklearn", "model": reg}
        if method in ("PropensityWeighted", "OracleIPW"):
            model = _MLP(in_dim, hidden, out_dim=1)
            prop = (
                est_prop.astype(np.float32)
                if method == "PropensityWeighted"
                else true_prop[obs_idx].astype(np.float32)
            )
            loss_fn = PropensityWeightedLoss()
            opt = torch.optim.Adam(model.parameters(), lr=config.lr)
            dev = current_device()
            model.to(dev)
            loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(Xo).float(),
                    torch.from_numpy(yo).float(),
                    torch.from_numpy(prop).float(),
                ),
                batch_size=config.batch_size,
                shuffle=True,
            )
            model.train()
            for _ in range(config.epochs):
                for Xb, yb, pb in loader:
                    opt.zero_grad(set_to_none=True)
                    Xb, yb, pb = Xb.to(dev), yb.to(dev), pb.to(dev)
                    out = model(Xb)
                    loss = loss_fn(out, yb.reshape(-1, 1), propensity=pb.reshape(-1, 1))
                    loss.backward()
                    opt.step()
            model.eval()
            return {"type": "point", "model": model}
        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # predict (point)
    # ------------------------------------------------------------------

    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        del method, config
        if model_state.get("type") == "sklearn":
            pred = np.asarray(
                model_state["model"].predict(np.asarray(X_test)), dtype=np.float64
            ).reshape(-1)
            return {"y_pred": pred}
        m = model_state["model"]
        with torch.no_grad():
            pred = (
                m(torch.from_numpy(np.asarray(X_test)).float().to(current_device()))
                .reshape(-1)
                .detach()
                .cpu()
                .numpy()
            )
        return {"y_pred": pred}
