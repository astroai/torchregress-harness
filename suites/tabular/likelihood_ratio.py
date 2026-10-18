"""Likelihood-ratio / simulation-based-inference parity suite.

Validates the likelihood-ratio stack added to torchregress against an external
classifier-based NRE reference (scikit-learn) on **self-contained synthetic
tasks with exact reference posteriors** (no ``sbi`` / ``sbibm`` dependency):

* ``synthetic_sbi_gaussian_linear``: 2-D parameters, 4-D Gaussian observations,
  truncated-Gaussian posterior.
* ``synthetic_sbi_bimodal``: 2-D parameters, quadratic observations, bimodal
  posterior marginals.

Methods
-------
- ``torchregress_NRE``: classifier NRE (balanced binary, ``NeuralRatioLoss``),
  mismatched joint-vs-product pairs; posterior on a prior grid via
  ``inference.posterior_log_weights``.
- ``torchregress_NRE_multiclass``: same with the multiclass (NRE-B/InfoNCE)
  criterion.
- ``torchregress_ContrastiveFlow``: conditional flow trained with
  ``ContrastiveFlowLoss`` in its likelihood-ratio regime (``temperature=1``,
  ``margin=0``); posterior on the same grid from flow log-densities.
- ``torchregress_NPE``: amortized posterior flow ``p(theta | x)`` trained with
  ``NormalizingFlowLoss``; posterior samples drawn directly.
- ``sklearn_NRE``: external reference — logistic regression on
  ``concat([theta, x])`` trained on joint vs product pairs; posterior on the
  grid from its log-odds.

Metrics
-------
Point/posterior accuracy against the *true* parameters (``PostMeanRMSE``,
``MLE_RMSE`` where a grid argmax exists), full-posterior accuracy against the
analytic reference draws (``W1_mean``, ``RefMeanRMSE``), marginal interval
coverage (``Coverage68`` / ``Coverage95``), and posterior-calibration
diagnostics computed with torchregress' own SBC/TARP metrics on an independent
held-out set of prior simulations (``SBC_min_p``, ``TARP_err``),
plus candidate-bank effective sample size (``ESS_mean``).

Notes
-----
- The grid posterior is a deliberate finite-bank approximation; ``ESS_mean``
  and ``W1_mean`` expose how well the grid/proposal covers the posterior.
- On the bimodal task the posterior *mean* is near zero by symmetry, so
  ``PostMeanRMSE`` is not an accuracy score there; prefer ``W1_mean``,
  ``MLE_RMSE``, coverage and SBC/TARP.
- sklearn's NRE is a linear-in-features reference; it is expected to be weaker
  on the quadratic task. It is included as an external parity/architecture
  baseline, not as a tuned competitor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torchregress.inference import (
    effective_sample_size,
    grid_log_posterior,
    maximum_likelihood_estimate,
    posterior_log_weights,
)
from torchregress.losses import (
    NeuralRatioLoss,
    build_ratio_classifier,
    neural_ratio_logits,
    neural_ratio_loss,
)
from torchregress.metrics import (
    sbc_rank_histogram,
    sbc_uniformity_pvalue,
    simulation_based_calibration_ranks,
    tarp_coverage,
)

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from harness.synthetic_sbi import reference_samples, task_from_splits
from suites._backbones import _MLP

N_NEGATIVES = 8
GRID_SIZE = 41
N_POSTERIOR_SAMPLES = 256
N_SBC_CASES = 200

_METHODS = [
    "torchregress_NRE",
    "torchregress_NRE_multiclass",
    "torchregress_ContrastiveFlow",
    "torchregress_NPE",
    "sklearn_NRE",
]


def _smoke_scaling(config: HarnessConfig) -> dict[str, int]:
    """Tiny budget under ``config.smoke`` so CI stays fast."""
    if config.smoke:
        return {
            "n_train": 320,
            "epochs": min(int(config.epochs), 4),
            "n_negatives": 2,
            "grid_size": 13,
            "n_samples": 64,
            "n_sbc": 24,
        }
    return {
        "n_train": 0,  # 0 == use the full training split
        "epochs": int(config.epochs),
        "n_negatives": N_NEGATIVES,
        "grid_size": GRID_SIZE,
        "n_samples": N_POSTERIOR_SAMPLES,
        "n_sbc": N_SBC_CASES,
    }


def _prior_negatives(
    rng: torch.Generator,
    batch_size: int,
    n_negatives: int,
    n_theta: int,
    low: float,
    high: float,
    device: torch.device,
) -> torch.Tensor:
    """Uniform prior draws for negative contexts, shape ``[B, K, D]``."""
    shape = (batch_size, n_negatives, n_theta)
    draws = torch.rand(shape, generator=rng)
    return (low + (high - low) * draws).to(device)


def _make_grid(low: float, high: float, n_theta: int, size: int) -> np.ndarray:
    axes = [np.linspace(low, high, size, dtype=np.float64)] * n_theta
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.reshape(-1) for m in mesh], axis=1)


def _grid_log_weights(scores: torch.Tensor, *, likelihood_scan: bool) -> torch.Tensor:
    """Normalize grid scores: likelihood scan vs log-ratio (both map to weights)."""
    weights = grid_log_posterior if likelihood_scan else posterior_log_weights
    return weights(scores.reshape(1, -1)).reshape(-1)


def _samples_from_grid(
    grid: np.ndarray,
    log_weights: np.ndarray,
    n_samples: int,
    rng: torch.Generator,
) -> np.ndarray:
    weights = torch.as_tensor(np.exp(log_weights - log_weights.max()), dtype=torch.float64)
    weights = weights / weights.sum()
    index = torch.multinomial(weights, n_samples, replacement=True, generator=rng)
    return grid[np.asarray(index.detach().cpu())]


class LikelihoodRatioSuite(HarnessRunner):
    """Likelihood-ratio / SBI methods on exact-reference synthetic tasks."""

    @property
    def name(self) -> str:
        return "likelihood_ratio"

    @property
    def description(self) -> str:
        return (
            "NRE / contrastive-flow / NPE posterior engines vs sklearn NRE on "
            "exact-reference synthetic SBI tasks (W1, coverage, SBC, TARP)"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return ["synthetic_sbi_gaussian_linear", "synthetic_sbi_bimodal"]

    def methods(self) -> list[str]:
        return list(_METHODS)

    def library_name(self, method: str) -> str:
        return "sklearn" if method.startswith("sklearn") else "torchregress"

    def library_version(self, method: str) -> str:
        if method.startswith("sklearn"):
            import sklearn

            return str(sklearn.__version__)
        import torchregress

        return str(torchregress.__version__)

    def method_notes(self, method: str) -> str:
        notes = {
            "torchregress_NRE": (
                "balanced binary NRE (mismatched pairs), grid posterior via posterior_log_weights"
            ),
            "torchregress_NRE_multiclass": ("multiclass NRE-B/InfoNCE criterion, grid posterior"),
            "torchregress_ContrastiveFlow": (
                "contrastive flow InfoNCE, temperature=1/margin=0, grid posterior"
            ),
            "torchregress_NPE": "amortized posterior flow p(theta|x), direct sampling",
            "sklearn_NRE": "linear logistic NRE on concat([theta,x]); external reference",
        }
        return notes.get(method, "")

    # ------------------------------------------------------------------
    # Data / model helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=device)

    def _fit_nre(
        self,
        method: str,
        X: np.ndarray,
        theta: np.ndarray,
        config: HarnessConfig,
        budget: dict[str, int],
        *,
        prior_low: float,
        prior_high: float,
    ) -> dict[str, Any]:
        device = current_device()
        X_t = self._tensor(X, device)
        theta_t = self._tensor(theta, device)
        n, n_theta = theta_t.shape
        hidden = 32 if config.smoke else 64
        classifier = build_ratio_classifier(
            context_dim=n_theta,
            observable_dim=X_t.shape[1],
            hidden_dims=(hidden, hidden),
        ).to(device)
        criterion = "multiclass" if method.endswith("multiclass") else "bce"
        loss_fn = NeuralRatioLoss(classifier, criterion=criterion, reduction="mean")
        optimizer = torch.optim.Adam(classifier.parameters(), lr=config.lr)
        rng = torch.Generator().manual_seed(int(config.seed) + 3)
        n_epochs = int(budget["epochs"])
        batch_size = min(int(config.batch_size), n)
        n_neg = int(budget["n_negatives"])

        for _ in range(n_epochs):
            order = torch.randperm(n, generator=rng)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size].to(device)
                x_batch = X_t[idx]
                theta_batch = theta_t[idx]
                neg_theta = _prior_negatives(
                    rng,
                    len(idx),
                    n_neg,
                    n_theta,
                    prior_low,
                    prior_high,
                    device,
                )
                perm = torch.randperm(n, generator=rng)[: len(idx)].to(device)
                neg_x = X_t[perm].unsqueeze(1).expand(-1, n_neg, -1)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(
                    theta_batch,
                    x_batch,
                    negative_context=neg_theta,
                    negative_target=neg_x,
                )
                if start == 0:
                    # Functional/class parity guard (same computation, same value).
                    with torch.no_grad():
                        check = neural_ratio_loss(
                            classifier,
                            theta_batch,
                            x_batch,
                            negative_context=neg_theta,
                            negative_target=neg_x,
                            criterion=criterion,
                        )
                    torch.testing.assert_close(check, loss.detach(), atol=1e-5, rtol=1e-5)
                loss.backward()
                optimizer.step()

        classifier.eval()
        return {"type": method, "classifier": classifier, "loss_fn": loss_fn}

    def _fit_contrastive_flow(
        self,
        X: np.ndarray,
        theta: np.ndarray,
        config: HarnessConfig,
        budget: dict[str, int],
        *,
        prior_low: float,
        prior_high: float,
    ) -> dict[str, Any]:
        from torchregress.losses import (
            ContrastiveFlowLoss,
            create_flow_model,
        )

        device = current_device()
        X_t = self._tensor(X, device)
        theta_t = self._tensor(theta, device)
        n, n_theta = theta_t.shape
        flow = create_flow_model(
            n_features=X_t.shape[1],
            context_dim=n_theta,
            flow_type="nsf",
            n_transforms=2 if config.smoke else 3,
            hidden_features=(32, 32) if config.smoke else (64, 64),
        ).to(device)
        loss_fn = ContrastiveFlowLoss(
            flow=flow,
            temperature=1.0,
            margin=0.0,
            chunk_size=128,
        ).to(device)
        optimizer = torch.optim.Adam(flow.parameters(), lr=config.lr)
        rng = torch.Generator().manual_seed(int(config.seed) + 5)
        batch_size = min(int(config.batch_size), n)
        n_neg = int(budget["n_negatives"])
        for _ in range(int(budget["epochs"])):
            order = torch.randperm(n, generator=rng)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size].to(device)
                neg_theta = _prior_negatives(
                    rng,
                    len(idx),
                    n_neg,
                    n_theta,
                    prior_low,
                    prior_high,
                    device,
                )
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(theta_t[idx], X_t[idx], negative_context=neg_theta)
                loss.backward()
                optimizer.step()
        loss_fn.eval()
        return {"type": "contrastive_flow", "loss_fn": loss_fn}

    def _fit_npe(
        self,
        X: np.ndarray,
        theta: np.ndarray,
        config: HarnessConfig,
        budget: dict[str, int],
    ) -> dict[str, Any]:
        from torchregress.losses import NormalizingFlowLoss, create_flow_model

        device = current_device()
        X_t = self._tensor(X, device)
        theta_t = self._tensor(theta, device)
        n, n_theta = theta_t.shape
        hidden = 32 if config.smoke else 64
        encoder = _MLP(in_dim=X_t.shape[1], hidden=hidden, out_dim=hidden).to(device)
        flow = create_flow_model(
            n_features=n_theta,
            context_dim=hidden,
            flow_type="nsf",
            n_transforms=2 if config.smoke else 3,
            hidden_features=(hidden, hidden),
        ).to(device)
        loss_fn = NormalizingFlowLoss(flow=flow, reduction="mean").to(device)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(loss_fn.parameters()), lr=config.lr
        )
        rng = torch.Generator().manual_seed(int(config.seed) + 7)
        batch_size = min(int(config.batch_size), n)
        for _ in range(int(budget["epochs"])):
            order = torch.randperm(n, generator=rng)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(encoder(X_t[idx]), theta_t[idx])
                loss.backward()
                optimizer.step()
        encoder.eval()
        loss_fn.eval()
        return {"type": "npe", "encoder": encoder, "loss_fn": loss_fn}

    def _fit_sklearn_nre(
        self,
        X: np.ndarray,
        theta: np.ndarray,
        config: HarnessConfig,
        budget: dict[str, int],
        *,
        prior_low: float,
        prior_high: float,
    ) -> dict[str, Any]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(int(config.seed) + 11)
        n, n_theta = theta.shape
        low, high = prior_low, prior_high
        n_neg = int(budget["n_negatives"])
        neg_theta = rng.uniform(low, high, size=(n, n_neg, n_theta))
        repeated_x = np.repeat(X[:, None, :], n_neg, axis=1)
        positives = np.concatenate([theta, X], axis=1)
        negatives = np.concatenate(
            [neg_theta.reshape(-1, n_theta), repeated_x.reshape(-1, X.shape[1])],
            axis=1,
        )
        features = np.concatenate([positives, negatives], axis=0)
        labels = np.concatenate([np.ones(n), np.zeros(n * n_neg)])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        classifier = LogisticRegression(max_iter=1000, C=1.0)
        classifier.fit(scaled, labels)
        return {"type": "sklearn_nre", "classifier": classifier, "scaler": scaler}

    def fit(
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)
        budget = _smoke_scaling(config)
        X = np.asarray(splits["X_train"], dtype=np.float32)
        theta = np.asarray(splits["y_train"], dtype=np.float32).reshape(len(X), -1)
        n = min(int(budget["n_train"]) or len(X), len(X))
        X, theta = X[:n], theta[:n]
        prior_low = float(splits.get("prior_low", theta.min()))
        prior_high = float(splits.get("prior_high", theta.max()))
        state: dict[str, Any]
        if method in {"torchregress_NRE", "torchregress_NRE_multiclass"}:
            state = self._fit_nre(
                method, X, theta, config, budget, prior_low=prior_low, prior_high=prior_high
            )
        elif method == "torchregress_ContrastiveFlow":
            state = self._fit_contrastive_flow(
                X, theta, config, budget, prior_low=prior_low, prior_high=prior_high
            )
        elif method == "torchregress_NPE":
            state = self._fit_npe(X, theta, config, budget)
        elif method == "sklearn_NRE":
            state = self._fit_sklearn_nre(
                X, theta, config, budget, prior_low=prior_low, prior_high=prior_high
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        state["prior_low"] = prior_low
        state["prior_high"] = prior_high
        state["n_theta"] = int(theta.shape[1])
        return state

    # ------------------------------------------------------------------
    # Posterior inference on the prior grid / direct flow samples
    # ------------------------------------------------------------------

    def _grid_for_splits(self, splits: dict[str, Any], budget: dict[str, int]) -> np.ndarray:
        return _make_grid(
            float(splits["prior_low"]),
            float(splits["prior_high"]),
            int(splits["n_targets"]),
            int(budget["grid_size"]),
        )

    def _grid_log_weights_for_x(
        self, state: dict[str, Any], grid: np.ndarray, x_obs: np.ndarray
    ) -> np.ndarray:
        device = current_device()
        grid_t = torch.as_tensor(grid, dtype=torch.float32, device=device)
        x_t = torch.as_tensor(x_obs, dtype=torch.float32, device=device).reshape(1, -1)
        kind = state["type"]
        with torch.no_grad():
            if kind in {"torchregress_NRE", "torchregress_NRE_multiclass"}:
                repeated_x = x_t.expand(grid_t.shape[0], -1)
                scores = neural_ratio_logits(state["classifier"], grid_t, repeated_x)
            elif kind == "contrastive_flow":
                repeated_x = x_t.expand(grid_t.shape[0], -1)
                scores = state["loss_fn"].log_prob(grid_t, repeated_x)
            elif kind == "sklearn_nre":
                features = np.concatenate(
                    [grid, np.repeat(x_obs.reshape(1, -1), len(grid), axis=0)], axis=1
                )
                scores_np = state["classifier"].decision_function(
                    state["scaler"].transform(features)
                )
                scores = torch.as_tensor(scores_np, dtype=torch.float32)
            else:
                raise ValueError(f"method {kind!r} has no grid posterior")
        return np.asarray(
            _grid_log_weights(scores, likelihood_scan=kind == "contrastive_flow").detach().cpu(),
            dtype=np.float64,
        )

    def _posterior_samples(
        self,
        state: dict[str, Any],
        x_obs: np.ndarray,
        grid: np.ndarray,
        n_samples: int,
        rng: torch.Generator,
    ) -> np.ndarray:
        kind = state["type"]
        if kind == "npe":
            device = current_device()
            encoder = state["encoder"]
            loss_fn = state["loss_fn"]
            with torch.no_grad():
                context = encoder(
                    torch.as_tensor(x_obs, dtype=torch.float32, device=device).reshape(1, -1)
                )
                draws = loss_fn.sample(context, n_samples=n_samples)
            return np.asarray(draws.detach().cpu(), dtype=np.float64).reshape(n_samples, -1)
        log_w = self._grid_log_weights_for_x(state, grid, x_obs)
        return _samples_from_grid(grid, log_w, n_samples, rng)

    def _grid_mle(
        self, state: dict[str, Any], grid: np.ndarray, x_obs: np.ndarray
    ) -> np.ndarray | None:
        if state["type"] == "npe":
            return None
        log_w = self._grid_log_weights_for_x(state, grid, x_obs)
        estimate, _ = maximum_likelihood_estimate(
            torch.as_tensor(log_w[None, :]), torch.as_tensor(grid)
        )
        return np.asarray(estimate[0].detach().cpu(), dtype=np.float64)

    def _posterior_ess(self, state: dict[str, Any], grid: np.ndarray, x_obs: np.ndarray) -> float:
        if state["type"] == "npe":
            return float("nan")
        log_w = self._grid_log_weights_for_x(state, grid, x_obs)
        return float(effective_sample_size(torch.as_tensor(log_w)).item())

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _marginal_coverage(samples: np.ndarray, truth: np.ndarray, level: float) -> float:
        lo = np.quantile(samples, (1.0 - level) / 2.0, axis=0)
        hi = np.quantile(samples, 1.0 - (1.0 - level) / 2.0, axis=0)
        return float(np.mean((lo <= truth) & (truth <= hi)))

    @staticmethod
    def _wasserstein_to_reference(samples: np.ndarray, reference: np.ndarray) -> float:
        from scipy.stats import wasserstein_distance

        n_dims = samples.shape[1]
        return float(
            np.mean(
                [wasserstein_distance(samples[:, dim], reference[:, dim]) for dim in range(n_dims)]
            )
        )

    def _calibration_metrics(
        self,
        state: dict[str, Any],
        splits: dict[str, Any],
        budget: dict[str, int],
        config: HarnessConfig,
    ) -> dict[str, float]:

        sbc_x = np.asarray(splits["sbc_x"])
        sbc_theta = np.asarray(splits["sbc_theta"], dtype=np.float64)
        n_cases = min(int(budget["n_sbc"]), len(sbc_x))
        if n_cases < 4:
            return {
                "SBC_min_p": float("nan"),
                "SBC_edge_mass": float("nan"),
                "TARP_err": float("nan"),
            }
        grid = self._grid_for_splits(splits, budget)
        rng = torch.Generator().manual_seed(int(config.seed) + 17)
        draws = np.stack(
            [
                self._posterior_samples(state, sbc_x[i], grid, int(budget["n_samples"]), rng)
                for i in range(n_cases)
            ],
            axis=1,
        )  # [S, N, D]
        # torchregress SBC/TARP APIs take [N, S, D].
        samples = np.transpose(draws, (1, 0, 2))
        ranks = simulation_based_calibration_ranks(samples, sbc_theta[:n_cases])
        n_bins = max(2, min(10, n_cases // 10))
        try:
            uniformity = sbc_uniformity_pvalue(ranks, n_bins=n_bins)
            min_p = float(uniformity["min_p_value"])
        except ValueError:
            min_p = float("nan")
        histogram = sbc_rank_histogram(ranks, n_posterior_samples=int(budget["n_samples"]))
        counts = np.asarray(histogram["counts"].detach().cpu(), dtype=np.float64)
        n_ranks = int(ranks.numel()) if hasattr(ranks, "numel") else int(np.asarray(ranks).size)
        edge_mass = float((counts[..., 0].sum() + counts[..., -1].sum()) / max(n_ranks, 1))
        prior_low = float(splits["prior_low"])
        prior_high = float(splits["prior_high"])
        reference = np.random.default_rng(int(config.seed) + 19).uniform(
            prior_low, prior_high, size=(n_cases, int(splits["n_targets"]))
        )
        tarp = tarp_coverage(samples, sbc_theta[:n_cases], reference)
        return {
            "SBC_min_p": min_p,
            "SBC_edge_mass": edge_mass,
            "TARP_err": float(tarp["tarp_error"]),
        }

    def _diagnostics(
        self,
        state: dict[str, Any],
        splits: dict[str, Any],
        budget: dict[str, int],
        config: HarnessConfig,
    ) -> dict[str, float]:
        task = task_from_splits(splits)
        x_test = np.asarray(splits["X_test"])
        theta_true = np.asarray(splits["y_test"], dtype=np.float64).reshape(len(x_test), -1)
        n_obs = min(len(x_test), 2 if config.smoke else len(x_test))
        grid = self._grid_for_splits(splits, budget)
        rng = torch.Generator().manual_seed(int(config.seed) + 23)
        ref_rng = np.random.default_rng(int(config.seed) + 29)

        post_means = []
        mles = []
        ess_values = []
        w1_values = []
        coverage68 = []
        coverage95 = []
        ref_mean_errors = []
        for obs in range(n_obs):
            x_obs = x_test[obs]
            samples = self._posterior_samples(state, x_obs, grid, int(budget["n_samples"]), rng)
            reference = reference_samples(task, x_obs, ref_rng, int(budget["n_samples"]))
            post_means.append(samples.mean(axis=0))
            w1_values.append(self._wasserstein_to_reference(samples, reference))
            ref_mean_errors.append(
                float(np.sqrt(np.mean((samples.mean(axis=0) - reference.mean(axis=0)) ** 2)))
            )
            coverage68.append(self._marginal_coverage(samples, theta_true[obs], 0.68))
            coverage95.append(self._marginal_coverage(samples, theta_true[obs], 0.95))
            ess_values.append(self._posterior_ess(state, grid, x_obs))
            mle = self._grid_mle(state, grid, x_obs)
            if mle is not None:
                mles.append(mle)

        post_mean_array = np.stack(post_means)
        truth = theta_true[:n_obs]
        metrics: dict[str, float] = {
            "PostMeanRMSE": float(np.sqrt(np.mean((post_mean_array - truth) ** 2))),
            "PostMeanMAE": float(np.mean(np.abs(post_mean_array - truth))),
            "RefMeanRMSE": float(np.mean(ref_mean_errors)),
            "W1_mean": float(np.mean(w1_values)),
            "Coverage68": float(np.mean(coverage68)),
            "Coverage95": float(np.mean(coverage95)),
            "ESS_mean": (
                float(np.nanmean(ess_values))
                if np.any(np.isfinite(np.asarray(ess_values, dtype=float)))
                else float("nan")
            ),
            "MLE_RMSE": (
                float(np.sqrt(np.mean((np.stack(mles) - truth) ** 2))) if mles else float("nan")
            ),
        }
        metrics.update(self._calibration_metrics(state, splits, budget, config))
        return metrics

    # ------------------------------------------------------------------
    # Harness interface (used by generic tooling/tests; run() is custom)
    # ------------------------------------------------------------------

    def predict(
        self,
        method: str,
        model: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        budget = _smoke_scaling(config)
        grid = _make_grid(
            float(model["prior_low"]),
            float(model["prior_high"]),
            int(model["n_theta"]),
            int(budget["grid_size"]),
        )
        rng = torch.Generator().manual_seed(int(config.seed) + 31)
        samples = np.stack(
            [
                self._posterior_samples(model, x, grid, int(budget["n_samples"]), rng)
                for x in np.asarray(X_test)
            ],
            axis=0,
        )
        return {"samples": samples, "mean": samples.mean(axis=1), "y_pred": samples.mean(axis=1)}

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        del config
        mean = np.asarray(predictions["mean"], dtype=np.float64)
        truth = np.asarray(y_test, dtype=np.float64).reshape(mean.shape)
        samples = np.asarray(predictions["samples"], dtype=np.float64)
        return {
            "PostMeanRMSE": float(np.sqrt(np.mean((mean - truth) ** 2))),
            "PostMeanMAE": float(np.mean(np.abs(mean - truth))),
            "Coverage95": float(
                np.mean(
                    (np.quantile(samples, 0.025, axis=1) <= truth)
                    & (truth <= np.quantile(samples, 0.975, axis=1))
                )
            ),
        }

    # ------------------------------------------------------------------
    # Custom run: fit + exact-reference diagnostics
    # ------------------------------------------------------------------

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: dict[str, Any] | None = None,
        *,
        scenario_name: str | None = None,
    ) -> list[dict[str, Any]]:
        from time import perf_counter

        cfg = config or HarnessConfig()
        dataset_name, splits = self._prepare_scenario_run(
            scenario_name, dataset_name, splits, standardize_targets=False
        )
        if splits is None:
            from harness.datasets import load_dataset

            splits = load_dataset(dataset_name, standardize_targets=False)

        budget = _smoke_scaling(cfg)
        rows: list[dict[str, Any]] = []
        for method in self.methods():
            set_harness_seed(cfg.seed)
            base: dict[str, Any] = {
                "Method": f"{self.name}/{method}",
                "Library": self.library_name(method),
                "LibraryVersion": self.library_version(method),
                "Dataset": dataset_name,
                "Suite": self.name,
                "SourceURL": splits.get("_source_url", ""),
                "SyntheticFallback": True,
                "Device": str(current_device()),
                "Notes": self.method_notes(method),
            }
            try:
                t0 = perf_counter()
                state = self.fit(method, splits, cfg)
                train_s = perf_counter() - t0
                t0 = perf_counter()
                metrics = self._diagnostics(state, splits, budget, cfg)
                eval_s = perf_counter() - t0
                rows.append({**base, **metrics, "train_s": train_s, "eval_s": eval_s})
            except ImportError as exc:
                rows.append(
                    {
                        **base,
                        "PostMeanRMSE": None,
                        "MLE_RMSE": None,
                        "train_s": None,
                        "eval_s": None,
                        "Notes": f"skipped: {type(exc).__name__}: {exc}",
                    }
                )
            except Exception as exc:  # keep one bad method from aborting the suite
                rows.append(
                    {
                        **base,
                        "PostMeanRMSE": None,
                        "MLE_RMSE": None,
                        "train_s": None,
                        "eval_s": None,
                        "Notes": f"failed: {type(exc).__name__}: {exc}",
                    }
                )
        return rows


__all__ = ["LikelihoodRatioSuite"]
