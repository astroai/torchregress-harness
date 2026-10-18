"""Self-contained synthetic simulation-based-inference (SBI) tasks.

These generators have **analytic or numerically exact reference posteriors**, so
likelihood-ratio estimators (NRE, contrastive flows, NPE) can be validated
without the optional ``sbi``/``sbibm`` dependencies or frozen benchmark files.

Two tasks are provided:

``gaussian_linear``
    2-D parameters ``theta ~ U([-2, 2]^2)``; 4-D observations
    ``x = A theta + eps`` with ``eps ~ N(0, sigma^2 I)``.  The posterior is a
    truncated Gaussian and can be sampled exactly by rejection.

``bimodal``
    2-D parameters ``theta ~ U([-2, 2]^2)``; observations
    ``x_d = theta_d^2 + eps_d``.  Each posterior marginal is bimodal (symmetric
    roots ``+/- sqrt(x_d)``); reference draws are obtained by grid inversion of
    the exact 1-D CDF.

Both generators return harness ``splits`` with:

* ``X_train``/``y_train``: prior simulations (observations / parameters),
* ``X_test``/``y_test``: benchmark observations and their true parameters,
* ``sbc_x``/``sbc_theta``: an independent held-out set for SBC/TARP diagnostics,
* ``prior_low``/``prior_high``, ``n_targets`` and the task spec
  (``sbi_task``, ``sbi_seed``, ``sbi_noise_std``, ``sbi_matrix``) needed to
  reconstruct the reference posterior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np


@dataclass(frozen=True)
class GaussianLinearTask:
    """Linear-Gaussian simulator with a truncated-Gaussian posterior."""

    matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    noise_std: float = 0.5
    prior_low: float = -2.0
    prior_high: float = 2.0
    n_theta: int = 2

    def __post_init__(self) -> None:
        if self.matrix.size == 0:
            matrix = np.array(
                [
                    [1.0, 0.2, -0.4, 0.1],
                    [0.3, -0.7, 0.5, 0.2],
                ],
                dtype=np.float64,
            )
            object.__setattr__(self, "matrix", matrix)
        if self.matrix.shape[0] != self.n_theta:
            raise ValueError(
                f"matrix must have shape ({self.n_theta}, n_observables); got {self.matrix.shape}"
            )

    @property
    def n_observables(self) -> int:
        return int(self.matrix.shape[1])

    def simulate(self, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Noise-free forward model plus Gaussian observation noise."""
        mean = np.asarray(theta, dtype=np.float64) @ self.matrix
        return mean + self.noise_std * rng.standard_normal(mean.shape)

    def posterior_mean_cov(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Un-truncated Gaussian posterior mean and covariance for one ``x``."""
        precision = self.matrix @ self.matrix.T / self.noise_std**2
        cov = np.linalg.inv(precision)
        mean = cov @ (self.matrix @ np.asarray(x, dtype=np.float64)) / self.noise_std**2
        return mean, cov

    def reference_samples(
        self, x: np.ndarray, rng: np.random.Generator, n_samples: int
    ) -> np.ndarray:
        """Samples from the prior-truncated Gaussian posterior at ``x``."""
        mean, cov = self.posterior_mean_cov(x)
        chol = np.linalg.cholesky(cov)
        accepted: list[np.ndarray] = []
        n_found = 0
        while n_found < n_samples:
            draw = mean + rng.standard_normal((4 * n_samples, self.n_theta)) @ chol.T
            inside = np.all((draw >= self.prior_low) & (draw <= self.prior_high), axis=1)
            if inside.any():
                accepted.append(draw[inside])
                n_found += int(inside.sum())
        return np.concatenate(accepted, axis=0)[:n_samples]

    def reference_grid_weights(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Posterior weights on a candidate grid (uniform prior, up to constant)."""
        theta = np.atleast_2d(theta)
        resid = np.asarray(x, dtype=np.float64)[None, :] - theta @ self.matrix
        log_w = -0.5 * np.sum(resid**2, axis=1) / self.noise_std**2
        inside = np.all((theta >= self.prior_low) & (theta <= self.prior_high), axis=1)
        log_w = np.where(inside, log_w, -np.inf)
        w = np.exp(log_w - log_w.max())
        return w / w.sum()


@dataclass(frozen=True)
class BimodalTask:
    """Quadratic simulator with independent bimodal 1-D posterior marginals."""

    noise_std: float = 0.4
    prior_low: float = -2.0
    prior_high: float = 2.0
    n_theta: int = 2
    grid_size: int = 4001

    def simulate(self, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mean = np.asarray(theta, dtype=np.float64) ** 2
        return mean + self.noise_std * rng.standard_normal(mean.shape)

    def _grid(self) -> np.ndarray:
        return np.linspace(self.prior_low, self.prior_high, self.grid_size)

    def reference_samples(
        self, x: np.ndarray, rng: np.random.Generator, n_samples: int
    ) -> np.ndarray:
        grid = self._grid()
        samples = np.empty((n_samples, self.n_theta), dtype=np.float64)
        for dim in range(self.n_theta):
            log_w = -0.5 * (float(x[dim]) - grid**2) ** 2 / self.noise_std**2
            w = np.exp(log_w - log_w.max())
            cdf = np.cumsum(w)
            cdf /= cdf[-1]
            uniform = rng.random(n_samples)
            index = np.searchsorted(cdf, uniform, side="right").clip(0, len(grid) - 1)
            # Linear interpolation between adjacent grid points for continuity.
            left = np.clip(index - 1, 0, len(grid) - 1)
            frac = np.where(
                index > 0,
                (uniform - cdf[left]) / np.maximum(cdf[index] - cdf[left], 1e-300),
                0.0,
            )
            samples[:, dim] = grid[left] + np.clip(frac, 0.0, 1.0) * (grid[index] - grid[left])
        return samples

    def reference_grid_weights(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Posterior weights on a candidate grid (uniform prior, up to constant)."""
        theta = np.atleast_2d(theta)
        resid = np.asarray(x, dtype=np.float64)[None, :] - theta**2
        log_w = -0.5 * np.sum(resid**2, axis=1) / self.noise_std**2
        inside = np.all((theta >= self.prior_low) & (theta <= self.prior_high), axis=1)
        log_w = np.where(inside, log_w, -np.inf)
        w = np.exp(log_w - log_w.max())
        return w / w.sum()


def _make_splits(
    task: GaussianLinearTask | BimodalTask,
    task_name: str,
    *,
    n_samples: int,
    n_observations: int,
    n_sbc_simulations: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(task.prior_low, task.prior_high, size=(int(n_samples), task.n_theta))
    x = task.simulate(theta, rng)

    obs_theta = rng.uniform(
        task.prior_low, task.prior_high, size=(int(n_observations), task.n_theta)
    )
    obs_x = task.simulate(obs_theta, rng)

    sbc_theta = rng.uniform(
        task.prior_low, task.prior_high, size=(int(n_sbc_simulations), task.n_theta)
    )
    sbc_x = task.simulate(sbc_theta, rng)

    splits: dict[str, object] = {
        "X_train": np.asarray(x, dtype=np.float32),
        "y_train": np.asarray(theta, dtype=np.float32),
        "X_test": np.asarray(obs_x, dtype=np.float32),
        "y_test": np.asarray(obs_theta, dtype=np.float32),
        "sbc_x": np.asarray(sbc_x, dtype=np.float32),
        "sbc_theta": np.asarray(sbc_theta, dtype=np.float32),
        "n_targets": int(task.n_theta),
        "prior_low": float(task.prior_low),
        "prior_high": float(task.prior_high),
        "_skip_y_standardization": True,
        "_synthetic_fallback": True,
        "sbi_task": task_name,
        "sbi_seed": int(seed),
        "sbi_noise_std": float(task.noise_std),
    }
    if isinstance(task, GaussianLinearTask):
        splits["sbi_matrix"] = np.asarray(task.matrix, dtype=np.float64)
    return splits


def make_gaussian_linear_splits(
    n_samples: int = 4000,
    n_observations: int = 3,
    n_sbc_simulations: int = 200,
    seed: int = 42,
) -> dict[str, object]:
    """Splits for the linear-Gaussian SBI task."""
    return _make_splits(
        GaussianLinearTask(),
        "gaussian_linear",
        n_samples=n_samples,
        n_observations=n_observations,
        n_sbc_simulations=n_sbc_simulations,
        seed=seed,
    )


def make_bimodal_splits(
    n_samples: int = 4000,
    n_observations: int = 3,
    n_sbc_simulations: int = 200,
    seed: int = 42,
) -> dict[str, object]:
    """Splits for the bimodal (quadratic) SBI task."""
    return _make_splits(
        BimodalTask(),
        "bimodal",
        n_samples=n_samples,
        n_observations=n_observations,
        n_sbc_simulations=n_sbc_simulations,
        seed=seed,
    )


def task_from_splits(splits: dict[str, object]) -> GaussianLinearTask | BimodalTask:
    """Reconstruct the task object described by a splits dict."""
    name = str(splits.get("sbi_task", ""))
    low = float(cast(float, splits.get("prior_low", -2.0)))
    high = float(cast(float, splits.get("prior_high", 2.0)))
    noise = float(cast(float, splits.get("sbi_noise_std", 0.5)))
    if name == "gaussian_linear":
        matrix = np.asarray(splits["sbi_matrix"], dtype=np.float64)
        return GaussianLinearTask(matrix=matrix, noise_std=noise, prior_low=low, prior_high=high)
    if name == "bimodal":
        return BimodalTask(noise_std=noise, prior_low=low, prior_high=high)
    raise ValueError(f"unknown synthetic SBI task {name!r}")


def reference_samples(
    task: GaussianLinearTask | BimodalTask,
    x: np.ndarray,
    rng: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    """Draw reference-posterior samples for one observation."""
    return task.reference_samples(np.asarray(x, dtype=np.float64), rng, int(n_samples))


def reference_grid_weights(
    task: GaussianLinearTask | BimodalTask,
    theta: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Reference posterior weights on a candidate grid for one observation."""
    return task.reference_grid_weights(theta, np.asarray(x, dtype=np.float64))


__all__ = [
    "BimodalTask",
    "GaussianLinearTask",
    "make_bimodal_splits",
    "make_gaussian_linear_splits",
    "reference_grid_weights",
    "reference_samples",
    "task_from_splits",
]
