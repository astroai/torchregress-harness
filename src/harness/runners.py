"""Abstract harness runner and concrete implementations.

Every suite inherits from ``HarnessRunner`` to provide a consistent
fit/predict/evaluate/run protocol that emits standardized result rows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from harness.devices import current_device

from .core import set_harness_seed
from .datasets import SplitsDict
from .metrics import compute_all_metrics


def _inverse_transform_y(y: np.ndarray, y_scaler: Any) -> np.ndarray:
    """Inverse-transform standardised targets back to original scale."""
    return y_scaler.inverse_transform(y.reshape(-1, 1)).ravel().astype(np.float64)


def _inverse_transform_predictions(
    predictions: dict[str, np.ndarray],
    y_scaler: Any,
) -> dict[str, np.ndarray]:
    """Inverse-transform predictions back to original target scale.

    Point estimates (``y_pred``, ``mean``, ``lower``, ``upper``) are fully
    inverse-transformed.  Standard deviations (``std``) are scaled by the
    y-scaler's scale_ only (they don't have a location shift).
    """
    out = dict(predictions)
    scale = float(y_scaler.scale_.item())
    for key in ("y_pred", "mean", "lower", "upper"):
        if key in out:
            out[key] = _inverse_transform_y(out.pop(key), y_scaler)
    if "std" in out and scale != 0:
        out["std"] = (out.pop("std") * scale).astype(np.float64)
    return out


@dataclass
class HarnessConfig:
    """Global harness configuration — shared across suites."""

    seed: int = 42
    epochs: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    device: str = "cpu"
    smoke: bool = False  # if True, use tiny data/fewer epochs for CI
    standardize_targets: bool = True  # if True, StandardScaler on y_train/y_test
    output_dir: str = "results/latest"
    warmup_epochs: int = 20  # MSE warmup epochs before switching to MC (MSE_WarmupMC)


class HarnessRunner(ABC):
    """Abstract base class for all harness suites.

    Subclasses implement:

    * ``methods()`` — return list of method identifiers this suite tests.
    * ``fit()`` — train a model given a method name and dataset split.
    * ``predict()`` — produce predictions dict from a fitted model.
    * ``evaluate()`` — compute metrics from predictions (default: use harness metrics).
    * ``run()`` — orchestrates the full fit/predict/evaluate loop, returning rows.

    The ``run()`` method is already implemented; override only if you need
    custom orchestration (e.g., calibration splits, conformal wrappers).
    """

    # ------------------------------------------------------------------
    # Suite metadata
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short suite name, e.g. ``"conformal"``, ``"bayesian_linear"``."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description of what this suite compares."""

    @property
    @abstractmethod
    def supported_datasets(self) -> list[str]:
        """List of dataset keys this suite supports."""

    @property
    def supported_methods(self) -> list[str]:
        """List of method keys this suite implements."""
        return self.methods()

    @property
    def canonical_scenarios(self) -> list[str]:
        """List of canonical scenario names this suite supports.

        When ``run()`` receives a ``scenario_name`` that matches an entry
        here, the harness resolves the canonical dataset and merges
        scenario-specific parameters (alpha, calibration_fraction, split
        config) from ``CANONICAL_SCENARIOS`` in ``datasets.py``.

        Override in suites that track upstream library examples.  The
        default is an empty list (no canonical scenarios).
        """
        return []

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def methods(self) -> list[str]:
        """Return list of method identifiers."""

    @abstractmethod
    def fit(
        self,
        method: str,
        splits: SplitsDict,
        config: HarnessConfig,
    ) -> Any:
        """Fit a model for the given method.

        Args:
            method: Method identifier from ``self.methods()``.
            splits: Dataset splits with ``X_train``, ``y_train`` keys.
            config: Harness configuration.

        Returns:
            A fitted model object (opaque — passed to ``predict()``).
        """

    @abstractmethod
    def predict(
        self,
        method: str,
        model: Any,
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        """Produce predictions for the given method and fitted model.

        Args:
            method: Method identifier.
            model: Fitted model from ``fit()``.
            X_test: Test features.
            config: Harness configuration.

        Returns:
            Predictions dict with keys like ``y_pred``, ``mean``, ``std``,
            ``lower``, ``upper``, or ``samples``.
        """

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        """Compute metrics from predictions. Default: use ``compute_all_metrics``."""
        # Inverse-transform standardised targets/predictions back to
        # original units so metrics are in the native data scale.
        y_scaler = predictions.get("_y_scaler")
        if y_scaler is not None:
            predictions = _inverse_transform_predictions(predictions, y_scaler)
            y_test = _inverse_transform_y(y_test, y_scaler)
        return compute_all_metrics(predictions, y_test)

    def library_name(self, method: str) -> str:
        """Return the library name for a method. Default: ``"torchregress"``."""
        return "torchregress"

    def library_version(self, method: str) -> str:
        """Return the version of the method's library, for artifact provenance.

        Defaults to ``""``.  Suites that call external software (MAPIE,
        torch-uncertainty, …) override this to report the installed version
        (and git commit when pip-installed from git) so result rows are
        self-describing — see ``EXTERNAL_PINS.md``.
        """
        return ""

    def method_notes(self, method: str) -> str:
        """Return notes for a method. Default: empty."""
        return ""

    # ------------------------------------------------------------------
    # Scenario resolution
    # ------------------------------------------------------------------

    def _prepare_scenario_run(
        self,
        scenario_name: str | None,
        dataset_name: str,
        splits: SplitsDict | None,
        *,
        standardize_targets: bool = True,
    ) -> tuple[str, SplitsDict | None]:
        """Validate, resolve, and merge a canonical scenario.

        Called at the top of ``run()``.  If ``scenario_name`` is not None,
        validates it against ``self.canonical_scenarios``, resolves the
        canonical dataset via ``_resolve_scenario()``, and merges scenario
        metadata (``_scenario``, ``_source_url``, ``_alpha``,
        ``_calibration_fraction``) into the splits dict.

        Returns:
            ``(effective_dataset_name, splits_or_none)`` — callers should
            replace their local ``dataset_name`` and ``splits`` variables
            with these.
        """
        if scenario_name is None:
            return dataset_name, splits

        if scenario_name not in self.canonical_scenarios:
            raise ValueError(
                f"Suite '{self.name}' does not support canonical scenario "
                f"{scenario_name!r}. Supported: {self.canonical_scenarios}"
            )

        scenario_dataset_key, scenario_splits = self._resolve_scenario(
            scenario_name,
            standardize_targets=standardize_targets,
        )

        effective_dataset = dataset_name if dataset_name else scenario_dataset_key

        effective_splits: SplitsDict
        if splits is None:
            effective_splits = scenario_splits
        else:
            # User provided their own splits — still merge scenario metadata
            effective_splits = {
                **splits,
                "_scenario": scenario_splits["_scenario"],
                "_source_url": scenario_splits["_source_url"],
            }
            if "_alpha" in scenario_splits:
                effective_splits["_alpha"] = scenario_splits["_alpha"]
            if "_calibration_fraction" in scenario_splits:
                effective_splits["_calibration_fraction"] = scenario_splits["_calibration_fraction"]

        return effective_dataset, effective_splits

    @staticmethod
    def _resolve_scenario(
        scenario_name: str,
        *,
        standardize_targets: bool = True,
    ) -> tuple[str, SplitsDict]:
        """Resolve a canonical scenario into (dataset_key, loaded_splits_with_metadata).

        Looks up ``scenario_name`` in ``CANONICAL_SCENARIOS``, loads the
        indicated dataset with scenario-specific ``loader_kwargs``, and
        attaches scenario metadata (``_scenario``, ``_source_url``,
        ``_alpha``, ``_calibration_fraction``) to the splits dict so that
        sub-suite ``run()`` overrides can consume them.

        Returns:
            ``(dataset_key, splits)`` — ``dataset_key`` is the scenario's
            canonical dataset name (used for result rows); ``splits`` is
            the loaded dataset dict with scenario metadata keys attached.

        Raises:
            KeyError: if ``scenario_name`` is not in ``CANONICAL_SCENARIOS``.
        """
        from .datasets import CANONICAL_SCENARIOS, load_dataset

        if scenario_name not in CANONICAL_SCENARIOS:
            raise KeyError(
                f"Unknown canonical scenario: {scenario_name!r}. "
                f"Available: {sorted(CANONICAL_SCENARIOS.keys())}"
            )

        scenario = CANONICAL_SCENARIOS[scenario_name]

        # Load dataset with scenario-specific kwargs
        loader_kwargs: dict[str, Any] = (
            dict(scenario.loader_kwargs) if scenario.loader_kwargs else {}
        )
        splits = load_dataset(
            scenario.dataset_key,
            standardize_targets=standardize_targets,
            **loader_kwargs,
        )

        # Attach scenario metadata so suite run() overrides can consume them
        splits["_scenario"] = scenario.scenario_name
        splits["_source_url"] = scenario.source_url
        if scenario.alpha is not None:
            splits["_alpha"] = scenario.alpha
        if scenario.calibration_fraction is not None:
            splits["_calibration_fraction"] = scenario.calibration_fraction

        return scenario.dataset_key, splits

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: SplitsDict | None = None,
        *,
        scenario_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run the suite on a dataset, returning result rows.

        Args:
            dataset_name: Dataset key (used for metadata in rows only).
                If empty **and** ``scenario_name`` is provided, this is
                resolved from the scenario's ``dataset_key``.
            config: Harness configuration (default: ``HarnessConfig()``).
            splits: Pre-loaded dataset splits (if None, load from registry).
            scenario_name: Optional canonical scenario key from
                ``CANONICAL_SCENARIOS``.  If provided:

                * The scenario's dataset is loaded with upstream kwargs.
                * Scenario metadata (``_alpha``, ``_calibration_fraction``,
                  ``_source_url``) is attached to the splits dict.
                * Must be declared in ``self.canonical_scenarios``.

        Returns:
            List of result rows, one per method.
        """
        cfg = config or HarnessConfig()

        # --- Resolve canonical scenario (reusable helper for sub-suites) ---
        dataset_name, splits = self._prepare_scenario_run(
            scenario_name,
            dataset_name,
            splits,
            standardize_targets=cfg.standardize_targets,
        )

        # --- Load dataset if not already loaded ---
        if splits is None:
            from .datasets import load_dataset

            splits = load_dataset(dataset_name, standardize_targets=cfg.standardize_targets)

        # Data extracted once (splits is guaranteed non-None after the guard above)
        X_train: np.ndarray = splits["X_train"]
        y_train: np.ndarray = splits["y_train"]
        X_test: np.ndarray = splits["X_test"]
        y_test: np.ndarray = splits["y_test"]

        if cfg.smoke:
            # Tiny subset for CI smoke checks
            n = min(100, len(X_train))
            X_train, y_train = X_train[:n], y_train[:n]
            X_test, y_test = X_test[: min(50, len(X_test))], y_test[: min(50, len(y_test))]

        rows: list[dict[str, Any]] = []

        for method in self.methods():
            set_harness_seed(cfg.seed)

            try:
                # Fit
                t0 = perf_counter()
                model = self.fit(method, splits, cfg)
                fit_time = perf_counter() - t0

                # Predict
                t0 = perf_counter()
                predictions: dict[str, Any] = self.predict(method, model, X_test, cfg)
                pred_time = perf_counter() - t0

                # Attach y-scaler so evaluate() can inverse-transform
                if "_y_scaler" in splits:
                    predictions["_y_scaler"] = splits["_y_scaler"]

                # Evaluate
                metrics = self.evaluate(method, predictions, y_test, cfg)

                row: dict[str, Any] = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "LibraryVersion": self.library_version(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    **metrics,
                    "train_s": fit_time,
                    "eval_s": pred_time,
                    "SourceURL": splits.get("_source_url", ""),
                    "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                    "Device": str(current_device()),
                    "Notes": self.method_notes(method),
                }

            except ImportError as exc:
                row = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "LibraryVersion": self.library_version(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    "RMSE": None,
                    "MAE": None,
                    "train_s": None,
                    "eval_s": None,
                    "SourceURL": splits.get("_source_url", ""),
                    "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                    "Device": str(current_device()),
                    "Notes": f"skipped: {type(exc).__name__}: {exc}",
                }

            except Exception as exc:
                row = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "LibraryVersion": self.library_version(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    "RMSE": None,
                    "MAE": None,
                    "train_s": None,
                    "eval_s": None,
                    "SourceURL": splits.get("_source_url", ""),
                    "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                    "Device": str(current_device()),
                    "Notes": f"failed: {type(exc).__name__}: {exc}",
                }

            rows.append(row)

        return rows

    def run_all_datasets(
        self,
        config: HarnessConfig | None = None,
        *,
        scenario_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run the suite on all supported datasets (or a single canonical scenario).

        If ``scenario_name`` is provided, runs only that scenario.
        Otherwise iterates over ``self.supported_datasets``.
        """
        cfg = config or HarnessConfig()
        all_rows: list[dict[str, Any]] = []

        if scenario_name is not None:
            all_rows.extend(self.run(scenario_name=scenario_name, config=cfg))
        else:
            for dataset_name in self.supported_datasets:
                all_rows.extend(self.run(dataset_name, cfg))

        return all_rows
