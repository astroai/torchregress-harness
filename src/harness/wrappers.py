"""PROPOSAL #3 — sklearn-style wrappers for torchregress probabilistic regressors.

Public API
----------
- :class:`GaussianNLLRegressor`      — diagonal Gaussian NLL
- :class:`MDNRegressor`              — mixture density network
- :class:`EvidentialRegressor`       — normal-inverse-gamma (NIG) evidential loss
- :class:`NormalizingFlowRegressor`  — zuko NSF conditional flow

Each wrapper exposes the sklearn-compatible protocol::

    import numpy as np
    reg = GaussianNLLRegressor(epochs=60, seed=42)
    reg.fit(X_train, y_train)                  # X: np.ndarray OR pd.DataFrame
                                               # y: np.ndarray OR pd.Series (1-D)
    mean       = reg.predict(X_test)           # np.ndarray output
    lower, upper = reg.predict_interval(X_test, confidence=0.9)
    samples    = reg.sample(X_test, n_samples=100)

PROPOSAL #8 — pandas DataFrame input adapters
---------------------------------------------
* ``X`` may be a ``pd.DataFrame`` (column-name aware) or ``np.ndarray``.
* ``y`` may be a ``pd.Series`` (with ``name`` preserved) or ``np.ndarray``.
* At fit-time, column names are captured in ``reg.feature_names_in_`` and
  ``reg.n_features_in_``; the target name (when ``y`` is a ``pd.Series``
  with ``name`` or a single-column ``pd.DataFrame``) is captured in
  ``reg.target_name_``.  This mirrors the sklearn introspection pattern.
* At predict-time, ``predict(X_df)`` validates that
  ``set(X_df.columns) == set(reg.feature_names_in_)`` (no missing, no
  extra columns) and reorders columns to the fit-time order before
  feeding the model.  Mixed numeric / non-numeric DataFrames raise a
  ``TypeError`` at fit-time with the offending column names listed.
* Outputs remain ``np.ndarray`` (per sklearn convention).  Users that
  need ``pd.Series`` / ``pd.DataFrame`` outputs can reconstruct names
  from ``reg.feature_names_in_`` / ``reg.target_name_``.

PROPOSAL #3 (P0) was: "Create high-level estimator classes: FlowRegressor with
.fit(X, y) and .predict(X); EvidentialRegressor with .fit() and .predict_interval();
MDNRegressor. These should internally handle: data conversion, model creation,
training loop, evaluation mode switching."

Differences from sklearn
-----------------------
* All wrappers train via Adam mini-batches on CPU (GPU supported by moving
  ``self._model.to('cuda')`` after ``.fit``).
* ``.predict_interval`` returns central intervals at ``confidence`` via:
    * analytic Gaussian quantiles (GaussianNLLRegressor)
    * empirical MC quantiles drawn from the predictive distribution
      (MDN / NormalizingFlow / Evidential)
* ``.sample`` returns the raw MC samples from the predictive distribution,
  shape ``(N, n_samples)``.
* A ``set_harness_seed(self.seed)`` call inside ``.fit`` keeps each
  wrapper deterministically reproducible; repeated ``.predict()`` /
  ``.predict_interval()`` / ``.sample()`` calls on the same fitted estimator
  produce **bit-identical MC samples** because each wrapper materialises a
  per-instance ``self._inference_generator`` (``torch.Generator``) at
  fit-time and re-asserts its seed (via ``torch.random.fork_rng`` +
  ``torch.manual_seed``) inside every ``_predict_diagnostics``.  The
  fork_rng context exits with the caller's global RNG state restored, so
  no leaks to surrounding code.  ``reg._inference_generator`` is publicly
  available as an introspection handle after ``.fit()`` runs
  (``reg._inference_generator.initial_seed()``); ``reg._inference_seed``
  is the corresponding integer cache (defaults to ``0`` pre-fit — only
  meaningful after ``.fit()``).

Validation
----------
Smoke tests in :mod:`tests.test_wrappers` exercise the full
fit → predict → predict_interval → sample flow on synthetic data
(50-60 samples, 3 features), with an Evidential-specific regression guard
``test_evidential_respects_confidence_arg`` that catches a previous
hardcoded-0.9 bug.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from harness.core import set_harness_seed
from harness.runners import HarnessConfig

# ---------------------------------------------------------------------------
# Shared input validation (PROPOSAL #8 — DataFrame input adapters)
# ---------------------------------------------------------------------------


def _coerce_features(
    X,
    *,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str] | None]:
    """Coerce ``X`` to shaped ``(N, D)`` float32 + capture/validate column names.

    Accepts ``pd.DataFrame`` (column-aware) or ``np.ndarray`` (or any
    array-like coercible via ``np.asarray``).  Returns the float32 numpy
    array plus the column-name list (None when ``X`` wasn't a DataFrame
    or hadn't been previously validated against ``feature_names``).

    Parameters
    ----------
    X : pd.DataFrame | np.ndarray | array-like
    feature_names : optional list of str
        When provided (predict-time validation), ``X`` MUST be a
        ``pd.DataFrame`` whose column set matches ``feature_names`` exactly
        (no missing, no extras).  Columns are reordered to ``feature_names``
        order before numerical extraction (silent permutation tolerance,
        matching sklearn 1.x).  When ``None`` (fit-time or absence of
        prior DataFrame history), any column order is accepted and the
        observed column order is recorded for later validation.

    Raises
    ------
    TypeError
        If ``X`` is a ``pd.DataFrame`` containing non-numeric dtype columns.
    ValueError
        If ``X`` ends up with ``ndim != 2`` after coercion, or if a
        predict-time DataFrame has column set mismatched with ``feature_names``.
        Also raised if the input data contains NaN or infinity.
    """
    if isinstance(X, pd.DataFrame):
        # ``pd.api.types.is_numeric_dtype`` per column is the canonical
        # dtype check; it correctly classifies int/float/bool/complex as
        # numeric and excludes object/string/categorical.  We reject
        # any non-numeric column at fit-time with a clear message.
        non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        if non_numeric:
            raise TypeError(
                f"X DataFrame has non-numeric columns: {non_numeric}; "
                "all DataFrame columns must be a numeric dtype."
            )
        col_list = list(X.columns)
        values = X.to_numpy(dtype=np.float32, copy=False)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
            col_list = col_list[:1]
        if values.ndim != 2:
            raise ValueError(f"X must be 1D or 2D, got shape {values.shape}")

        # Predict-time column validation + reorder
        if feature_names is not None:
            cols_set = set(col_list)
            fit_set = set(feature_names)
            if cols_set != fit_set:
                missing = sorted(fit_set - cols_set)
                extra = sorted(cols_set - fit_set)
                raise ValueError(
                    f"predict(X) DataFrame columns mismatch — "
                    f"missing={missing}, extra={extra}; "
                    f"expected {feature_names}"
                )
            if col_list != feature_names:
                values = X[feature_names].to_numpy(dtype=np.float32, copy=False)
            ret_cols = feature_names
        else:
            ret_cols = col_list

    else:
        # Fallback path: numpy ndarray or array-like
        if isinstance(X, np.ndarray):
            values = X.astype(np.float32, copy=False)
        else:
            try:
                values = np.asarray(X, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"X must be a numpy ndarray or pandas DataFrame, got {type(X).__name__}"
                ) from exc
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2:
            raise ValueError(f"X must be 1D or 2D, got shape {values.shape}")
        ret_cols = None

    # Check for NaN / Inf
    if not np.isfinite(values).all():
        raise ValueError("Input X contains NaN or infinity.")

    return values, ret_cols


def _coerce_target(y, *, expected_len: int | None = None) -> tuple[np.ndarray, str | None]:
    """Coerce ``y`` to ``(N,)`` float32 + capture the target's name.

    Accepts ``np.ndarray``, ``pd.Series`` (1-D with ``name``), or single-column
    ``pd.DataFrame`` (column label = target name).  Multi-column DataFrames
    raise ``ValueError`` because the four PROPOSAL #3 wrappers target scalar
    regression — multivariate targets live in the dedicated suite.

    Returns
    -------
    (np.ndarray of shape (N,) dtype float32, str or None).
    """
    if isinstance(y, pd.Series):
        if expected_len is not None and len(y) != expected_len:
            raise ValueError(f"y length {len(y)} != X length {expected_len}")
        values = y.to_numpy(dtype=np.float32).reshape(-1)
        name = y.name if isinstance(y.name, str) else None
    elif isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(
                f"y must be scalar (1-D Series or 1-D ndarray) for the "
                f"single-output PROPOSAL #3 wrappers; got pd.DataFrame "
                f"with {y.shape[1]} columns.  Use the multivariate suite "
                f"directly for multi-output regression."
            )
        if expected_len is not None and len(y) != expected_len:
            raise ValueError(f"y length {len(y)} != X length {expected_len}")
        values = y.iloc[:, 0].to_numpy(dtype=np.float32).reshape(-1)
        name = str(y.columns[0])
    elif isinstance(y, np.ndarray):
        if expected_len is not None and len(y) != expected_len:
            raise ValueError(f"y length {len(y)} != X length {expected_len}")
        values = y.astype(np.float32, copy=False).reshape(-1)
        name = None
    else:
        try:
            values = np.asarray(y, dtype=np.float32).reshape(-1)
            if expected_len is not None and len(values) != expected_len:
                raise ValueError(f"y length {len(values)} != X length {expected_len}")
            name = None
        except Exception as exc:
            if isinstance(exc, ValueError) and "length" in str(exc):
                raise
            raise TypeError(
                f"y must be a numpy ndarray, pandas Series, or array-like, got {type(y).__name__}"
            ) from exc

    # Check for NaN / Inf
    if not np.isfinite(values).all():
        raise ValueError("Input y contains NaN or infinity.")

    return values, name


# ---------------------------------------------------------------------------
# Diagnostics helpers — turn a fitted model into a dict the base predict
# protocol can consume (``mean`` / ``std`` / ``samples`` / ``lower`` / ``upper``)
# ---------------------------------------------------------------------------


def _diagnostics_to_mean(diag: dict[str, np.ndarray]) -> np.ndarray:
    """Return predictive mean from a diagnostics dict.  Prefers ``samples``
    (consistent with ``predict_interval`` quantile estimates) over ``mean`` /
    ``y_pred`` so all three predict APIs share one source of uncertainty."""
    if "samples" in diag:
        return np.asarray(diag["samples"]).mean(axis=1)
    if "mean" in diag:
        return diag["mean"]
    if "y_pred" in diag:
        return diag["y_pred"]
    raise ValueError(f"No mean signal in diagnostics: keys={sorted(diag)}")


def _diagnostics_to_interval(
    diag: dict[str, np.ndarray],
    *,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lower, upper)`` central intervals at the given ``confidence``.

    Dispatches (first match wins):
    * ``samples``     → empirical MC quantiles (NF / MDN / Evidential path).
    * ``mean``+``std`` → analytic Gaussian quantiles (GaussianNLL path).
    * ``lower``+``upper`` → pass-through (caller already computed).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")
    alpha = 1.0 - confidence
    if "samples" in diag:
        s = np.asarray(diag["samples"])
        return (
            np.quantile(s, alpha / 2, axis=1),
            np.quantile(s, 1.0 - alpha / 2, axis=1),
        )
    if {"mean", "std"}.issubset(diag):
        from scipy.stats import norm

        z = float(norm.ppf(1.0 - alpha / 2))
        return diag["mean"] - z * diag["std"], diag["mean"] + z * diag["std"]
    if {"lower", "upper"}.issubset(diag):
        return np.asarray(diag["lower"]), np.asarray(diag["upper"])
    raise ValueError(f"No interval signal in diagnostics: keys={sorted(diag)}")


def _diagnostics_to_samples(
    diag: dict[str, np.ndarray],
    *,
    n_samples: int,
) -> np.ndarray:
    """Return MC samples, shape ``(N, n_samples)``, from a diagnostics dict.

    Resolution rules:
    * If ``samples`` is present, slice the first ``min(n_samples, pool)``
      rows from the per-row pool — never substitute a different
      distribution's samples silently.
    * Else, fall back to Gaussian resampling via ``mean``+``std``.

    Note: the contract is "if the user asked for more samples than the
    pool holds, return the largest available slice rather than swap
    distributions".  Callers wanting larger pools should re-train with
    more ``NF_N_SAMPLES`` / ``EVIDENTIAL_N_SAMPLES`` / ask
    ``MixtureDensityLoss.sample`` for more draws.
    """
    if "samples" in diag:
        s = np.asarray(diag["samples"])
        if s.ndim == 1:  # defensive: a 1-D pool is treated as N=1, S=len
            return s[:n_samples]
        # Take as many as the pool provides — caller can decide what to do
        # if n_samples exceeds pool size (we never resample from a different
        # distribution).
        n_pool = s.shape[1]
        return s[:, : min(n_samples, n_pool)].copy()
    if {"mean", "std"}.issubset(diag):
        rng = np.random.default_rng(0)
        n = diag["mean"].shape[0]
        return (
            diag["mean"][:, None] + diag["std"][:, None] * rng.standard_normal((n, n_samples))
        ).astype(np.float64)
    raise ValueError(f"No sample signal in diagnostics: keys={sorted(diag)}")


# ---------------------------------------------------------------------------
# Base protocol
# ---------------------------------------------------------------------------


class _HarnessRegressorBase:
    """Common sklearn-style protocol for torchregress probabilistic regressors.

    Subclasses MUST define:
    * ``_build_model(in_dim, hidden)``  — populate ``self._model`` (+ extras).
    * ``_default_loss_factory()``      — return the loss ``nn.Module`` used
      by the default Adam training loop.  Required only when ``_fit_impl``
      is NOT overridden; ``EvidentialRegressor`` and
      ``NormalizingFlowRegressor`` override ``_fit_impl`` and therefore
      need not define this factory.
    * ``_predict_diagnostics(X)`` — return a dict consumed by the three
      diagnostic helpers above.

    Subclasses MAY override:
    * ``_fit_impl(X_t, y_t, cfg)`` — custom training loop used by
      ``EvidentialRegressor`` (softplus NIG constraints) and
      ``NormalizingFlowRegressor`` (passes flow coupling-network params via
      ``loss_params=``).
    """

    def to(self, device: str | torch.device) -> "_HarnessRegressorBase":
        """Move the model and all submodules/losses to the specified device."""
        if hasattr(self, "_model") and self._model is not None:
            self._model.to(device)
        if hasattr(self, "_loss") and self._loss is not None:
            self._loss.to(device)
        if hasattr(self, "_loss_fn") and self._loss_fn is not None:
            self._loss_fn.to(device)
        if hasattr(self, "_flow") and self._flow is not None:
            self._flow.to(device)
        return self


    def __init__(
        self,
        *,
        epochs: int = 60,
        batch_size: int = 64,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self._fitted: bool = False
        # Per-instance inference seed, captured in ``.fit()``.  The
        # canonical re-seeding target inside ``_predict_diagnostics``
        # (consumed by ``torch.manual_seed`` after entering
        # ``torch.random.fork_rng``).  Defaults to ``0`` pre-fit (safe but
        # not meaningful — re-read after ``.fit()`` for the actual seed).
        self._inference_seed: int = 0
        # Per-instance ``torch.Generator`` materialised in ``.fit()`` —
        # publicly introspectable as ``reg._inference_generator``.  Carries
        # the same seed as ``_inference_seed``; useful when a downstream
        # caller wants to thread the inference RNG into user-supplied
        # samplers or extend determinism to private MC paths.
        self._inference_generator: torch.Generator | None = None
        # Sklearn-style introspection (PROPOSAL #8).  Populated in ``.fit()``
        # when ``X`` is a ``pd.DataFrame`` or ``y`` is a ``pd.Series`` /
        # single-column ``pd.DataFrame``.  Defaults ``None`` so non-DataFrame
        # callers can still rely on the same attribute names without surprises.
        self.n_features_in_: int | None = None
        self.feature_names_in_: list[str] | None = None
        self.target_name_: str | None = None

    # ----- sklearn-style public API -----

    @property
    def name(self) -> str:
        return type(self).__name__

    def fit(self, X, y) -> "_HarnessRegressorBase":
        """Fit the wrapper to ``(X, y)``.  Returns self for chaining.

        ``X`` may be a ``np.ndarray`` (or array-like) OR a ``pd.DataFrame``
        (column-name aware).  ``y`` may be a ``np.ndarray`` (or array-like),
        a 1-D ``pd.Series`` (with ``name`` preserved), or a single-column
        ``pd.DataFrame``.  See PROPOSAL #8 in the module docstring for the
        full DataFrame contract and the sklearn-style introspection
        attributes set on success (``n_features_in_``, ``feature_names_in_``,
        ``target_name_``).
        """
        X_arr, feature_names = _coerce_features(X)
        y_arr, target_name = _coerce_target(y, expected_len=len(X_arr))

        in_dim = X_arr.shape[1]
        from suites._backbones import _hidden_size
        hidden = _hidden_size(in_dim)

        # Seed BEFORE ``_build_model`` so weight init is deterministic
        # across consecutive ``.fit()`` calls (global RNG must be set
        # before ``nn.Linear`` reads it).
        set_harness_seed(self.seed)
        self._build_model(in_dim, hidden)
        cfg = HarnessConfig(
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            seed=self.seed,
        )
        # Capture the per-instance inference seed and materialise the
        # corresponding ``torch.Generator``.  The int captures the
        # canonical seed re-asserted at the start of every
        # ``_predict_diagnostics`` call (inside a ``fork_rng`` block);
        # the Generator instance is publicly available as
        # ``reg._inference_generator`` for downstream introspection or
        # extending determinism to private MC paths.
        self._inference_seed = int(self.seed)
        self._inference_generator = torch.Generator().manual_seed(self._inference_seed)
        # Populate sklearn-style DataFrame introspection attributes
        # (PROPOSAL #8).  ``n_features_in_`` mirrors sklearn; the other
        # two are only set when the fit-time input was a pandas object.
        self.n_features_in_ = in_dim
        self.feature_names_in_ = feature_names
        self.target_name_ = target_name
        X_t = torch.from_numpy(X_arr).float()
        y_t = torch.from_numpy(y_arr).float().reshape(-1, 1)
        self._fit_impl(X_t, y_t, cfg)
        self._fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Return point predictions = predictive mean, shape ``(N,)``.

        ``X`` may be a ``np.ndarray`` or ``pd.DataFrame``.  When the fit-time
        ``X`` was a ``pd.DataFrame``, ``X`` is expected to be a ``pd.DataFrame``
        too — its columns are validated against ``feature_names_in_`` and
        reordered to match the fit-time order before the prediction runs.
        """
        if not self._fitted:
            raise RuntimeError(f"{self.name} not fitted; call .fit(X, y) first.")
        X_arr, _ = _coerce_features(X, feature_names=self.feature_names_in_)
        return _diagnostics_to_mean(self._predict_diagnostics(X_arr))

    def predict_interval(
        self,
        X,
        *,
        confidence: float = 0.9,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)`` central intervals at the given confidence.

        Accepts ``np.ndarray`` or ``pd.DataFrame`` (column validation +
        reordering applied when ``feature_names_in_`` is set; see ``predict``).
        """
        if not self._fitted:
            raise RuntimeError(f"{self.name} not fitted; call .fit(X, y) first.")
        X_arr, _ = _coerce_features(X, feature_names=self.feature_names_in_)
        return _diagnostics_to_interval(
            self._predict_diagnostics(X_arr),
            confidence=confidence,
        )

    def sample(self, X, *, n_samples: int = 100) -> np.ndarray:
        """Return MC samples, shape ``(N, n_samples)``, from the predictive distribution.

        Accepts ``np.ndarray`` or ``pd.DataFrame`` (column validation +
        reordering applied when ``feature_names_in_`` is set; see ``predict``).
        """
        if not self._fitted:
            raise RuntimeError(f"{self.name} not fitted; call .fit(X, y) first.")
        X_arr, _ = _coerce_features(X, feature_names=self.feature_names_in_)
        return _diagnostics_to_samples(
            self._predict_diagnostics(X_arr),
            n_samples=n_samples,
        )

    # ----- subclass hooks -----

    def _build_model(self, in_dim: int, hidden: int) -> None:
        raise NotImplementedError

    def _fit_impl(
        self,
        X_t: torch.Tensor,
        y_t: torch.Tensor,
        cfg: HarnessConfig,
    ) -> None:
        from suites._backbones import _train_one
        loss_fn = self._default_loss_factory()
        _train_one(self._model, loss_fn, X_t, y_t, cfg)
        self._model.eval()

    def _default_loss_factory(self) -> nn.Module:
        """Build the loss used by the default Adam training loop.

        Required only when ``_fit_impl`` is NOT overridden.  Evidential
        and NormalizingFlow override ``_fit_impl`` directly and don't need
        to implement this factory.
        """
        raise NotImplementedError(
            f"{self.name}: _default_loss_factory() not implemented; "
            "either implement it for the default train loop, or override "
            "_fit_impl directly."
        )

    def _predict_diagnostics(self, X: np.ndarray) -> dict[str, np.ndarray]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete wrappers
# ---------------------------------------------------------------------------


class GaussianNLLRegressor(_HarnessRegressorBase):
    """Diagonal-Gaussian NLL estimator (Proposal #3 P0 #3).

    Predictive distribution: Gaussian with per-row mean and std derived from
    a 2-output MLP (column 0 = mean, column 1 = log-variance).  Intervals
    are analytic z-score intervals via ``scipy.stats.norm.ppf``.

    .predict returns ``mean`` directly (since no MC samples are drawn).
    """

    def _build_model(self, in_dim: int, hidden: int) -> None:
        from suites._backbones import _MLP
        self._model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=2)

    def _default_loss_factory(self) -> nn.Module:
        from torchregress.losses import GaussianNLLLoss

        return GaussianNLLLoss()

    def _predict_diagnostics(self, X: np.ndarray) -> dict[str, np.ndarray]:
        from suites._backbones import _gaussian_nll_predict
        mean, std = _gaussian_nll_predict(self._model, X)
        return {"mean": mean, "std": std}


class MDNRegressor(_HarnessRegressorBase):
    """Mixture-Density-Network estimator (Proposal #3 P0 #3).

    Parameters
    ----------
    n_components : int
        Number of mixture components (default 3, matching
        ``distributional_bins.MDN_K3``).
    n_features : int
        Output dimensionality (default 1 = scalar regression).
    """

    def __init__(
        self,
        *,
        n_components: int = 3,
        n_features: int = 1,
        epochs: int = 60,
        batch_size: int = 64,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> None:
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
        self.n_components = n_components
        self.n_features = n_features
        self._loss_fn: nn.Module | None = None

    def _build_model(self, in_dim: int, hidden: int) -> None:
        from torchregress.losses import MixtureDensityLoss

        self._loss_fn = MixtureDensityLoss(
            n_components=self.n_components,
            n_features=self.n_features,
            covariance_type="diagonal",
        )
        from suites._backbones import _MLP
        self._model = _MLP(
            in_dim=in_dim,
            hidden=hidden,
            out_dim=self._loss_fn.expected_output_size,
        )

    def _default_loss_factory(self) -> nn.Module:
        assert self._loss_fn is not None
        return self._loss_fn

    def _predict_diagnostics(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Predictive mean/std from ``predict_mean_std`` + MC samples from
        ``loss_fn.sample``.  Forces ``samples`` to ``(n_test, n_samples)``
        so single-row edge cases (``n_test == 1``) don't get collapsed to 1-D.
        """
        assert self._loss_fn is not None
        device = next(self._model.parameters()).device
        X_t = torch.from_numpy(X).float().to(device)
        n_test = X_t.shape[0]
        with torch.no_grad():
            out = self._model(X_t)
            mean, std = self._loss_fn.predict_mean_std(out)
            # Re-assert the cached fit-time seed inside ``fork_rng`` so
            # repeated ``.predict()`` calls produce bit-identical MC
            # samples irrespective of intervening global-RNG consumption;
            # the context exits with the caller's RNG state restored
            # (no leak to the wider harness).
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self._inference_seed)
                samples = self._loss_fn.sample(out, n_samples=256).squeeze(-1)
            # Accept (N, S), (S, N), or 1-D fallback for n_test=1: always
            # reshape to (n_test, n_samples) so downstream consumers see
            # consistent 2-D shape.
            if samples.ndim == 1:
                samples = samples.reshape(n_test, -1)
            elif samples.ndim == 2 and samples.shape[0] != n_test and samples.shape[1] == n_test:
                samples = samples.T  # (S, N) → (N, S)
            else:
                samples = samples.reshape(n_test, -1)
        return {
            "mean": mean.squeeze(-1).cpu().numpy(),
            "std": std.squeeze(-1).cpu().numpy(),
            "samples": samples.cpu().numpy(),
        }


class EvidentialRegressor(_HarnessRegressorBase):
    """NIG-evidential estimator (Proposal #3 P0 #3).

    Predictive distribution: Normal-Inverse-Gamma posterior → Student-t
    with per-row ``(gamma, nu, alpha, beta)`` parameters (closed form
    documented in :func:`harness.metrics._pit_from_student_t`):

        df       = 2 α
        loc      = γ
        scale²   = β·(1 + ν) / (να)

    Sampling uses ``torch.distributions.StudentT`` (broadcasts per-row
    df/loc/scale correctly with ``sample(sample_shape)``), seeded via
    ``torch.manual_seed`` for version-tolerance (the ``generator=`` kwarg
    isn't supported across all torch versions).  ``EVIDENTIAL_N_SAMPLES =
    512`` (vs 256 for NF/MDN) lowers variance of 99%-quantile estimates on
    the heavy-tailed Student-t predictive with df ≈ 2.
    """

    EVIDENTIAL_N_SAMPLES: int = 512

    def __init__(
        self,
        *,
        coeff_nig: float = 0.01,
        epochs: int = 60,
        batch_size: int = 64,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> None:
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
        self.coeff_nig = coeff_nig
        self._loss_fn: nn.Module | None = None

    def _build_model(self, in_dim: int, hidden: int) -> None:
        from torchregress.losses import EvidentialRegressionLoss

        self._loss_fn = EvidentialRegressionLoss(coeff_nig=self.coeff_nig)
        from suites._backbones import _MLP
        # 4 raw outputs (gamma direct, nu/alpha/beta softplus-bounded below).
        self._model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=4)

    def _fit_impl(
        self,
        X_t: torch.Tensor,
        y_t: torch.Tensor,
        cfg: HarnessConfig,
    ) -> None:
        """Train with softplus-constrained ``(nu > 0, alpha > 1, beta > 0)`` outputs.

        Mirrors ``suites/tabular/distributional_bins._train_evidential`` so
        wrapper behaviour matches the suite's reference fit exactly.
        """
        import torch.nn.functional as F
        import torch.optim as optim

        assert self._loss_fn is not None
        params = list(self._model.parameters())
        for p in self._loss_fn.parameters():
            if p.requires_grad:
                params.append(p)
        opt = optim.Adam(params, lr=cfg.lr)

        device = next(self._model.parameters()).device
        X_dev = X_t.to(device)
        y_dev = y_t.to(device)
        n = X_dev.shape[0]
        for _ in range(cfg.epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, cfg.batch_size):
                idx = perm[i : i + cfg.batch_size]
                opt.zero_grad(set_to_none=True)
                raw = self._model(X_dev[idx])
                gamma = raw[:, 0:1]
                nu = F.softplus(raw[:, 1:2]) + 0.01
                alpha = F.softplus(raw[:, 2:3]) + 1.01
                beta = F.softplus(raw[:, 3:4]) + 0.01
                constrained = torch.cat([gamma, nu, alpha, beta], dim=1)
                loss = self._loss_fn(constrained, y_dev[idx])
                loss.backward()
                opt.step()
        self._model.eval()

    def _predict_diagnostics(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Sample Student-t predictive → MC samples + NIG parameters.

        Returns ``samples`` so ``predict_interval`` properly respects the
        user's ``confidence`` kwarg (regression guard tested by
        ``tests/test_wrappers.py::test_evidential_respects_confidence_arg``).

        Uses ``torch.manual_seed`` rather than
        ``torch.distributions.StudentT.sample(generator=...)`` for
        cross-version compatibility — ``Distribution.sample``'s ``generator=``
        kwarg was added in torch 2.0+, so we keep the global seed path
        portable.
        """
        import torch.nn.functional as F

        assert self._loss_fn is not None
        device = next(self._model.parameters()).device
        X_t = torch.from_numpy(X).float().to(device)
        n_test = X_t.shape[0]
        with torch.no_grad():
            raw = self._model(X_t)
            gamma = raw[:, 0:1]
            nu = F.softplus(raw[:, 1:2]) + 0.01
            alpha = F.softplus(raw[:, 2:3]) + 1.01
            beta = F.softplus(raw[:, 3:4]) + 0.01

        gamma_np = gamma.cpu().numpy().astype(np.float64).reshape(-1)
        nu_np = nu.cpu().numpy().astype(np.float64).reshape(-1)
        alpha_np = alpha.cpu().numpy().astype(np.float64).reshape(-1)
        beta_np = beta.cpu().numpy().astype(np.float64).reshape(-1)

        # Closed form matches ``_pit_from_student_t`` in metrics.py so MC
        # samples and PIT-CDF agree on the same data-generating process.
        df_t = torch.as_tensor(2.0 * alpha_np, dtype=torch.float32, device=device)
        loc_t = torch.as_tensor(gamma_np, dtype=torch.float32, device=device)
        scale_np = np.sqrt(beta_np * (1.0 + nu_np) / (nu_np * alpha_np))
        scale_t = torch.as_tensor(scale_np, dtype=torch.float32, device=device)

        dist = torch.distributions.StudentT(df=df_t, loc=loc_t, scale=scale_t)
        # ``sample((S,))`` returns shape (S, N); .T → (N, S).
        # Re-assert the cached fit-time seed inside ``fork_rng`` so
        # repeated ``.predict()`` calls produce bit-identical MC
        # samples irrespective of intervening global-RNG consumption;
        # the context exits with the caller's RNG state restored (no leak).
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._inference_seed)
            samples = dist.sample((self.EVIDENTIAL_N_SAMPLES,))
        samples = samples.T  # (N, S) regardless of internal layout
        if samples.ndim == 1:
            samples = samples.reshape(n_test, -1)
        samples_np = samples.cpu().numpy().astype(np.float64)

        return {
            "samples": samples_np,
            "gamma": gamma_np,
            "nu": nu_np,
            "alpha": alpha_np,
            "beta": beta_np,
        }


class NormalizingFlowRegressor(_HarnessRegressorBase):
    """Conditional Normalizing-Flow estimator (Proposal #3 P0 #3).

    Internally uses a zuko NSF flow with the same architecture as
    ``distributional_bins.NormalizingFlow``: ``hidden_features=[64, 64]``,
    ``n_transforms=3``, context MLP size = ``_hidden_size(in_dim)``.

    The wrapper targets **scalar regression** (``n_features=1``).  For
    multivariate regression, use
    ``suites/tabular/multivariate_intervals.MultivariateIntervalsSuite``.
    """

    NF_HIDDEN: tuple[int, ...] = (64, 64)
    NF_TRANSFORMS: int = 3
    NF_N_SAMPLES: int = 256

    def _build_model(self, in_dim: int, hidden: int) -> None:
        from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model

        context_dim = hidden
        self._flow = create_flow_model(
            n_features=1,
            context_dim=context_dim,
            flow_type="nsf",
            n_transforms=self.NF_TRANSFORMS,
            hidden_features=list(self.NF_HIDDEN),
        )
        self._loss = NormalizingFlowLoss(flow=self._flow)
        from suites._backbones import _MLP
        self._model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=context_dim)

    def _fit_impl(
        self,
        X_t: torch.Tensor,
        y_t: torch.Tensor,
        cfg: HarnessConfig,
    ) -> None:
        """Train backbone MLP + flow's coupling-network weights jointly."""
        from suites._backbones import _train_one
        _train_one(
            self._model,
            self._loss,
            X_t,
            y_t,
            cfg,
            loss_params=list(self._flow.parameters()),
        )
        self._model.eval()

    def _predict_diagnostics(self, X: np.ndarray) -> dict[str, np.ndarray]:
        device = next(self._model.parameters()).device
        X_t = torch.from_numpy(X).float().to(device)
        n_test = X_t.shape[0]
        with torch.no_grad():
            context = self._model(X_t)
            # Re-assert the cached fit-time seed inside ``fork_rng`` so
            # repeated ``.predict()`` calls produce bit-identical MC
            # samples irrespective of intervening global-RNG consumption;
            # the context exits with the caller's RNG state restored (no leak).
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self._inference_seed)
                samples = self._loss.sample(context, n_samples=self.NF_N_SAMPLES).squeeze(-1)
            # Force 2-D (N, S); handle (S, N), 1-D edge cases.
            if samples.ndim == 1:
                samples = samples.reshape(n_test, -1)
            elif samples.ndim == 2 and samples.shape[0] != n_test and samples.shape[1] == n_test:
                samples = samples.T
            else:
                samples = samples.reshape(n_test, -1)
        return {"samples": samples.cpu().numpy()}


class DeepEnsembleRegressor(_HarnessRegressorBase):
    """Deep Ensemble regressor wrapping torchregress.ensemble.models.DeepEnsemble."""

    def __init__(
        self,
        *,
        ensemble_size: int = 5,
        epochs: int = 60,
        batch_size: int = 64,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> None:
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
        self.ensemble_size = ensemble_size

    def _build_model(self, in_dim: int, hidden: int) -> None:
        from torchregress.ensemble.models import DeepEnsemble
        from suites._backbones import _MLP
        self._model = DeepEnsemble(
            base_model=_MLP,
            ensemble_size=self.ensemble_size,
            device="cpu",
            in_dim=in_dim,
            hidden=hidden,
            out_dim=1,
        )

    def _fit_impl(
        self,
        X_t: torch.Tensor,
        y_t: torch.Tensor,
        cfg: HarnessConfig,
    ) -> None:
        device = next(self._model.parameters()).device
        loss_fn = nn.MSELoss()
        
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self._model.fit(
            train_loader=loader,
            criterion=loss_fn,
            epochs=self.epochs,
            lr=self.lr,
            verbose=False,
            device=device,
        )
        self._model.eval()

    def _predict_diagnostics(self, X: np.ndarray) -> dict[str, np.ndarray]:
        device = next(self._model.parameters()).device
        X_t = torch.from_numpy(X).float().to(device)
        with torch.no_grad():
            res = self._model.predict(X_t)
            mean = res["mean"].squeeze(-1).cpu().numpy()
            var = res["variance"].squeeze(-1).cpu().numpy()
            std = np.sqrt(np.clip(var, 1e-8, None))
        return {"mean": mean, "std": std}


class HeteroscedasticEnsembleRegressor(_HarnessRegressorBase):
    """Heteroscedastic Ensemble regressor wrapping torchregress.ensemble.models.HeteroscedasticEnsembleModel."""

    def __init__(
        self,
        *,
        ensemble_size: int = 5,
        beta: float = 0.5,
        epochs: int = 60,
        batch_size: int = 64,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> None:
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
        self.ensemble_size = ensemble_size
        self.beta = beta

    def _build_model(self, in_dim: int, hidden: int) -> None:
        from torchregress.ensemble.models import HeteroscedasticEnsembleModel
        from suites._backbones import _MLP
        self._model = HeteroscedasticEnsembleModel(
            base_model=_MLP,
            ensemble_size=self.ensemble_size,
            device="cpu",
            in_dim=in_dim,
            hidden=hidden,
            out_dim=2,
        )

    def _fit_impl(
        self,
        X_t: torch.Tensor,
        y_t: torch.Tensor,
        cfg: HarnessConfig,
    ) -> None:
        device = next(self._model.parameters()).device
        from torchregress.losses import BetaNLLLoss
        loss_fn = BetaNLLLoss(beta=self.beta)
        
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self._model.fit(
            train_loader=loader,
            criterion=loss_fn,
            epochs=self.epochs,
            lr=self.lr,
            verbose=False,
            device=device,
        )
        self._model.eval()

    def _predict_diagnostics(self, X: np.ndarray) -> dict[str, np.ndarray]:
        device = next(self._model.parameters()).device
        X_t = torch.from_numpy(X).float().to(device)
        with torch.no_grad():
            res = self._model.predict(X_t)
            mean = res["mean"].squeeze(-1).cpu().numpy()
            var = res["variance"].squeeze(-1).cpu().numpy()
            std = np.sqrt(np.clip(var, 1e-8, None))
        return {"mean": mean, "std": std}


# ---------------------------------------------------------------------------
# Public API mirror
# ---------------------------------------------------------------------------


__all__ = [
    "DeepEnsembleRegressor",
    "EvidentialRegressor",
    "GaussianNLLRegressor",
    "HeteroscedasticEnsembleRegressor",
    "MDNRegressor",
    "NormalizingFlowRegressor",
]
