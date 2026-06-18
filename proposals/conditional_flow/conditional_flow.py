"""PROPOSAL #2 — Bundled ``ConditionalFlow`` estimator (working draft).

Scaffolds the upstream class targeted at ``torchregress``.  Self-contained aside
from ``torchregress`` itself (uses ``NormalizingFlowLoss`` and ``create_flow_model``
from ``torchregress.losses.nflows``).

Decisions encoded in this draft
-------------------------------
* **Naming** — ``ConditionalFlow`` (matches :doc:`PROPOSALS.md` §2 verbatim).
* **Default flow_type** — ``"nsf"`` (zuko NSF); MAF / RealNVP left as future work.
* **Bench-tested hyperparameters** — ``hidden_features=(64, 64)``,
  ``n_transforms=3`` (matches :doc:`PARITY_REPORT.md` §13/§14 sweet spot).
* **Coercion** — ``np.asarray`` only; ``pd.DataFrame`` is harness-side (PROPOSAL #8).
* **Target standardization** — left to the caller (sklearn convention).
* **Determinism** — per-instance ``torch.Generator`` captured in ``.fit()`` so
  repeated MC draws (in ``sample``/``predict``/``predict_interval``/``posterior``)
  are bit-identical regardless of intervening global-RNG consumption.
* **Persistence** — ``state_dict`` (``*.pt``) + JSON sidecar (``*.meta.json``).

Locked signatures (from installed ``torchregress`` v0.1.0)
----------------------------------------------------------
* ``NormalizingFlowLoss.forward(y_pred, target, mask=None, weights=None, **kwargs)``
* ``NormalizingFlowLoss.sample(y_pred, n_samples=1)``
* ``NormalizingFlowLoss.log_prob(y_pred, target)``
* ``create_flow_model(n_features, context_dim=0, flow_type='nsf', n_transforms=5,
  hidden_features=None, n_hidden_layers=None, **kwargs) -> torch.nn.Module``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- Real imports (installed torchregress >= 0.1.0) --------------------------------
try:
    from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "proposals.conditional_flow requires torchregress >= 0.1.0 with "
        "torchregress.losses.nflows.{NormalizingFlowLoss, create_flow_model}. "
        "Install via: uv pip install torchregress"
    ) from exc


# --- Inline backbone (zero-deps on harness-internal _MLP) -------------------------
class _FlowBackbone(nn.Module):
    """Backbone MLP: ``X -> context`` consumed by the zuko flow.

    Mirrors the topology used by ``suites/_backbones._MLP`` (input projection,
    two ReLU activations, output projection) but is reimplemented here so the
    upstream draft has no dependency on harness internals.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _default_context_dim(in_dim: int) -> int:
    """Heuristic for auto-inferred ``context_dim`` (PROPOSAL #13)."""
    return max(16, min(in_dim * 4, 64))


# --- Public class -----------------------------------------------------------------


class ConditionalFlow:
    """Bundled conditional normalizing-flow estimator for torchregress.

    One-line construction wraps the backbone MLP + zuko flow +
    :class:`NormalizingFlowLoss` behind a single sklearn-style interface::

        >>> from proposals.conditional_flow import ConditionalFlow
        >>> flow = ConditionalFlow(n_features=1, epochs=60, seed=42)
        >>> flow.fit(X_train, y_train)               # returns self
        >>> mean      = flow.predict(X_test)          # (N, n_features)
        >>> lo, hi    = flow.predict_interval(X_test, confidence=0.9)
        >>> samples   = flow.sample(X_test, n_samples=128)         # (N, 128, n_features)
        >>> log_probs = flow.log_prob(X_test, y_test)               # (N, n_features)
        >>> pdf       = flow.posterior(X_test, grid=grid)        # n_features==1 only
        >>> flow.save("/tmp/cf.pt")
        >>> flow_reloaded = ConditionalFlow.load("/tmp/cf.pt")

    Parameters
    ----------
    n_features : int
        Number of output features (1 for scalar regression).
    context_dim : int or None
        Flow context dimension. If ``None`` (PROPOSAL #13), the value is auto-
        inferred at ``.fit()`` time via ``_default_context_dim(n_features_in_)``.
    flow_type : str
        Forwarded to :func:`create_flow_model`. Currently only ``"nsf"`` is
        wired; other types will raise ``ValueError`` until extended upstream.
    n_transforms : int
        Number of coupling transforms (default 3, matching :doc:`PARITY_REPORT.md`
        §13/§14 sweet spot for the 60-epoch scalar benchmark).
    hidden_features : tuple of int
        Hidden widths of each coupling transform (default (64, 64); same source).
    lr : float
        Adam learning rate for joint backbone + flow training.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    seed : int
        Seed for parameter init, dropout (if any), and per-instance ``torch.Generator``
        used inside the MC-sampling methods.
    """

    def __init__(
        self,
        n_features: int,
        context_dim: int | None = None,
        *,
        flow_type: str = "nsf",
        n_transforms: int = 3,
        hidden_features: tuple[int, ...] = (64, 64),
        lr: float = 1e-3,
        epochs: int = 60,
        batch_size: int = 64,
        seed: int = 42,
    ) -> None:
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1; got {n_features}")
        if flow_type != "nsf":
            raise ValueError(f"this draft only implements flow_type='nsf'; got {flow_type!r}")
        self.n_features = n_features
        self.context_dim = context_dim
        self.flow_type = flow_type
        self.n_transforms = n_transforms
        self.hidden_features = tuple(hidden_features)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = int(seed)

        # sklearn-style introspection (post-fit population)
        self.is_fitted_: bool = False
        self.n_features_in_: int | None = None
        self.feature_names_in_: list[str] | None = None
        self.target_name_: str | None = None

        # Internal state — populated in ``fit()``
        self._backbone: nn.Module | None = None
        self._flow: nn.Module | None = None
        self._loss: nn.Module | None = None
        self._generator: torch.Generator | None = None
        self._resolved_context_dim: int | None = None

    # ----- fit ------------------------------------------------------------------

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        device: str = "cpu",
        feature_names: list[str] | None = None,
        target_name: str | None = None,
    ) -> "ConditionalFlow":
        """Train backbone + zuko flow jointly via Adam + Negative Log-Likelihood.

        Accepts ``X`` / ``y`` as ``np.ndarray`` or array-likes coercible via
        ``np.asarray``.  Sets ``torch.manual_seed(self.seed)`` so repeated fits
        with the same seed are deterministic.  After ``fit`` returns, all
        :ref:`inference methods <sklearn-style-protocol>` are available.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X_arr, y_arr = self._coerce_inputs(X, y)

        # Auto-resolve context_dim if not user-supplied (PROPOSAL #13)
        resolved_c_dim = (
            self.context_dim
            if self.context_dim is not None
            else _default_context_dim(self.n_features_in_)
        )
        self._resolved_context_dim = resolved_c_dim

        self._backbone = _FlowBackbone(
            self.n_features_in_,
            resolved_c_dim,
            resolved_c_dim,
        ).to(device)
        self._flow = create_flow_model(
            n_features=self.n_features,
            context_dim=resolved_c_dim,
            flow_type=self.flow_type,
            n_transforms=self.n_transforms,
            hidden_features=list(self.hidden_features),
        ).to(device)
        self._loss = NormalizingFlowLoss(flow=self._flow).to(device)

        X_t = torch.from_numpy(X_arr).to(device)
        y_t = torch.from_numpy(y_arr).to(device)

        params = list(self._backbone.parameters()) + list(self._flow.parameters())
        opt = optim.Adam(params, lr=self.lr)

        self._backbone.train()
        self._flow.train()

        n = X_t.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                opt.zero_grad(set_to_none=True)
                ctx = self._backbone(X_t[idx])
                # ``NormalizingFlowLoss.forward(y_pred=ctx, target=y)`` returns
                # the per-row NLL reduced by ``self.reduction`` (mean by default).
                loss = self._loss(ctx, y_t[idx])
                loss.backward()
                opt.step()

        # Freeze for inference.
        self._backbone.eval()
        self._flow.eval()

        # Per-instance Generator captured for bit-identical MC re-runs
        # (mirrors `harness/wrappers.py`'s determinism contract — see §3 of PROPOSALS.md).
        self._generator = torch.Generator(device=device).manual_seed(self.seed)

        self.feature_names_in_ = feature_names
        self.target_name_ = target_name
        self.is_fitted_ = True
        return self

    # ----- inference ------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(f"{type(self).__name__} not fitted; call .fit(X, y) first.")

    def _coerce_inputs(self, X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        elif X_arr.ndim != 2:
            raise ValueError(f"X must be 1-D or 2-D; got shape {X_arr.shape}")
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        elif y_arr.ndim != 2:
            raise ValueError(f"y must be 1-D or 2-D; got shape {y_arr.shape}")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"X has {X_arr.shape[0]} rows but y has {y_arr.shape[0]}")
        if y_arr.shape[1] != self.n_features:
            raise ValueError(f"y has {y_arr.shape[1]} columns but n_features={self.n_features}")
        # sklearn-style post-fit introspection on first call.
        if self.n_features_in_ is None:
            self.n_features_in_ = X_arr.shape[1]
        elif X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} columns; expected {self.n_features_in_}")
        return X_arr, y_arr

    def _forward_context(self, X: Any) -> torch.Tensor:
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        elif X_arr.ndim != 2:
            raise ValueError(f"X must be 1-D or 2-D; got shape {X_arr.shape}")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} columns; expected {self.n_features_in_}")
        device = next(self._backbone.parameters()).device
        with torch.no_grad():
            return self._backbone(torch.from_numpy(X_arr).to(device))

    def _coerce_y_for_eval(self, X_ctx_batch_shape: tuple[int, ...], y: Any) -> np.ndarray:
        """Length- and shape-check ``y`` against an existing context batch."""
        n_ctx = X_ctx_batch_shape[0]
        y_arr = np.asarray(y, dtype=np.float32)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        elif y_arr.ndim != 2:
            raise ValueError(f"y must be 1-D or 2-D; got shape {y_arr.shape}")
        if y_arr.shape[1] != self.n_features:
            raise ValueError(f"y has {y_arr.shape[1]} columns but n_features={self.n_features}")
        if y_arr.shape[0] != n_ctx:
            raise ValueError(f"y length {y_arr.shape[0]} != X length {n_ctx}")
        return y_arr

    def _sample_in_fork(self, ctx: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Sample inside ``fork_rng`` so repeated MC draws are deterministic
        regardless of intervening global-RNG consumption."""
        with torch.random.fork_rng(devices=[]):
            if self._generator is not None:
                torch.manual_seed(self._generator.initial_seed())
            samples = self._loss.sample(ctx, n_samples=n_samples)
        return samples

    def sample(self, X: Any, *, n_samples: int = 256) -> np.ndarray:
        """Return MC samples of shape ``(N, n_samples, n_features)``."""
        ctx = self._forward_context(X)
        s = self._sample_in_fork(ctx, n_samples).cpu().numpy()
        # Normalize shapes for downstream consumers.
        if s.ndim == 2:  # (N, n_samples) -> (N, n_samples, 1)
            s = s[..., None]
        elif s.ndim == 3 and s.shape[0] != ctx.shape[0] and s.shape[1] == ctx.shape[0]:
            # some zuko flows emit (n_samples, N, n_features); transpose.
            s = np.transpose(s, (1, 0, 2))
        return s

    def predict(self, X: Any) -> np.ndarray:
        """Predictive mean (MC sample-mean over 256 draws).

        Shape ``(N, n_features)``.  ``NormalizingFlowLoss`` has no closed-form
        mean; we use the sample-mean of n=256 MC draws for stability.
        """
        return self.sample(X, n_samples=256).mean(axis=1)

    def predict_interval(self, X: Any, *, confidence: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)`` central intervals at the given ``confidence``.

        Materialises n=512 MC samples per row, then computes the
        ``alpha/2`` and ``1 - alpha/2`` empirical quantiles along the
        ``n_samples`` axis.  Shape ``(N, n_features)`` for each of ``lower`` and ``upper``.
        """
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"confidence must be in (0, 1); got {confidence}")
        alpha = 1.0 - confidence
        s = self.sample(X, n_samples=512)
        lo = np.quantile(s, alpha / 2.0, axis=1)
        hi = np.quantile(s, 1.0 - alpha / 2.0, axis=1)
        return lo, hi

    def log_prob(self, X: Any, y: Any, *, per_feature: bool = False) -> np.ndarray:
        """Return joint log-probability of ``y`` given ``X``.

        Wraps :meth:`NormalizingFlowLoss.log_prob`.  ``NormalizingFlowLoss``
        already reduces the joint log-likelihood over the ``n_features`` axis,
        so the default return shape is ``(N,)``.  Pass
        ``per_feature=True`` to opt into per-dimension decomposition; the
        total joint log-prob is then ``log_prob(...).sum(axis=-1)`` for
        callers that need the scalar.
        """
        ctx = self._forward_context(X)
        y_arr = self._coerce_y_for_eval(ctx.shape, y)
        device = ctx.device
        y_t = torch.from_numpy(y_arr).to(device)
        with torch.no_grad():
            lp = self._loss.log_prob(ctx, y_t)  # shape (N,) — joint already summed
        out = lp.cpu().numpy()
        if per_feature and self.n_features > 1:
            # If a future NFL change exposes per-feature decompositions,
            # re-compute via ``lp = self._loss.log_prob_per_feature(ctx, y_t)``.
            # For now this branch is unreachable; raise to flag the contract.
            raise NotImplementedError(
                "per_feature=True is reserved for a future NFL release that "
                "exposes per-dimension log-probs; the current NFL returns a "
                "single joint scalar per row."
            )
        return out

    def posterior(
        self,
        X: Any,
        *,
        grid: np.ndarray,
        n_posterior_samples: int = 512,
        bins: int = 64,
    ) -> np.ndarray:
        """1-D approximate conditional PDF evaluated on ``grid``.

        Only valid for ``n_features == 1`` (raises :class:`NotImplementedError`
        otherwise).  Histogram-binning approach: draw ``n_posterior_samples``
        MC samples per test point, build a histogram over ``bins`` equal-width
        bins, then linearly interpolate the histogram onto the user-supplied
        ``grid``.  Returns 0 outside the observed sample range.

        Shape ``(N, len(grid))``.
        """
        if self.n_features != 1:
            raise NotImplementedError(
                "posterior() is currently implemented only for n_features==1; "
                f"got n_features={self.n_features}.  Multi-output posteriors "
                "require per-dimension slices in a follow-up PR."
            )
        grid_arr = np.asarray(grid, dtype=np.float64)
        if grid_arr.ndim != 1:
            raise ValueError(f"grid must be 1-D; got shape {grid_arr.shape}")
        s = self.sample(X, n_samples=n_posterior_samples).squeeze(-1)  # (N, S)
        N = s.shape[0]
        out = np.zeros((N, len(grid_arr)), dtype=np.float64)
        for i in range(N):
            hist, edges = np.histogram(s[i], bins=bins, density=True)
            idx = np.searchsorted(edges, grid_arr, side="right") - 1
            idx = np.clip(idx, 0, len(hist) - 1)
            pdf_vals = hist[idx]
            oob = (grid_arr < edges[0]) | (grid_arr > edges[-1])
            pdf_vals = np.where(oob, 0.0, pdf_vals)
            out[i] = pdf_vals
        return out

    # ----- persistence -----------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist model + metadata to disk.

        Writes a PyTorch ``state_dict`` at ``path`` (``*.pt``) and a JSON sidecar
        at ``path + ".meta.json"`` capturing hyperparameters and sklearn-style
        introspection attrs so a downstream caller can introspect the saved
        model without re-loading.
        """
        self._check_fitted()
        path = Path(path)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        torch.save(
            {"backbone": self._backbone.state_dict(), "flow": self._flow.state_dict()},
            path,
        )
        meta = {
            "n_features": self.n_features,
            "context_dim": self.context_dim,
            "resolved_context_dim": self._resolved_context_dim,
            "flow_type": self.flow_type,
            "n_transforms": self.n_transforms,
            "hidden_features": list(self.hidden_features),
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
            # sklearn-style introspection (None when caller never set it).
            "n_features_in_": self.n_features_in_,
            "feature_names_in_": self.feature_names_in_,
            "target_name_": self.target_name_,
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "ConditionalFlow":
        """Reload model + metadata from disk (inverse of :meth:`save`)."""
        path = Path(path)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta = json.loads(meta_path.read_text())
        inst = cls(
            n_features=meta["n_features"],
            context_dim=meta["context_dim"],
            flow_type=meta["flow_type"],
            n_transforms=meta["n_transforms"],
            hidden_features=tuple(meta["hidden_features"]),
            lr=meta["lr"],
            epochs=meta["epochs"],
            batch_size=meta["batch_size"],
            seed=meta["seed"],
        )
        inst.n_features_in_ = meta["n_features_in_"]
        inst.feature_names_in_ = meta["feature_names_in_"]
        inst.target_name_ = meta["target_name_"]
        inst._resolved_context_dim = meta["resolved_context_dim"]

        c_dim = inst._resolved_context_dim
        inst._backbone = _FlowBackbone(inst.n_features_in_, c_dim, c_dim)
        inst._flow = create_flow_model(
            n_features=inst.n_features,
            context_dim=c_dim,
            flow_type=inst.flow_type,
            n_transforms=inst.n_transforms,
            hidden_features=list(inst.hidden_features),
        )
        inst._loss = NormalizingFlowLoss(flow=inst._flow)
        states = torch.load(path, map_location="cpu")
        inst._backbone.load_state_dict(states["backbone"])
        inst._flow.load_state_dict(states["flow"])
        inst._backbone.eval()
        inst._flow.eval()
        inst._generator = torch.Generator().manual_seed(inst.seed)
        inst.is_fitted_ = True
        return inst


# --- quick self-test --------------------------------------------------------------


def _demo() -> None:  # pragma: no cover
    """Sanity smoke: 50-sample, 3-feature linear regression at 2 epochs.

    Confirms the class is instantiable, trainable, and that every public method
    returns the documented shape.  Not a substitute for ``tests/test_wrappers.py``.
    """
    rng = np.random.default_rng(7)
    X = rng.standard_normal((50, 3)).astype(np.float32)
    y = (
        (X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2] + 0.1 * rng.standard_normal(50))
        .astype(np.float32)
        .reshape(-1, 1)
    )

    flow = ConditionalFlow(
        n_features=1,
        epochs=2,
        batch_size=16,
        seed=42,
    ).fit(X, y)
    assert flow.is_fitted_
    assert flow.n_features_in_ == 3

    pred = flow.predict(X)
    assert pred.shape == (50, 1), f"predict shape {pred.shape}"

    lo, hi = flow.predict_interval(X, confidence=0.9)
    assert lo.shape == hi.shape == (50, 1), f"interval shape {lo.shape}/{hi.shape}"

    s = flow.sample(X, n_samples=4)
    assert s.shape == (50, 4, 1), f"sample shape {s.shape}"

    lp = flow.log_prob(X, y)
    assert lp.shape == (50,), f"log_prob shape {lp.shape}"

    grid = np.linspace(-3.0, 3.0, 25, dtype=np.float32)
    pdf = flow.posterior(X[:3], grid=grid)
    assert pdf.shape == (3, 25), f"posterior shape {pdf.shape}"

    # save / load round-trip
    tmp_path = Path("/tmp/_conditional_flow_demo.pt")
    flow.save(tmp_path)
    reloaded = ConditionalFlow.load(tmp_path)
    pred_reloaded = reloaded.predict(X[:5])
    assert pred_reloaded.shape == (5, 1)
    tmp_path.unlink(missing_ok=True)

    print(
        f"[conditional_flow._demo] OK: is_fitted={flow.is_fitted_}, "
        f"predict={pred.shape}, "
        f"interval={lo.shape}x{hi.shape}, "
        f"samples={s.shape}, "
        f"log_prob={lp.shape}, "
        f"posterior={pdf.shape}"
    )


if __name__ == "__main__":  # pragma: no cover
    _demo()
