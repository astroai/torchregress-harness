"""Synthetic likelihood-family designs for the ``likelihoods`` suite.

Each design draws ``y | x`` from one torchregress flexible-shape family
(skew-normal, skew-t, Johnson SU, sinh-arcsinh, GEV, beta regression), with
raw parameters linear in three features (location and scale vary with ``x``,
shape parameters are constant) and the raw -> parameter maps the library uses
(softplus + 1e-6 for positive parameters, sigmoid for the beta mean).  The
family is therefore the true model and a linear head is well specified.

Rows per design (``Dataset`` = ``lik_<family>``):

* ``TrueModel`` - NLL of the test targets under the true parameters, from
  ``scipy.stats``; ``LibNLLAtTruth`` is the torchregress functional NLL
  (``skew_normal_nll`` ...) at the same parameters and ``TruthNLLParity`` the
  absolute difference (a direct density cross-check, ~1e-6 expected).
* ``tr_Linear`` - the torchregress loss class on a linear head, full-batch
  L-BFGS in float64 (the conditional MLE).
* ``scipy_MLE_Linear`` - the same linear model fitted by
  ``scipy.optimize.minimize`` on the ``scipy.stats`` log-density: the
  reference for ``tr_Linear`` (``NLLGapToScipyLinear`` should be ~0).
* ``tr_MLP`` - the torchregress loss on the harness MLP (Adam, config epochs).
* ``tr_GaussianNLL_MLP`` - Gaussian NLL on the same MLP: the misspecified
  baseline the family must beat.
* ``scipy_MLE_Unconditional`` - the ``scipy.stats`` distribution fitted by
  MLE to the training targets alone (``dist.fit``; skew-t and sinh-arcsinh,
  which scipy.stats lacks, are built from ``scipy.stats.t`` / ``norm`` and
  fitted with ``scipy.optimize``).

Metrics: ``NLL`` (test mean, over rows with a finite NLL), ``FracInfNLL``
(test targets outside a predicted GEV support get an infinite NLL; reported,
not averaged), ``NLLGapToTruth`` and parameter recovery against the true
per-row parameters: ``LocRMSE`` (location; beta mean), ``ScaleRMSE`` (scale;
beta precision), ``Shape1MAE`` / ``Shape2MAE`` (shape parameters, in the order
of ``FAMILIES[f].params``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch
from scipy import optimize, special, stats
from torch import nn

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig
from suites._backbones import _MLP, _train_one

N_FEATURES = 3
N_TRAIN, N_TEST = 2000, 2000
N_TRAIN_SMOKE, N_TEST_SMOKE = 400, 400
_EPS = 1e-6  # the library's positive-parameter floor (softplus(raw) + eps)
_BARRIER = 1e10  # objective value for infeasible parameters (target outside the support)


def softplus(r: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, r) + _EPS


def inv_softplus(v: float) -> float:
    s = v - _EPS
    return float(s + math.log(-math.expm1(-s)))


# ---------------------------------------------------------------------------
# scipy log-densities on constrained parameters (rows of P = [N, k])
# ---------------------------------------------------------------------------


def _lp_skewnormal(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return stats.skewnorm.logpdf(y, p[:, 2], loc=p[:, 0], scale=p[:, 1])


def _lp_skewt(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    # Azzalini & Capitanio (2003) skew-t; scipy.stats has only Jones-Faddy's.
    xi, om, al, nu = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    z = (y - xi) / om
    w = al * z * np.sqrt((nu + 1.0) / (nu + z * z))
    return math.log(2.0) + stats.t.logpdf(z, nu) - np.log(om) + stats.t.logcdf(w, nu + 1.0)


def _lp_johnsonsu(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return stats.johnsonsu.logpdf(y, p[:, 2], p[:, 3], loc=p[:, 0], scale=p[:, 1])


def _lp_sinharcsinh(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    # Jones & Pewsey (2009): y = mu + sigma * sinh(delta * asinh(Z) + eps), Z ~ N(0, 1).
    mu, sig, eps, delta = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    z = (y - mu) / sig
    w = (np.arcsinh(z) - eps) / delta
    return (
        stats.norm.logpdf(np.sinh(w))
        + np.log(np.cosh(w))
        - np.log(delta)
        - 0.5 * np.log1p(z * z)
        - np.log(sig)
    )


def _lp_gev(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    # scipy's genextreme shape c is minus the Coles shape xi.
    return stats.genextreme.logpdf(y, -p[:, 2], loc=p[:, 0], scale=p[:, 1])


def _lp_beta(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    mu, phi = p[:, 0], p[:, 1]
    return stats.beta.logpdf(y, mu * phi, (1.0 - mu) * phi)


# ---------------------------------------------------------------------------
# samplers (true model)
# ---------------------------------------------------------------------------


def _draw_skewnormal(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return stats.skewnorm.rvs(p[:, 2], loc=p[:, 0], scale=p[:, 1], random_state=rng)


def _draw_skewt(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sn = stats.skewnorm.rvs(p[:, 2], random_state=rng)
    v = rng.chisquare(p[:, 3])
    return p[:, 0] + p[:, 1] * sn / np.sqrt(v / p[:, 3])


def _draw_johnsonsu(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return stats.johnsonsu.rvs(p[:, 2], p[:, 3], loc=p[:, 0], scale=p[:, 1], random_state=rng)


def _draw_sinharcsinh(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    z = rng.standard_normal(len(p))
    return p[:, 0] + p[:, 1] * np.sinh(p[:, 3] * np.arcsinh(z) + p[:, 2])


def _draw_gev(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return stats.genextreme.rvs(-p[:, 2], loc=p[:, 0], scale=p[:, 1], random_state=rng)


def _draw_beta(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = rng.beta(p[:, 0] * p[:, 1], (1.0 - p[:, 0]) * p[:, 1])
    return np.clip(y, 1e-6, 1.0 - 1e-6)  # the library requires 0 < y < 1


# ---------------------------------------------------------------------------
# family registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Family:
    """One flexible-shape family: torchregress names, scipy density, truth."""

    loss_class: str  # torchregress.losses class used for training
    functional: str  # torchregress.losses functional used for test NLL
    params: tuple[str, ...]  # constrained parameter names, in raw-column order
    positive: tuple[int, ...]  # raw columns mapped through softplus + eps
    logpdf: Callable[[np.ndarray, np.ndarray], np.ndarray]
    draw: Callable[[np.ndarray, np.random.Generator], np.ndarray]
    shapes: tuple[float, ...]  # true constant shape parameters (constrained)
    scipy_fit: Callable[[np.ndarray], np.ndarray] | None  # unconditional dist.fit -> params
    sigmoid_loc: bool = False  # beta: raw column 0 is a logit

    @property
    def k(self) -> int:
        return len(self.params)

    def constrain(self, raw: np.ndarray) -> np.ndarray:
        p = np.array(raw, dtype=np.float64, copy=True)
        for j in self.positive:
            p[:, j] = softplus(p[:, j])
        if self.sigmoid_loc:
            p[:, 0] = special.expit(p[:, 0])
        return p

    def raw_shapes(self) -> list[float]:
        out = []
        for j, v in enumerate(self.shapes, start=self.k - len(self.shapes)):
            out.append(inv_softplus(v) if j in self.positive else float(v))
        return out


def _fit_skewnorm(y: np.ndarray) -> np.ndarray:
    a, loc, scale = stats.skewnorm.fit(y)
    return np.array([loc, scale, a])


def _fit_johnsonsu(y: np.ndarray) -> np.ndarray:
    a, b, loc, scale = stats.johnsonsu.fit(y)
    return np.array([loc, scale, a, b])


def _fit_gev(y: np.ndarray) -> np.ndarray:
    c, loc, scale = stats.genextreme.fit(y)
    return np.array([loc, scale, -c])


def _fit_beta(y: np.ndarray) -> np.ndarray:
    a, b, _, _ = stats.beta.fit(y, floc=0.0, fscale=1.0)
    return np.array([a / (a + b), a + b])


FAMILIES: dict[str, Family] = {
    "skewnormal": Family(
        "SkewNormalNLLLoss",
        "skew_normal_nll",
        ("xi", "omega", "alpha"),
        (1,),
        _lp_skewnormal,
        _draw_skewnormal,
        (4.0,),
        _fit_skewnorm,
    ),
    "skewt": Family(
        "SkewTLoss",
        "skew_t_nll",
        ("xi", "omega", "alpha", "nu"),
        (1, 3),
        _lp_skewt,
        _draw_skewt,
        (3.0, 5.0),
        None,
    ),
    "johnsonsu": Family(
        "JohnsonSUNLLLoss",
        "johnson_su_nll",
        ("xi", "lambda", "gamma", "delta"),
        (1, 3),
        _lp_johnsonsu,
        _draw_johnsonsu,
        (-1.0, 1.5),
        _fit_johnsonsu,
    ),
    "sinharcsinh": Family(
        "SinhArcsinhNLLLoss",
        "sinh_arcsinh_nll",
        ("mu", "sigma", "epsilon", "delta"),
        (1, 3),
        _lp_sinharcsinh,
        _draw_sinharcsinh,
        (0.8, 0.7),
        None,
    ),
    "gev": Family(
        "GEVNLLLoss",
        "gev_nll",
        ("mu", "sigma", "xi"),
        (1,),
        _lp_gev,
        _draw_gev,
        (0.1,),
        _fit_gev,
    ),
    "beta": Family(
        "BetaRegressionNLLLoss",
        "beta_regression_nll",
        ("mu", "phi"),
        (1,),
        _lp_beta,
        _draw_beta,
        (),
        _fit_beta,
        sigmoid_loc=True,
    ),
}

DESIGNS: dict[str, str] = {f"lik_{name}": name for name in FAMILIES}

METHODS = (
    "TrueModel",
    "tr_Linear",
    "scipy_MLE_Linear",
    "tr_MLP",
    "tr_GaussianNLL_MLP",
    "scipy_MLE_Unconditional",
)


def true_raw(family: Family, x: np.ndarray) -> np.ndarray:
    """True raw head output [N, k]: location/scale linear in x, shapes constant."""
    raw = np.zeros((len(x), family.k))
    if family.sigmoid_loc:
        raw[:, 0] = 0.3 + 1.2 * x[:, 0] - 0.8 * x[:, 1]
        raw[:, 1] = 3.0 + 1.0 * x[:, 2]
    else:
        raw[:, 0] = 1.0 + 1.5 * x[:, 0] - 1.0 * x[:, 1]
        raw[:, 1] = -0.3 + 0.6 * x[:, 2]
    for j, v in enumerate(family.raw_shapes(), start=family.k - len(family.shapes)):
        raw[:, j] = v
    return raw


def draw_design(name: str, seed: int, *, smoke: bool) -> dict[str, np.ndarray]:
    family = FAMILIES[DESIGNS[name]]
    n_tr, n_te = (N_TRAIN_SMOKE, N_TEST_SMOKE) if smoke else (N_TRAIN, N_TEST)
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n_tr + n_te, N_FEATURES))
    raw = true_raw(family, x)
    y = family.draw(family.constrain(raw), rng)
    return {
        "X_train": x[:n_tr],
        "y_train": y[:n_tr],
        "X_test": x[n_tr:],
        "y_test": y[n_tr:],
        "raw_test": raw[n_tr:],
    }


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------


def initial_raw(family: Family, y: np.ndarray) -> np.ndarray:
    """Moment-based starting intercepts shared by the torch and scipy fits."""
    b = np.zeros(family.k)
    if family.sigmoid_loc:
        m, v = float(np.mean(y)), float(np.var(y))
        b[0] = float(special.logit(m))
        b[1] = inv_softplus(max(m * (1.0 - m) / max(v, 1e-6) - 1.0, 0.5))
        return b
    b[0] = float(np.mean(y))
    b[1] = inv_softplus(float(np.std(y)))
    # Skew-t starts at alpha = 0.1, not 0: torchregress SkewTLoss has an exactly
    # zero gradient in alpha at alpha = 0 (the Student-t CDF term clamps its
    # incomplete-beta argument at 1 - 1e-15 when w = 0), so a symmetric start
    # never learns skewness. Both fits share the start.
    neutral = {"skewnormal": (0.0,), "skewt": (0.1, 10.0), "johnsonsu": (0.0, 1.0)}
    neutral |= {"sinharcsinh": (0.0, 1.0), "gev": (0.01,)}
    name = next(n for n, f in FAMILIES.items() if f is family)
    for j, v in enumerate(neutral[name], start=family.k - len(family.shapes)):
        b[j] = inv_softplus(v) if j in family.positive else v
    return b


def _mean_nll(family: Family, y: np.ndarray, raw: np.ndarray) -> float:
    with np.errstate(all="ignore"):
        lp = family.logpdf(y, family.constrain(raw))
    val = -float(np.mean(lp))
    return val if np.isfinite(val) else _BARRIER


def scipy_fit_linear(
    family: Family, x: np.ndarray, y: np.ndarray, *, intercept_only: bool, maxiter: int
) -> np.ndarray:
    """MLE of the linear raw-parameter model with scipy.optimize (returns [k, d+1])."""
    d = 0 if intercept_only else x.shape[1]
    theta0 = np.zeros((family.k, d + 1))
    theta0[:, -1] = initial_raw(family, y)

    def raw_of(theta: np.ndarray) -> np.ndarray:
        return _apply_linear(theta.reshape(family.k, d + 1), x)

    res = optimize.minimize(
        lambda th: _mean_nll(family, y, raw_of(th)),
        theta0.ravel(),
        method="L-BFGS-B",
        options={"maxiter": maxiter},
    )
    return res.x.reshape(family.k, d + 1)


def _apply_linear(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Raw head output [N, k] of a linear model ``theta = [W | b]`` ([k, d+1])."""
    d = theta.shape[1] - 1
    return x[:, :d] @ theta[:, :d].T + theta[:, -1]


def torch_fit_linear(family: Family, x: np.ndarray, y: np.ndarray, *, max_iter: int) -> np.ndarray:
    """Library loss on a float64 linear head, full-batch L-BFGS (returns [k, d+1])."""
    import torchregress.losses as trl

    head = nn.Linear(x.shape[1], family.k).double()
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.as_tensor(initial_raw(family, y)))
    loss = getattr(trl, family.loss_class)()
    xt = torch.as_tensor(x, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.float64).reshape(-1, 1)
    opt = torch.optim.LBFGS(
        head.parameters(),
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        opt.zero_grad()
        value = loss(head(xt), yt)
        if not torch.isfinite(value):
            # A trial step put a training target outside the GEV support
            # (infinite NLL): the same finite barrier as the scipy fit, so the
            # strong-Wolfe line search backtracks instead of stalling on inf.
            return torch.tensor(_BARRIER, dtype=torch.float64)
        value.backward()
        return value

    head.train()
    opt.step(closure)
    w = head.weight.detach().numpy()
    b = head.bias.detach().numpy()[:, None]
    return np.concatenate([w, b], axis=1)


def torch_fit_mlp(
    loss_class: str, k: int, x: np.ndarray, y: np.ndarray, cfg: HarnessConfig
) -> nn.Module:
    import torchregress.losses as trl

    model = _MLP(x.shape[1], 64, k)
    _train_one(
        model,
        getattr(trl, loss_class)(),
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1),
        cfg,
    )
    model.eval()
    return model


def _mlp_raw(model: nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        out = model(torch.as_tensor(x, dtype=torch.float32).to(current_device()))
    return out.cpu().numpy().astype(np.float64)


def library_nll_rows(family: Family, raw: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-row test NLL from the torchregress functional (float64)."""
    import torchregress.losses as trl

    fn = getattr(trl, family.functional)
    out = fn(
        torch.as_tensor(raw, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        reduction="none",
    )
    return out.detach().numpy().reshape(-1)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def nll_summary(nll_rows: np.ndarray) -> dict[str, float]:
    """Mean NLL over finite rows plus the fraction of infinite / NaN rows."""
    finite = np.isfinite(nll_rows)
    return {
        "NLL": float(np.mean(nll_rows[finite])) if finite.any() else float("nan"),
        "FracInfNLL": float(1.0 - finite.mean()),
    }


def param_recovery(family: Family, raw_pred: np.ndarray, raw_true: np.ndarray) -> dict[str, float]:
    p, t = family.constrain(raw_pred), family.constrain(raw_true)
    out = {
        "LocRMSE": float(np.sqrt(np.mean((p[:, 0] - t[:, 0]) ** 2))),
        "ScaleRMSE": float(np.sqrt(np.mean((p[:, 1] - t[:, 1]) ** 2))),
    }
    for i, j in enumerate(range(2, family.k), start=1):
        out[f"Shape{i}MAE"] = float(np.mean(np.abs(p[:, j] - t[:, j])))
    return out


def _gaussian_nll_rows(raw: np.ndarray, y: np.ndarray) -> np.ndarray:
    # GaussianNLLLoss head layout: [mean, log variance].
    mean, logvar = raw[:, 0], raw[:, 1]
    return 0.5 * (math.log(2 * math.pi) + logvar + (y - mean) ** 2 / np.exp(logvar))


# ---------------------------------------------------------------------------
# one design -> rows
# ---------------------------------------------------------------------------

METHOD_NOTES = {
    "TrueModel": "scipy.stats NLL at the true parameters; TruthNLLParity = |library - scipy|",
    "tr_Linear": "torchregress loss on a linear head, float64 L-BFGS (conditional MLE)",
    "scipy_MLE_Linear": "same linear model, scipy.stats log-density + scipy.optimize L-BFGS-B",
    "tr_MLP": "torchregress loss on the harness MLP (Adam)",
    "tr_GaussianNLL_MLP": "GaussianNLLLoss on the same MLP (misspecified baseline)",
    "scipy_MLE_Unconditional": "scipy.stats distribution fitted by MLE to y_train (ignores x)",
}


def method_library(method: str) -> str:
    return "scipy" if method.startswith("scipy") or method == "TrueModel" else "torchregress"


def run_design(dataset_name: str, cfg: HarnessConfig, suite_name: str) -> list[dict[str, Any]]:
    """All rows of one ``lik_<family>`` design."""
    family = FAMILIES[DESIGNS[dataset_name]]
    data = draw_design(dataset_name, cfg.seed, smoke=cfg.smoke)
    x_tr, y_tr, x_te, y_te = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    raw_true = data["raw_test"]
    truth_nll = -float(np.mean(family.logpdf(y_te, family.constrain(raw_true))))
    maxiter = 100 if cfg.smoke else 500
    scipy_linear_nll: float | None = None
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        base: dict[str, Any] = {
            "Method": f"{suite_name}/{method}",
            "Library": method_library(method),
            "LibraryVersion": _version(method_library(method)),
            "Dataset": dataset_name,
            "Suite": suite_name,
            "SourceURL": "",
            "SyntheticFallback": False,  # synthetic by design: known true model
            "Device": str(current_device()),
        }
        try:
            set_harness_seed(cfg.seed)
            t0 = perf_counter()
            metrics: dict[str, float] = {}
            if method == "TrueModel":
                lib = library_nll_rows(family, raw_true, y_te)
                metrics = {
                    "NLL": truth_nll,
                    "FracInfNLL": 0.0,
                    "LibNLLAtTruth": float(np.mean(lib)),
                    "TruthNLLParity": abs(float(np.mean(lib)) - truth_nll),
                }
            elif method in ("tr_Linear", "scipy_MLE_Linear"):
                if method == "tr_Linear":
                    theta = torch_fit_linear(family, x_tr, y_tr, max_iter=maxiter)
                else:
                    theta = scipy_fit_linear(
                        family, x_tr, y_tr, intercept_only=False, maxiter=maxiter
                    )
                raw = _apply_linear(theta, x_te)
                nll_rows = (
                    library_nll_rows(family, raw, y_te)
                    if method == "tr_Linear"
                    else -family.logpdf(y_te, family.constrain(raw))
                )
                metrics = {**nll_summary(nll_rows), **param_recovery(family, raw, raw_true)}
                if method == "scipy_MLE_Linear":
                    scipy_linear_nll = metrics["NLL"]
            elif method == "tr_MLP":
                model = torch_fit_mlp(family.loss_class, family.k, x_tr, y_tr, cfg)
                raw = _mlp_raw(model, x_te)
                metrics = {
                    **nll_summary(library_nll_rows(family, raw, y_te)),
                    **param_recovery(family, raw, raw_true),
                }
            elif method == "tr_GaussianNLL_MLP":
                model = torch_fit_mlp("GaussianNLLLoss", 2, x_tr, y_tr, cfg)
                metrics = nll_summary(_gaussian_nll_rows(_mlp_raw(model, x_te), y_te))
            elif method == "scipy_MLE_Unconditional":
                if family.scipy_fit is not None:
                    params = family.scipy_fit(y_tr)
                else:
                    theta = scipy_fit_linear(
                        family, x_tr, y_tr, intercept_only=True, maxiter=maxiter
                    )
                    params = family.constrain(theta[:, -1][None, :])[0]
                p = np.broadcast_to(params, (len(y_te), family.k))
                metrics = nll_summary(-family.logpdf(y_te, np.asarray(p)))
            else:
                raise ValueError(f"Unknown method: {method}")
            elapsed = perf_counter() - t0
            metrics["NLLGapToTruth"] = metrics["NLL"] - truth_nll
            rows.append(
                {
                    **base,
                    **metrics,
                    "TrueNLL": truth_nll,
                    "train_s": elapsed,
                    "eval_s": 0.0,
                    "Notes": METHOD_NOTES[method],
                }
            )
        except Exception as exc:  # noqa: BLE001 — one method failing must not drop the rest
            rows.append({**base, "Notes": f"failed: {type(exc).__name__}: {exc}"})
    # The linear torchregress fit and the scipy fit target the same MLE.
    for row in rows:
        if row["Method"].endswith("/tr_Linear") and scipy_linear_nll is not None and "NLL" in row:
            row["NLLGapToScipyLinear"] = row["NLL"] - scipy_linear_nll
    return rows


def _version(dist: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return "not installed"
