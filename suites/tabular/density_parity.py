"""Density-estimator parity on real data: torchregress vs sbi vs pzflow.

Compares conditional density estimators ``p(y | x)`` on real tabular
regression data, including multi-target sets, where no reference posterior
exists.  Unlike ``sbi_parity`` (sbibm toy simulators scored by C2ST against
reference posteriors), this suite scores held-out real targets with proper
scoring rules and calibration diagnostics.

Paired rows share an estimator family and an identical feature encoder
(the harness ``_MLP``, passed to sbi as ``embedding_net``) and are trained by
the *same* loop (Adam, mini-batch, early stopping on a 10% validation split,
best weights restored), so differences come from the density estimator
implementation alone:

* NSF: ``torchregress_NSF`` (zuko) vs ``sbi_NSF`` (sbi default, nflows) vs
  ``sbi_zuko_NSF`` (sbi wrapper of zuko) vs ``pzflow_NSF`` (JAX, own trainer)
* MAF: ``torchregress_MAF`` (zuko) vs ``sbi_MAF`` (nflows) vs ``sbi_zuko_MAF``
  (sbi wrapper of zuko: same backend, so gaps come from the wrapper)
* MDN, full covariance, K=10: ``torchregress_MDN`` vs ``sbi_MDN``

``sbi_NPE_NSF`` is a reference row: sbi's own NPE trainer and defaults
(``append_simulations(theta=y, x=X)``), i.e. sbi used out of the box.

Targets are standardised per dimension on the training split and every
metric is reported in those units (comparable across methods and datasets).
Metrics: joint NLL; energy score (ES); variogram score (VS, p=0.5, sensitive
to the dependence structure); mean marginal CRPS; mean marginal 90% coverage;
joint 90% HPD coverage (truth inside the density's 90% highest-density
region, from the log-density of samples); mean marginal PIT KS; and CorrErr,
the mean absolute gap between the empirical residual correlation and the
average predicted conditional correlation (multi-target only).
"""

from __future__ import annotations

import copy
import os
from time import perf_counter
from typing import Any

import numpy as np
import torch

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _MLP, _hidden_size

N_EVAL_SAMPLES = 200
MAX_TEST_ROWS = 1000
FLOW_HIDDEN = 64
FLOW_TRANSFORMS = 3
MDN_COMPONENTS = 10
VAL_FRACTION = 0.1
PATIENCE = 10
# Early stopping decides the length; the cap only bounds runaway runs.
MAX_EPOCHS = 300
N_DUMP_ROWS = 5
# Set to a directory to save predicted samples of the first N_DUMP_ROWS test rows
# per method (tools/plot_density_parity.py draws them as corner plots).
DUMP_ENV = "DENSITY_PARITY_DUMP_DIR"

_TR_FLOWS = {
    "torchregress_NSF": "nsf",
    "torchregress_MAF": "maf",
    "torchregress_cINN": "realnvp",
    "torchregress_BPF": "bpf",
}
_SBI_MODELS = {
    "sbi_NSF": "nsf",
    "sbi_MAF": "maf",
    "sbi_MDN": "mdn",
    "sbi_zuko_NSF": "zuko_nsf",
    "sbi_zuko_MAF": "zuko_maf",
}
_METRICS = (
    "NLL",
    "ES",
    "VS",
    "CRPS",
    "Cov90_marg",
    "Cov90_joint",
    "PIT_KS",
    "CorrErr",
    "Params",
)


# ---------------------------------------------------------------------------
# Shared training loop
# ---------------------------------------------------------------------------


def _fit_shared(
    modules: list[torch.nn.Module],
    nll: Any,
    X: np.ndarray,
    Y: np.ndarray,
    cfg: HarnessConfig,
) -> int:
    """Adam + early stopping on a validation split; restores the best weights.

    ``nll(xb, yb)`` returns per-row negative log-likelihoods.  Returns the
    number of epochs run.
    """
    dev = current_device()
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(X))
    n_val = max(1, int(VAL_FRACTION * len(X)))
    va, tr = perm[:n_val], perm[n_val:]
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=dev)
    params = [p for m in modules for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=cfg.lr)
    best, best_state, bad, epoch = float("inf"), None, 0, 0
    batch = int(cfg.batch_size)
    for epoch in range(1, (2 if cfg.smoke else MAX_EPOCHS) + 1):
        for m in modules:
            m.train()
        order = tr[rng.permutation(len(tr))]
        for start in range(0, len(order), batch):
            idx = torch.as_tensor(order[start : start + batch], device=dev)
            loss = nll(Xt[idx], Yt[idx]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
        for m in modules:
            m.eval()
        with torch.no_grad():
            vidx = torch.as_tensor(va, device=dev)
            v = float(nll(Xt[vidx], Yt[vidx]).mean())
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = [copy.deepcopy(m.state_dict()) for m in modules]
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        for m, st in zip(modules, best_state, strict=True):
            m.load_state_dict(st)
    for m in modules:
        m.eval()
    return epoch


# ---------------------------------------------------------------------------
# Scores (all in standardised target units)
# ---------------------------------------------------------------------------


def _pairwise_mean_dist(S: torch.Tensor, p: float = 1.0) -> torch.Tensor:
    """E||X - X'||^p over sample pairs, per row: S is [N, S, D] -> [N]."""
    return torch.cdist(S, S).pow(p).mean(dim=(1, 2))


def _scores(
    y: np.ndarray, samples: np.ndarray, lp_true: np.ndarray, lp_samples: np.ndarray | None
) -> dict[str, float | None]:
    """Proper scores and calibration from [N, D] truths and [N, S, D] samples."""
    Y = torch.as_tensor(y, dtype=torch.float64)
    S = torch.as_tensor(samples, dtype=torch.float64)
    N, n_s, D = S.shape
    # Row chunks keep the [rows, S, S, D] pairwise terms bounded (rf1/scm20d OOM'd).
    es_parts, crps_parts = [], []
    for start in range(0, N, 50):
        Sb, Yb = S[start : start + 50], Y[start : start + 50]
        es_parts.append((Sb - Yb[:, None, :]).norm(dim=-1).mean(1) - 0.5 * _pairwise_mean_dist(Sb))
        spread = (Sb[:, :, None, :] - Sb[:, None, :, :]).abs().mean((1, 2))
        crps_parts.append((Sb - Yb[:, None, :]).abs().mean(1) - 0.5 * spread)
    es = torch.cat(es_parts)
    crps = torch.cat(crps_parts)
    has_lp = lp_true is not None and bool(np.all(np.isfinite(lp_true)))
    out: dict[str, float | None] = {
        "NLL": float(-np.mean(lp_true)) if has_lp else None,
        "ES": float(es.mean()),
        "CRPS": float(crps.mean()),
    }
    if D > 1:
        iu = torch.triu_indices(D, D, offset=1)
        dy = (Y[:, iu[0]] - Y[:, iu[1]]).abs().sqrt()
        ds = (S[:, :, iu[0]] - S[:, :, iu[1]]).abs().sqrt().mean(1)
        out["VS"] = float(((dy - ds) ** 2).sum(1).mean())
        resid = (Y - S.mean(1)).numpy()
        r_emp = np.corrcoef(resid.T)
        Sc = S - S.mean(1, keepdim=True)
        cov = Sc.transpose(1, 2) @ Sc / (n_s - 1)
        sd = cov.diagonal(dim1=1, dim2=2).clamp_min(1e-12).sqrt()
        r_pred = (cov / (sd[:, :, None] * sd[:, None, :])).mean(0).numpy()
        mask = ~np.eye(D, dtype=bool)
        out["CorrErr"] = float(np.abs(r_emp - r_pred)[mask].mean())
    else:
        out["VS"] = None
        out["CorrErr"] = None
    lo = torch.quantile(S, 0.05, dim=1)
    hi = torch.quantile(S, 0.95, dim=1)
    out["Cov90_marg"] = float(((Y >= lo) & (Y <= hi)).double().mean())
    pit = (S <= Y[:, None, :]).double().mean(1).numpy()  # [N, D]
    grid = np.sort(pit, axis=0)
    ecdf = np.arange(1, N + 1)[:, None] / N
    out["PIT_KS"] = float(np.max(np.abs(grid - ecdf), axis=0).mean())
    if lp_samples is not None and has_lp:
        thr = np.quantile(lp_samples, 0.10, axis=1)
        out["Cov90_joint"] = float(np.mean(lp_true >= thr))
    else:
        out["Cov90_joint"] = None
    return out


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class DensityParitySuite(HarnessRunner):
    """Conditional density estimators on real data (custom run)."""

    _image_channels: int | None = None  # set per dataset in run()

    def _encoder(self, in_dim: int, hidden: int) -> torch.nn.Module:
        """Shared feature encoder: MLP for tables, the harness CNN for images."""
        if self._image_channels is not None:
            from suites.image._base import _ImageCNN

            return _ImageCNN(in_channels=self._image_channels, out_dim=hidden)
        return _MLP(in_dim=in_dim, hidden=hidden, out_dim=hidden)

    @property
    def name(self) -> str:
        return "density_parity"

    @property
    def description(self) -> str:
        return (
            "Conditional density estimators on real (multi-target) data: torchregress "
            "NSF/MAF/MDN vs sbi NSF/MAF/MDN vs pzflow NSF, shared encoder and trainer"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "mt_enb",
            "mt_jura",
            "mt_rf1",
            "mt_scm20d",
            "energy_efficiency",
            "california_housing",
            "pzflow_galaxies",
        ]

    def methods(self) -> list[str]:
        requested = os.environ.get("TORCHREGRESS_DENSITY_METHODS", "").strip()
        if requested:
            wanted = {part.strip() for part in requested.split(",") if part.strip()}
            return [
                m
                for m in (
                    "torchregress_NSF",
                    "sbi_NSF",
                    "sbi_zuko_NSF",
                    "pzflow_NSF",
                    "torchregress_MAF",
                    "torchregress_cINN",
                    "torchregress_BPF",
                    "torchregress_CFM",
                    "sbi_MAF",
                    "sbi_zuko_MAF",
                    "torchregress_MDN",
                    "sbi_MDN",
                    "sbi_NPE_NSF",
                )
                if m in wanted
            ]
        return [
            "torchregress_NSF",
            "sbi_NSF",
            "sbi_zuko_NSF",
            "pzflow_NSF",
            "torchregress_MAF",
            "torchregress_cINN",
            "torchregress_BPF",
            "torchregress_CFM",
            "torchregress_NFM",
            "sbi_MAF",
            "sbi_zuko_MAF",
            "torchregress_MDN",
            "sbi_MDN",
            "sbi_NPE_NSF",
        ]

    def library_name(self, method: str) -> str:
        return method.split("_", 1)[0]

    def library_version(self, method: str) -> str:
        import importlib.metadata as md

        pkg = {"torchregress": "torchregress", "sbi": "sbi", "pzflow": "pzflow"}[
            self.library_name(method)
        ]
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return ""

    def method_notes(self, method: str) -> str:
        return {
            "torchregress_NSF": "zuko NSF (3 transforms, 64 hidden) on shared encoder; shared trainer",
            "sbi_NSF": "sbi posterior_nn('nsf') = nflows NSF (3 transforms, 64 hidden), shared encoder as embedding_net; shared trainer",
            "sbi_zuko_NSF": "sbi posterior_nn('zuko_nsf') (3 transforms, 64 hidden), shared encoder as embedding_net; shared trainer",
            "pzflow_NSF": "pzflow Flow (default bijector), conditional columns = x; pzflow's own trainer",
            "torchregress_MAF": "zuko MAF (3 transforms, 64 hidden) on shared encoder; shared trainer",
            "torchregress_cINN": "zuko conditional affine coupling / cINN-style RealNVP on shared encoder",
            "torchregress_BPF": "zuko Bernstein-polynomial flow on shared encoder",
            "torchregress_CFM": "conditional flow matching (OT-CFM, 32-step Euler sampler); no exact likelihood",
            "torchregress_NFM": "tabular Normalizing Flow Map (time-conditioned affine coupling, exact likelihood)",
            "sbi_MAF": "sbi posterior_nn('maf') = nflows MAF (3 transforms, 64 hidden), shared encoder as embedding_net; shared trainer",
            "sbi_zuko_MAF": "sbi posterior_nn('zuko_maf') (3 transforms, 64 hidden), shared encoder as embedding_net; shared trainer",
            "torchregress_MDN": "MixtureDensityLoss K=10, full covariance; shared encoder + 2x64 hidden head (as sbi MDN); shared trainer",
            "sbi_MDN": "sbi posterior_nn('mdn') K=10 full covariance, shared encoder as embedding_net; shared trainer",
            "sbi_NPE_NSF": "reference: sbi NPE trainer and defaults (5 transforms, 50 hidden, own early stopping)",
        }[method]

    # ------------------------------------------------------------------

    def _build(
        self, method: str, in_dim: int, out_dim: int, Y: np.ndarray, X: np.ndarray
    ) -> dict[str, Any]:
        dev = current_device()
        hidden = FLOW_HIDDEN if self._image_channels is not None else _hidden_size(in_dim)
        if method in _TR_FLOWS:
            from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model

            enc = self._encoder(in_dim, hidden).to(dev)
            flow_type = _TR_FLOWS[method]
            if flow_type == "bpf":
                flow = create_flow_model(
                    n_features=out_dim,
                    context_dim=hidden,
                    flow_type="bpf",
                    degree=8,
                ).to(dev)
            else:
                flow = create_flow_model(
                    n_features=out_dim,
                    context_dim=hidden,
                    flow_type=flow_type,
                    n_transforms=FLOW_TRANSFORMS,
                    hidden_features=[FLOW_HIDDEN, FLOW_HIDDEN],
                ).to(dev)
            loss = NormalizingFlowLoss(flow=flow).to(dev)
            return {"kind": "tr_flow", "modules": [enc, loss], "enc": enc, "loss": loss}
        if method == "torchregress_NFM":
            from torchregress.models import NormalizingFlowMap

            enc = self._encoder(in_dim, hidden).to(dev)
            nfm = NormalizingFlowMap(
                out_dim,
                hidden,
                n_layers=3,
                hidden=(FLOW_HIDDEN, FLOW_HIDDEN),
            ).to(dev)
            return {"kind": "tr_nfm", "modules": [enc, nfm], "enc": enc, "nfm": nfm}
        if method == "torchregress_CFM":
            from torchregress.models import FlowMatchingMLP

            enc = self._encoder(in_dim, hidden).to(dev)
            velocity = FlowMatchingMLP(out_dim, hidden, hidden=(FLOW_HIDDEN, FLOW_HIDDEN)).to(dev)
            return {"kind": "tr_cfm", "modules": [enc, velocity], "enc": enc, "velocity": velocity}
        if method == "torchregress_MDN":
            from torchregress.losses import MixtureDensityLoss

            mdn = MixtureDensityLoss(
                n_components=MDN_COMPONENTS, n_features=out_dim, covariance_type="full"
            ).to(dev)
            # Same shape as sbi's MDN: shared encoder, then two 64-unit hidden layers.
            net = torch.nn.Sequential(
                self._encoder(in_dim, hidden),
                torch.nn.Linear(hidden, FLOW_HIDDEN),
                torch.nn.ReLU(),
                torch.nn.Linear(FLOW_HIDDEN, FLOW_HIDDEN),
                torch.nn.ReLU(),
                torch.nn.Linear(FLOW_HIDDEN, mdn.expected_output_size),
            ).to(dev)
            return {"kind": "tr_mdn", "modules": [net], "net": net, "mdn": mdn}
        if method in _SBI_MODELS:
            from sbi.neural_nets import posterior_nn

            kind = _SBI_MODELS[method]
            arch: dict[str, Any] = (
                {"num_components": MDN_COMPONENTS}
                if kind == "mdn"
                else {"num_transforms": FLOW_TRANSFORMS}
            )
            enc = self._encoder(in_dim, hidden)
            builder = posterior_nn(
                model=kind,
                hidden_features=FLOW_HIDDEN,
                embedding_net=enc,
                z_score_theta="none",
                z_score_x="none",
                **arch,
            )
            Yb = torch.as_tensor(Y[:512], dtype=torch.float32)
            Xb = torch.as_tensor(X[: len(Yb)], dtype=torch.float32)
            est = builder(Yb, Xb).to(dev)
            return {"kind": "sbi_est", "modules": [est], "est": est}
        raise ValueError(method)

    @staticmethod
    def _nll_fn(state: dict[str, Any]) -> Any:
        k = state["kind"]
        if k == "tr_flow":
            return lambda xb, yb: -state["loss"].log_prob(state["enc"](xb), yb)
        if k == "tr_mdn":
            mdn = state["mdn"]
            return lambda xb, yb: mdn._calculate_nll(
                yb, mdn._extract_distribution_parameters(state["net"](xb))
            )
        if k == "tr_nfm":
            nfm = state["nfm"]
            return lambda xb, yb: nfm.training_loss(state["enc"](xb), yb)
        if k == "tr_cfm":
            from torchregress.models import cfm_loss

            return lambda xb, yb: cfm_loss(state["velocity"], state["enc"](xb), yb)
        if k == "sbi_est":
            return lambda xb, yb: state["est"].loss(yb, xb)
        raise ValueError(k)

    def _log_prob_and_sample(
        self, state: dict[str, Any], X: np.ndarray, Y: np.ndarray, n_s: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """log p(y_true | x) [N], samples [N, S, D], log p(samples | x) [N, S]."""
        dev = current_device()
        k = state["kind"]
        Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
        Yt = torch.as_tensor(Y, dtype=torch.float32, device=dev)
        N, D = Y.shape
        lps, samp, lps_s = [], [], []
        with torch.no_grad():
            for start in range(0, N, 100):
                xb, yb = Xt[start : start + 100], Yt[start : start + 100]
                b = len(xb)
                if k == "tr_flow":
                    ctx = state["enc"](xb)
                    lp = state["loss"].log_prob(ctx, yb)
                    s = state["loss"].sample(ctx, n_samples=n_s)  # [B, S, D]
                    ctx_r = ctx.repeat_interleave(n_s, dim=0)
                    lp_s = state["loss"].log_prob(ctx_r, s.reshape(b * n_s, D)).reshape(b, n_s)
                elif k == "tr_nfm":
                    nfm = state["nfm"]
                    ctx = state["enc"](xb)
                    lp = nfm.log_prob(ctx, yb)
                    s = nfm.sample(ctx, n_samples=n_s)
                    ctx_r = ctx.repeat_interleave(n_s, dim=0)
                    lp_s = nfm.log_prob(ctx_r, s.reshape(b * n_s, D)).reshape(b, n_s)
                elif k == "tr_cfm":
                    from torchregress.models import sample_cfm

                    s = sample_cfm(state["velocity"], state["enc"](xb), n_samples=n_s, steps=32)
                    lp = torch.full((b,), float("nan"), device=dev)
                    lp_s = None
                elif k == "tr_mdn":
                    mdn = state["mdn"]
                    out = state["net"](xb)
                    lp = -mdn._calculate_nll(yb, mdn._extract_distribution_parameters(out))
                    s = mdn.sample(out, n_samples=n_s).transpose(0, 1)  # [B, S, D]
                    out_r = out.repeat_interleave(n_s, dim=0)
                    lp_s = -mdn._calculate_nll(
                        s.reshape(b * n_s, D), mdn._extract_distribution_parameters(out_r)
                    ).reshape(b, n_s)
                else:  # sbi ConditionalDensityEstimator
                    est = state["est"]
                    lp = est.log_prob(yb.unsqueeze(0), xb).squeeze(0)
                    s = est.sample((n_s,), xb)  # [S, B, D]
                    lp_s = est.log_prob(s, xb).transpose(0, 1)  # [B, S]
                    s = s.transpose(0, 1)
                lps.append(lp.reshape(-1).cpu().numpy())
                samp.append(s.reshape(b, n_s, D).cpu().numpy())
                if lp_s is not None:
                    lps_s.append(lp_s.reshape(b, n_s).cpu().numpy())
        lp_out = np.concatenate(lps)
        lp_s_out = np.concatenate(lps_s) if lps_s else None
        return lp_out, np.concatenate(samp), lp_s_out

    def _pzflow(
        self, X: np.ndarray, Y: np.ndarray, Xte: np.ndarray, Yte: np.ndarray, cfg: HarnessConfig
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
        import pandas as pd
        import pzflow

        cx = [f"x{i}" for i in range(X.shape[1])]
        cy = [f"y{i}" for i in range(Y.shape[1])]
        df = pd.DataFrame(np.hstack([X, Y]), columns=cx + cy)
        flow = pzflow.Flow(
            data_columns=cy,
            conditional_columns=cx,
            latent=pzflow.distributions.Normal(input_dim=Y.shape[1]),
            seed=int(cfg.seed),
        )
        flow.train(
            df,
            epochs=int(cfg.epochs),
            batch_size=int(cfg.batch_size),
            verbose=False,
            seed=int(cfg.seed),
        )
        dte = pd.DataFrame(np.hstack([Xte, Yte]), columns=cx + cy)
        lp = np.asarray(flow.log_prob(dte))
        s = flow.sample(N_EVAL_SAMPLES, conditions=dte[cx], seed=int(cfg.seed))
        samples = np.stack([np.asarray(s[c]) for c in cy], axis=1).reshape(
            len(Xte), N_EVAL_SAMPLES, len(cy)
        )
        n_params = int(sum(np.size(leaf) for leaf in _jax_leaves(flow._params)))
        flat = pd.DataFrame(
            np.hstack([np.repeat(Xte, N_EVAL_SAMPLES, axis=0), samples.reshape(-1, len(cy))]),
            columns=cx + cy,
        )
        lp_s = np.asarray(flow.log_prob(flat)).reshape(len(Xte), N_EVAL_SAMPLES)
        return lp, samples, lp_s, n_params

    def _sbi_npe(self, X: np.ndarray, Y: np.ndarray, cfg: HarnessConfig) -> dict[str, Any]:
        from sbi.inference import NPE
        from sbi.utils import BoxUniform

        D = Y.shape[1]
        prior = BoxUniform(low=-20.0 * torch.ones(D), high=20.0 * torch.ones(D))
        inf = NPE(prior=prior, density_estimator="nsf", show_progress_bars=False)
        inf.append_simulations(
            torch.as_tensor(Y, dtype=torch.float32), torch.as_tensor(X, dtype=torch.float32)
        )
        est = inf.train(max_num_epochs=int(cfg.epochs), training_batch_size=int(cfg.batch_size))
        return {"kind": "sbi_est", "modules": [est], "est": est.to(current_device())}

    # ------------------------------------------------------------------

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: dict[str, np.ndarray] | None = None,
        *,
        scenario_name: str | None = None,
    ) -> list[dict[str, Any]]:
        cfg = config or HarnessConfig()
        if splits is None:
            from harness.datasets import load_dataset

            splits = load_dataset(dataset_name)
        X = np.asarray(splits["X_train"], dtype=np.float32)
        Y = np.asarray(splits["y_train"], dtype=np.float32)
        Xte = np.asarray(splits["X_test"], dtype=np.float32)
        Yte = np.asarray(splits["y_test"], dtype=np.float32)
        if Y.ndim == 1:
            Y, Yte = Y[:, None], Yte[:, None]
        mu, sd = Y.mean(0), Y.std(0) + 1e-8
        Y, Yte = (Y - mu) / sd, (Yte - mu) / sd
        if len(Xte) > MAX_TEST_ROWS:
            idx = np.random.default_rng(cfg.seed).choice(len(Xte), MAX_TEST_ROWS, replace=False)
            Xte, Yte = Xte[idx], Yte[idx]
        if cfg.smoke:
            X, Y = X[:512], Y[:512]
            Xte, Yte = Xte[:64], Yte[:64]
        self._image_channels = (
            int(splits.get("_n_channels", X.shape[1])) if splits.get("_image_data") else None
        )
        in_dim, out_dim = X.shape[1], Y.shape[1]
        n_s = 50 if cfg.smoke else N_EVAL_SAMPLES

        rows: list[dict[str, Any]] = []
        for method in self.methods():
            set_harness_seed(cfg.seed)
            base = {
                "Method": f"{self.name}/{method}",
                "Library": self.library_name(method),
                "LibraryVersion": self.library_version(method),
                "Dataset": dataset_name,
                "Suite": self.name,
                "NTargets": out_dim,
                "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                "Device": str(current_device()),
            }
            if self._image_channels is not None and method in ("pzflow_NSF", "sbi_NPE_NSF"):
                continue  # pzflow takes tabular columns only; NPE reference is tabular-only
            try:
                t0 = perf_counter()
                if method == "pzflow_NSF":
                    lp, samples, lp_s, n_params = self._pzflow(X, Y, Xte, Yte, cfg)
                    epochs = int(cfg.epochs)
                    fit_s = perf_counter() - t0
                    t1 = perf_counter()
                else:
                    if method == "sbi_NPE_NSF":
                        state = self._sbi_npe(X, Y, cfg)
                        epochs = -1
                    else:
                        state = self._build(method, in_dim, out_dim, Y, X)
                        epochs = _fit_shared(state["modules"], self._nll_fn(state), X, Y, cfg)
                    fit_s = perf_counter() - t0
                    n_params = int(sum(p.numel() for m in state["modules"] for p in m.parameters()))
                    t1 = perf_counter()
                    lp, samples, lp_s = self._log_prob_and_sample(state, Xte, Yte, n_s)
                scores = _scores(Yte, samples, lp, lp_s)
                dump_dir = os.environ.get(DUMP_ENV)
                if dump_dir:
                    os.makedirs(dump_dir, exist_ok=True)
                    np.savez_compressed(
                        os.path.join(dump_dir, f"{dataset_name}__seed{cfg.seed}__{method}.npz"),
                        samples=samples[:N_DUMP_ROWS].astype(np.float32),
                        y_true=Yte[:N_DUMP_ROWS],
                        y_mean=mu,
                        y_std=sd,
                    )
                rows.append(
                    {
                        **base,
                        **scores,
                        "Params": n_params,
                        "Epochs": epochs,
                        "train_s": fit_s,
                        "eval_s": perf_counter() - t1,
                        "Notes": self.method_notes(method),
                    }
                )
            except Exception as exc:  # noqa: BLE001 — row-level resilience
                rows.append(
                    {
                        **base,
                        **dict.fromkeys(_METRICS),
                        "train_s": None,
                        "eval_s": None,
                        "Notes": f"failed: {type(exc).__name__}: {exc}",
                    }
                )
        return rows

    # HarnessRunner abstract surface (unused: custom run drives everything).
    def fit(self, method: str, splits: dict[str, np.ndarray], config: HarnessConfig) -> Any:
        raise NotImplementedError("density_parity uses custom run()")

    def predict(
        self, method: str, model_state: Any, X_test: np.ndarray, config: HarnessConfig
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError("density_parity uses custom run()")

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        raise NotImplementedError("density_parity uses custom run()")


def _jax_leaves(tree: Any) -> list[Any]:
    import jax

    return jax.tree_util.tree_leaves(tree)


class ImageDensityParitySuite(DensityParitySuite):
    """Image -> target density parity: shared ``_ImageCNN`` encoder for every method."""

    @property
    def name(self) -> str:
        return "image_density_parity"

    @property
    def description(self) -> str:
        return (
            "Conditional density estimators on images: torchregress NSF/MAF/MDN vs sbi "
            "NSF/MAF/MDN with the same CNN encoder (embedding_net) and trainer"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "synthetic_image_keypoint",
            "udacity_steering",
            "ixi_brainage",
        ]
