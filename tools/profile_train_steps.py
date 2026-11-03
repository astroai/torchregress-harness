"""Per-step cost breakdown of the likelihoods-suite training loops on the current device.

Times mini-batch training steps (batch 1024, the suite's ``_BIG_BATCH``) of the
MLP family heads, the NSF head and the FT-Transformer, and prints a
``torch.profiler`` summary (top ops, host-device syncs) for each, so a slow
row can be attributed to kernels, Python overhead or synchronisation.

    pixi run -e gpu python tools/profile_train_steps.py [--steps 200] [--only skewt,nsf]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

import torch  # noqa: E402
from suites._backbones import _MLP, _hidden_size  # noqa: E402
from torch import nn  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from harness.devices import set_default_device  # noqa: E402

BS = 1024


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _step_fn(model: nn.Module, loss_fn: nn.Module, params, X, y, opt_kw):
    opt = torch.optim.Adam(params, lr=1e-3, **opt_kw)
    n = X.shape[0]

    def step(i: int) -> None:
        idx = torch.randint(0, n, (BS,), device=X.device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(X[idx]), y[idx])
        loss.backward()
        opt.step()

    return step


def _bench(name: str, step, steps: int, prof_steps: int = 10) -> float:
    for i in range(10):
        step(i)
    _sync()
    t = time.perf_counter()
    for i in range(steps):
        step(i)
    _sync()
    dt = (time.perf_counter() - t) / steps
    print(f"STEP {name}: {dt * 1e3:.2f} ms/step", flush=True)
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    with profile(activities=acts) as prof:
        for i in range(prof_steps):
            step(i)
        _sync()
    ka = prof.key_averages()
    syncs = sum(e.count for e in ka if e.key in ("cudaStreamSynchronize", "cudaDeviceSynchronize"))
    d2h = sum(e.count for e in ka if "Memcpy DtoH" in e.key)
    launches = sum(
        e.count
        for e in ka
        if e.key in ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC")
    )
    print(
        f"PROF {name}: per step syncs={syncs / prof_steps:.1f} d2h={d2h / prof_steps:.1f} "
        f"kernel_launches={launches / prof_steps:.0f}",
        flush=True,
    )
    print(ka.table(sort_by="self_cpu_time_total", row_limit=12), flush=True)
    return dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--only", default="")
    ap.add_argument("--mem-gib", type=float, default=10.75)
    args = ap.parse_args()
    dev = set_default_device("auto")
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(
            f"ENV {p.name} sms={p.multi_processor_count} mem={p.total_memory / 2**30:.2f}GiB",
            flush=True,
        )
    only = set(filter(None, args.only.split(",")))
    g = torch.Generator().manual_seed(0)
    n = 200_000
    from torchregress.losses import (
        GaussianNLLLoss,
        MixtureDensityLoss,
        SkewTLoss,
    )
    from torchregress.losses import families as fam

    for d_in in (8, 70):
        X = torch.randn(n, d_in, generator=g).to(dev)
        y = (torch.randn(n, 1, generator=g).exp() - 1.6).to(dev)  # right-skewed, z-ish
        h = _hidden_size(d_in)
        cases = {
            "gaussian": (lambda: _MLP(d_in, h, 2), GaussianNLLLoss),
            "skewt": (lambda: _MLP(d_in, h, 4), SkewTLoss),
        }
        for key, (mk, lcls) in cases.items():
            if only and key not in only:
                continue
            for opt_kw in ({}, {"fused": True}):
                torch.manual_seed(0)
                m = mk().to(dev)
                lf = lcls().to(dev)
                tag = f"{key}_d{d_in}{'_fused' if opt_kw else ''}"
                if key == "skewt":
                    calls = {"n": 0, "it": 0}
                    orig = fam._betacf

                    def counting(a, b, x, itmax=300, eps=3e-14, _o=orig):
                        calls["n"] += 1
                        # replicate the loop count cheaply: run with no grad
                        with torch.no_grad():
                            for k in (8, 16, 32, 64, 128, 256, 300):
                                h1 = _o(a.detach(), b.detach(), x.detach(), itmax=k, eps=0.0)
                                h2 = _o(a.detach(), b.detach(), x.detach(), itmax=k + 1, eps=0.0)
                                if bool(((h2 / h1) - 1).abs().max() < eps):
                                    break
                        calls["it"] += k
                        return _o(a, b, x, itmax=itmax, eps=eps)

                    fam._betacf = counting
                    step = _step_fn(m, lf, list(m.parameters()), X, y, opt_kw)
                    for i in range(20):
                        step(i)
                    fam._betacf = orig
                    print(
                        f"SKEWT_ITERS {tag}: ~{calls['it'] / max(calls['n'], 1):.0f} CF iterations/call (upper bracket)"
                    )
                step = _step_fn(m, lf, list(m.parameters()), X, y, opt_kw)
                _bench(tag, step, args.steps if key != "skewt" else 50)
        if (not only or "mdn" in only) and d_in == 8:
            m = _MLP(
                d_in, h, MixtureDensityLoss(n_components=3, n_features=1).expected_output_size
            ).to(dev)
            lf = MixtureDensityLoss(n_components=3, n_features=1)
            _bench("mdn_d8", _step_fn(m, lf, list(m.parameters()), X, y, {}), args.steps)
        if not only or "nsf" in only:
            from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model

            flow = create_flow_model(
                n_features=1,
                context_dim=h,
                flow_type="nsf",
                n_transforms=3,
                hidden_features=[64, 64],
            )
            m = _MLP(d_in, h, h).to(dev)
            lf = NormalizingFlowLoss(flow=flow).to(dev)
            _bench(
                f"nsf_d{d_in}",
                _step_fn(m, lf, list(m.parameters()) + list(flow.parameters()), X, y, {}),
                100,
            )
        if not only or "ftt" in only:
            from torchregress.models import FTTransformer

            for mode in ("fp32", "tf32", "bf16"):
                torch.manual_seed(0)
                m = FTTransformer(d_in, out_dim=2).to(dev)
                lf = GaussianNLLLoss()
                opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-5)
                torch.backends.cuda.matmul.allow_tf32 = mode == "tf32"
                torch.backends.cudnn.allow_tf32 = mode == "tf32"

                def step(i: int, m=m, lf=lf, opt=opt, mode=mode) -> None:
                    idx = torch.randint(0, n, (BS,), device=X.device)
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=mode == "bf16"):
                        out = m(X[idx])
                    loss = lf(out.float(), y[idx])
                    loss.backward()
                    nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step()

                _bench(f"ftt_d{d_in}_{mode}", step, 50 if d_in == 70 else 100)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False


if __name__ == "__main__":
    main()
