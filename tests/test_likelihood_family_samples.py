"""The likelihoods suite draws CRPS/PIT samples from exactly the fitted predictive.

Regression test for the 0.3.0 TLC-fare run: ``_family_samples`` clamped shape
parameters (Johnson-SU ``delta >= 0.3``, sinh-arcsinh ``eps in [-8, 8]`` and
``delta <= 10``, skew-t ``nu >= 2.1``, ALD ``kappa >= 0.2``, ...), so sample
scores described a lighter-tailed distribution than the one whose NLL the row
reported.  Each case below sits outside an old clamp; the empirical CDF of the
draws must match the CDF obtained by integrating the library density.
"""

from __future__ import annotations

import importlib
import math

import pytest
import suites.tabular.likelihoods as lik
import torch


def _raw_pos(value: float) -> float:
    """Raw head output that ``softplus(.) + 1e-6`` maps to ``value``."""
    v = value - 1e-6
    return v + math.log(-math.expm1(-v))


# (family, raw head row) — every row is outside the old sampling clamps.
_CASES = [
    ("johnsonsu", [0.0, _raw_pos(0.5), 0.5, _raw_pos(0.1)]),  # delta 0.1 < 0.3
    ("sinharcsinh", [0.0, _raw_pos(1e-3), 10.0, _raw_pos(2.0)]),  # eps 10 > 8
    ("sinharcsinh", [0.0, _raw_pos(1e-3), 0.5, _raw_pos(12.0)]),  # delta 12 > 10
    ("skewt", [0.0, _raw_pos(1.0), 2.0, _raw_pos(1.2)]),  # nu 1.2 < 2.1
    ("aslaplace", [0.0, _raw_pos(1.0), _raw_pos(0.1)]),  # kappa 0.1 < 0.2
    ("aslaplace", [0.0, _raw_pos(1.0), _raw_pos(8.0)]),  # kappa 8 > 5
    ("skewnormal", [0.0, _raw_pos(1.0), -60.0]),  # |alpha| 60 > 10
]


def _density_cdf(family: str, raw: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """CDF at ``y = loc + scale*sinh(t)`` from the library NLL (trapezoid in t)."""
    modname, clsname, _ = lik._FAMILY_LOSS[family]
    loss_fn = getattr(importlib.import_module(modname), clsname)(reduction="none")
    loc = raw[0]
    scale = torch.nn.functional.softplus(raw[1]) + 1e-6
    y = loc + scale * torch.sinh(t)
    nll = loss_fn(raw[None].expand(len(t), -1), y[:, None]).reshape(-1)
    dens = torch.exp(-nll) * scale * torch.cosh(t)  # density of t
    cdf = torch.cat([torch.zeros(1, dtype=t.dtype), torch.cumulative_trapezoid(dens, t)])
    return cdf


@pytest.mark.parametrize(("family", "row"), _CASES)
def test_family_draws_match_library_density(family: str, row: list[float]) -> None:
    raw = torch.tensor(row, dtype=torch.float64)
    t = torch.linspace(-60.0, 60.0, 600_001, dtype=torch.float64)
    cdf = _density_cdf(family, raw, t)
    assert abs(cdf[-1].item() - 1.0) < 1e-3  # the grid holds the whole density

    n = 100_000
    draws = lik._family_samples(family, raw[None].float(), n, seed=0)
    assert draws.dtype == torch.float64 and draws.shape == (n, 1)
    assert torch.isfinite(draws).all()
    scale = torch.nn.functional.softplus(raw[1]) + 1e-6
    t_draws = torch.sort(torch.asinh((draws[:, 0] - raw[0]) / scale)).values
    emp = torch.searchsorted(t_draws, t).double() / n
    ks = (emp - cdf).abs().max().item()
    # 100k draws: the KS statistic of an exact sampler is ~0.004 at most here;
    # the old clamped draws are off by 0.02-0.9 on these rows.
    assert ks < 0.01, f"{family} draws disagree with the library density (KS={ks:.4f})"
