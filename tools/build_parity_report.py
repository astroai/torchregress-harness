#!/usr/bin/env python
"""Build ``PARITY_REPORT.md`` from a release run's per-job JSON artifacts.

The run directory holds one sub-directory per job group (``tabular``,
``tabular_big``, ``sbi``, ``fm``, ``image``, ...).  Each group contains one
harness JSON per (suite, dataset, seed), named ``<suite>__<dataset>__seed<k>.json``,
plus optional ``versions_replica_*.txt`` files written by the release jobs
(installed package versions, including git commits of pinned comparators).

Usage::

    pixi run python tools/build_parity_report.py --run-dir RUN --out PARITY_REPORT.md \\
        --run-tag release_20261013_1917 --harness-ref 9e8c72b --lib-ref 6ae7272 \\
        --preamble results/release_20261013_1917/parity_preamble.md \\
        --note "image reruns pending"

The output is deterministic for a given input (sorted keys, fixed float
formatting, the date taken from ``--date``), so regenerating it after a rerun
gives a clean diff.

Report contents
---------------
* header: date, run tag, commits, comparator versions (union over groups, with
  differing versions flagged), environment per group, seeds / epochs / config,
  provenance (row counts, synthetic-fallback alarm);
* summary: counts per group / suite and the torchregress-vs-comparator parity
  table built from the explicit :data:`PAIR_RULES`;
* one section per suite with per-dataset tables (mean ± std over seeds, best
  value per column in bold, runtime, failed / skipped rows listed beneath);
* appendix: failure / skip ledger, method descriptions, methodology.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "tools"))

# --------------------------------------------------------------------------- #
# Conventions
# --------------------------------------------------------------------------- #

EXACT_TOL = 1e-6
CLOSE_TOL = 0.02
SPREAD_TOL = 0.05
Z_NOISE = 2.0
KINDS: tuple[str, ...] = ("deterministic", "matched", "method")
NOTE_WIDTH = 120

CPU_MARK = "‡"
SPREAD_MARK = "†"

#: Metrics where larger is better.
HIGHER_IS_BETTER: frozenset[str] = frozenset(
    {
        "R2",
        "R2_clean",
        "Corr",
        "Concordance",
        "Accuracy",
        "LibAccuracy",
        "BalancedAccuracy",
        "QWK",
        "RefLogProb",
        "TARP_KSp",
        "SBI_SBC_KSpMin",
        "AdaptGain",
        "WeightESS",
        "EvalRowsPerSec",
        "ValidPixelFrac",
    }
)
#: Metrics whose ideal value is 0 in absolute value (best = smallest |x|).
ABS_IS_BETTER: frozenset[str] = frozenset(
    {
        "Bias",
        "TARP_ATC",
        "CoverageDeviation",
        "CoverageDeviationMean",
        "PIT_Mean",
        "NLLGapToTruth",
        "PolicyErr",
        "ATEErr",
        "Diag_Shift",
    }
)
#: Metrics with a nominal target or no natural order: never bolded.
NEUTRAL: frozenset[str] = frozenset(
    {
        "Coverage",
        "Coverage50",
        "Coverage80",
        "Coverage90",
        "Coverage95",
        "Cov95",
        "Cov90_marg",
        "Cov90_joint",
        "Cov90_ell",
        "JointCoverage",
        "Coverage_D0",
        "Coverage_D1",
        "Coverage_D2",
        "SBC_Coverage95",
        "ShiftCoverage",
        "TailCoverage",
        "WorstDecileCoverage",
        "Covers",
        "Width",
        "CIWidth",
        "Estimate",
        "SEratio",
        "Attenuation",
        "ZeroFracPred",
        "PIT_Std",
        "SBC_RankMean",
        "Theta",
        "PredictedPositiveRate",
        "InfiniteIntervalFrac",
        "MeanTrustWeight",
        "VTS_Temperature",
    }
)
#: Log-scale metrics: parity differences are absolute nats over max(|ref|, 1).
LOG_SCALE: frozenset[str] = frozenset({"NLL", "RefLogProb", "LogLoss", "TrueNLL", "EnsembleNLL"})

#: Fallback metric priority when a suite has no explicit column list.
DEFAULT_METRICS: tuple[str, ...] = (
    "RMSE",
    "MAE",
    "R2",
    "NLL",
    "CRPS",
    "Coverage",
    "Width",
    "IntervalScore",
    "ECE",
    "PIT_KS",
)

#: Main columns per suite (only columns present in the suite's rows are shown).
SUITE_METRICS: dict[str, tuple[str, ...]] = {
    "conformal": ("Coverage", "Width", "IntervalScore", "InfiniteIntervalFrac"),
    "image_conformal": ("Coverage", "Width", "IntervalScore", "InfiniteIntervalFrac"),
    "uq_layer": ("Coverage", "Width", "IntervalScore", "RMSE", "CRPS"),
    "r2c_conformal": ("Coverage", "Width", "IntervalScore", "RMSE"),
    "fm_uq_scenarios": ("Coverage90", "Width", "IntervalScore", "RMSE", "WorstDecileCoverage"),
    "fm_multitarget": ("ES", "VS", "CRPS", "Cov90_marg", "Cov90_joint", "CorrErr"),
    "foundation_model": ("RMSE", "R2", "NLL", "CRPS", "Coverage", "IntervalScore"),
    "sbi_parity": ("C2ST", "RefLogProb", "MeanRMSE", "SBC_Coverage95", "TARP_ATC"),
    "density_parity": ("NLL", "CRPS", "ES", "Cov90_marg", "Cov90_joint", "PIT_KS"),
    "image_density_parity": ("NLL", "CRPS", "ES", "Cov90_marg", "Cov90_joint", "PIT_KS"),
    "pzflow_parity": ("NLL", "Coverage", "Width", "IntervalScore", "EnergyScore", "PIT_KS"),
    "orthogonal_inference": ("Bias", "RMSE", "Coverage95", "CIWidth", "SEratio"),
    "tweedie": ("RMSE", "R2", "TweedieDeviance", "ZeroFracPred"),
    "likelihoods": ("NLL", "NLLGapToTruth", "LocRMSE", "ScaleRMSE", "RMSE", "CRPS"),
    "causal": ("ATEErr", "PEHE", "Covers", "PolicyErr"),
    "ppi": ("Estimate", "CIWidth", "Covers", "RMSE"),
    "ordinal": ("MAE", "Accuracy", "QWK", "ClassMAE"),
    "censored": ("RMSE", "MAE", "Concordance", "ObservedMAE"),
    "censored_imaging": ("RMSE", "MAE", "Concordance", "ObservedMAE"),
    "label_shift": ("RMSE", "WeightRMSE", "TargetMAE", "ShiftCoverage", "ShiftWidth"),
    "classification_shift": ("Accuracy", "BalancedAccuracy", "LogLoss", "Brier", "TargetPriorL1"),
    "multivariate_intervals": ("JointCoverage", "IntervalScore", "EnergyScore", "VariogramScore"),
    "distributional_bins": ("CoverageDeviationAbsMean", "Coverage", "IntervalScore", "CRPS", "NLL"),
    "eiv": ("RMSE", "R2", "Attenuation", "RMSE_clean", "Coverage", "IntervalScore"),
    "missing_data": ("RMSE", "MAE", "R2"),
    "robust_regression": ("RMSE", "MAE", "R2", "TailRMSE"),
    "sklearn_baselines": ("RMSE", "MAE", "R2", "NLL", "Coverage", "IntervalScore"),
    "ensembles": ("RMSE", "NLL", "CRPS", "Coverage", "IntervalScore"),
    "semi_supervised": ("RMSE", "MAE", "R2"),
    "constrained_outputs": ("RMSE", "MAE", "ViolationRate", "Coverage", "Pinball"),
    "cell_counting": ("RMSE", "MAE", "NLL", "Coverage", "IntervalScore"),
    "crowd_counting": ("CountMAE", "CountRMSE", "RMSE", "MAE"),
    "depth_estimation": ("RMSE", "MAE", "NLL", "CRPS", "Coverage", "ValidPixelFrac"),
    "streaming_tta": ("RMSE", "Coverage", "Width", "IntervalScore", "AdaptGain"),
    "test_time_adaptation": ("RMSE", "Coverage", "Width", "IntervalScore"),
}

#: Datasets that are synthetic by design but not in the harness registry.
EXTRA_SYNTHETIC_PREFIXES: tuple[str, ...] = (
    "synthetic_",
    "con_",
    "lik_",
    "plr_",
    "ssl_",
    "stress_",
    "sbibm_",
)


# --------------------------------------------------------------------------- #
# Pairing rules
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PairRule:
    """torchregress method(s) matched to comparator method(s) in one or more suites.

    ``tr`` is a full-match regex on the method name (suite prefix removed);
    each ``comps`` entry is a template formatted with the regex groups
    (``{1}``, ``{2}``, ...).  ``kind`` is ``"deterministic"`` when the two rows
    apply the same algorithm to the same fitted model / data and should agree
    to numerical precision, ``"matched"`` when they train the same architecture
    and capacity independently (agreement within seed noise expected), or
    ``"method"`` when they share the method but not the training recipe or
    model (agreement is informative, not expected).
    """

    suites: tuple[str, ...]
    tr: str
    comps: tuple[str, ...]
    metrics: tuple[str, ...]
    kind: str
    why: str


PAIR_RULES: tuple[PairRule, ...] = (
    PairRule(
        ("conformal",),
        r"Split\+Linear\(post-hoc\)",
        ("MAPIE_Split+Linear", "crepes_Split+Linear", "torchcp_Split+Linear"),
        ("IntervalScore", "Coverage", "Width"),
        "deterministic",
        "same LinearRegression base, absolute-residual split conformal",
    ),
    PairRule(
        ("conformal",),
        r"Split\+MLP",
        ("MAPIE_Split+MLP",),
        ("IntervalScore", "Coverage", "Width"),
        "matched",
        "split conformal on the harness MLP",
    ),
    PairRule(
        ("image_conformal",),
        r"Split\+CNN",
        ("MAPIE_Split+CNN", "crepes_Split+CNN", "torchcp_Split+CNN"),
        ("IntervalScore", "Coverage", "Width"),
        "deterministic",
        "split conformal on the same CNN predictions",
    ),
    PairRule(
        ("uq_layer",),
        r"tr_(Split|Normalized|CQR|CVPlus|JackknifePlus)\+(.+)",
        ("MAPIE_{1}+{2}", "crepes_{1}+{2}"),
        ("IntervalScore", "Coverage", "Width"),
        "deterministic",
        "same conformal method on the same fitted base model (`tr_X+Base` vs `MAPIE_X+Base` / `crepes_X+Base`)",
    ),
    PairRule(
        ("r2c_conformal",),
        r"R2C\+Softmax",
        ("torchcp_R2C",),
        ("IntervalScore", "Coverage", "Width"),
        "method",
        "R2C conformal; torchcp trains with its own R2ccp loss",
    ),
    PairRule(
        ("sbi_parity",),
        r"torchregress_NSF",
        ("sbi_NPE",),
        ("C2ST", "MeanRMSE"),
        "matched",
        "NSF posterior, matched capacity (row note `pair: sbi_NPE`)",
    ),
    PairRule(
        ("sbi_parity",),
        r"torchregress_(MAF|MDN)",
        ("sbi_NPE_{1}",),
        ("C2ST", "MeanRMSE"),
        "matched",
        "matched-capacity posterior estimator (row note `pair: sbi_NPE_X`)",
    ),
    PairRule(
        ("sbi_parity",),
        r"torchregress_NSF",
        ("pzflow_NSF",),
        ("C2ST", "MeanRMSE"),
        "method",
        "NSF posterior; pzflow uses its default bijector and trainer",
    ),
    PairRule(
        ("density_parity", "image_density_parity"),
        r"torchregress_(NSF|MAF)",
        ("sbi_{1}", "sbi_zuko_{1}"),
        ("NLL", "CRPS", "Cov90_marg"),
        "matched",
        "same flow size, shared encoder and shared trainer",
    ),
    PairRule(
        ("density_parity", "image_density_parity"),
        r"torchregress_MDN",
        ("sbi_MDN",),
        ("NLL", "CRPS", "Cov90_marg"),
        "matched",
        "mixture density network, shared encoder and trainer",
    ),
    PairRule(
        ("density_parity",),
        r"torchregress_NSF",
        ("pzflow_NSF", "sbi_NPE_NSF"),
        ("NLL", "CRPS", "Cov90_marg"),
        "method",
        "NSF with the comparator's own defaults and trainer",
    ),
    PairRule(
        ("pzflow_parity",),
        r"torchregress_NSF",
        ("pzflow_NSF", "normflows_NSF"),
        ("NLL", "IntervalScore", "Coverage"),
        "method",
        "conditional NSF; comparator uses its default architecture and trainer",
    ),
    PairRule(
        ("orthogonal_inference",),
        r"tr_Orthogonal_Poly3",
        ("DoubleML_PLR_Poly3Ridge",),
        ("RMSE", "Coverage95", "CIWidth"),
        "matched",
        "partially linear DML, same cubic ridge basis (row note `pair: tr_Orthogonal_Poly3`)",
    ),
    PairRule(
        ("tweedie",),
        r"(Tweedie|CompoundPoisson)_Linear",
        ("statsmodels_GLM",),
        ("TweedieDeviance", "RMSE", "R2"),
        "deterministic",
        "linear log-link Tweedie GLM: L-BFGS vs IRLS",
    ),
    PairRule(
        ("robust_regression",),
        r"Huber_Linear",
        ("statsmodels_RLM",),
        ("RMSE", "MAE"),
        "deterministic",
        "linear Huber regression vs statsmodels RLM(HuberT)",
    ),
    PairRule(
        ("robust_regression",),
        r"IRLS",
        ("statsmodels_RLM",),
        ("RMSE", "MAE"),
        "method",
        "library IRLS (Huber weights, max_iter=10, float32); tuning not matched to RLM",
    ),
    PairRule(
        ("robust_regression",),
        r"Quantile_Linear",
        ("sklearn_QuantileRegressor",),
        ("RMSE", "MAE"),
        "deterministic",
        "linear median regression (L-BFGS vs LP)",
    ),
    PairRule(
        ("bayesian_linear",),
        r"BLR_Linear",
        ("sklearn_BayesianRidge",),
        ("RMSE", "NLL", "CRPS"),
        "deterministic",
        "Bayesian linear regression, hyper-parameters matched",
    ),
    PairRule(
        ("likelihoods",),
        r"tr_Linear",
        ("scipy_MLE_Linear",),
        ("NLL", "LocRMSE", "ScaleRMSE"),
        "deterministic",
        "same linear conditional likelihood: torchregress loss vs scipy.stats log-density",
    ),
    PairRule(
        ("calibration",),
        r"IsotonicMeanCalibrator",
        ("sklearn_IsotonicRegression",),
        ("RMSE", "MAE"),
        "deterministic",
        "PAVA isotonic regression on the same predicted means",
    ),
    PairRule(
        ("sklearn_baselines",),
        r"OLS|GaussianNLL_Linear",
        ("sklearn_LinearRegression",),
        ("RMSE", "MAE"),
        "deterministic",
        "ordinary least squares mean",
    ),
    PairRule(
        ("causal",),
        r"DR",
        ("EconML_DRLearner",),
        ("ATEErr", "PEHE"),
        "method",
        "cross-fitted doubly robust learner; nuisances not identical",
    ),
    PairRule(
        ("ppi",),
        r"PPI_(Mean|Quantile)",
        ("PPI_py_{1}",),
        ("Estimate", "CIWidth"),
        "method",
        "prediction-powered inference; ppi_py mean uses PPI++ power tuning",
    ),
    PairRule(
        ("ordinal",),
        r"CumulativeLink",
        ("statsmodels_OrderedModel",),
        ("MAE", "QWK"),
        "method",
        "cumulative-logit ordinal model; MLP vs linear",
    ),
    PairRule(
        ("censored", "censored_imaging"),
        r"AFT",
        ("lifelines_LogNormalAFT",),
        ("RMSE", "Concordance"),
        "method",
        "log-normal AFT; MLP vs linear",
    ),
    PairRule(
        ("label_shift",),
        r"LibDomainClassifier",
        ("DomainClassifier",),
        ("WeightRMSE", "RMSE"),
        "method",
        "domain-classifier density ratio; MLP vs sklearn classifier",
    ),
    PairRule(
        ("missing_data",),
        r"PropensityWeighted",
        ("sklearn_IPW",),
        ("RMSE",),
        "method",
        "inverse propensity weighting; MLP vs Ridge",
    ),
    PairRule(
        ("image_ensembles_uq",),
        r"MC_Dropout",
        ("luqbox_MCDropout",),
        ("RMSE", "NLL"),
        "method",
        "MC dropout; lightning-uq-box head on CNN features",
    ),
    PairRule(
        ("stress_matrix",),
        r"stress_ood/MC_Dropout",
        ("stress_ood/luqbox_MCDropout",),
        ("RMSE", "NLL"),
        "method",
        "MC dropout under covariate shift",
    ),
    PairRule(
        ("ensembles",),
        r"Stacking",
        ("sklearn_StackingRegressor",),
        ("RMSE",),
        "method",
        "stacking; different base learners",
    ),
    PairRule(
        ("age_regression",),
        r"ResNet\+GaussianNLL",
        ("timm_resnet18+GaussianNLL",),
        ("RMSE", "NLL"),
        "method",
        "ResNet + Gaussian NLL; timm backbone",
    ),
    PairRule(
        ("age_regression",),
        r"CNN\+GaussianNLL",
        ("torch_uncertainty",),
        ("RMSE", "NLL"),
        "method",
        "distributional CNN; torch-uncertainty RegressionRoutine",
    ),
)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


@dataclass
class Row:
    group: str
    suite: str
    dataset: str
    method: str  # suite prefix removed; Scenario appended for scenario suites
    library: str
    seed: int
    status: str  # ok | failed | skipped
    notes: str
    data: dict[str, Any]
    source: str

    @property
    def cpu_retry(self) -> bool:
        return "ran on cpu" in self.notes.lower()


@dataclass
class Job:
    group: str
    suite: str
    dataset: str
    seed: int
    path: Path
    header: dict[str, Any]


@dataclass
class Run:
    jobs: list[Job] = field(default_factory=list)
    rows: list[Row] = field(default_factory=list)
    detail_rows: list[Row] = field(default_factory=list)
    versions: dict[str, list[tuple[str, str, str]]] = field(default_factory=dict)
    env_lines: dict[str, list[str]] = field(default_factory=dict)


_JOB_RE = re.compile(r"^(?P<suite>.+?)__(?P<dataset>.+)__seed(?P<seed>\d+)\.json$")
_DETAIL_RE = re.compile(r"/bin_\d+$")


def row_status(notes: str) -> str:
    low = notes.strip().lower()
    if low.startswith("skipped"):
        return "skipped"
    if low.startswith(("failed", "known-bad", "error")):
        return "failed"
    return "ok"


def _method_name(raw: Mapping[str, Any], suite: str) -> str:
    name = str(raw.get("Method", "?"))
    if name.startswith(suite + "/"):
        name = name[len(suite) + 1 :]
    scenario = raw.get("Scenario")
    if scenario:
        name = f"{name} [{scenario}]"
    return name


def _parse_versions(path: Path) -> tuple[list[tuple[str, str, str]], list[str]]:
    pkgs: list[tuple[str, str, str]] = []
    env: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("python ") and "|" in line:
            env.append(line.strip())
            continue
        parts = line.split("\t")
        if len(parts) < 2 or not parts[0] or " " in parts[0]:
            continue
        name, version = parts[0].strip(), parts[1].strip()
        commit = ""
        if len(parts) > 2 and parts[2].strip().startswith("{"):
            try:
                meta = json.loads(parts[2])
                commit = str(meta.get("vcs_info", {}).get("commit_id", ""))[:9]
            except json.JSONDecodeError:
                commit = ""
        pkgs.append((name, version, commit))
    return pkgs, env


def load_run_dir(run_dir: Path) -> Run:
    run = Run()
    for group_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        group = group_dir.name
        for path in sorted(group_dir.glob("*.json")):
            m = _JOB_RE.match(path.name)
            if not m:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
                continue
            header = {k: v for k, v in payload.items() if k != "rows"}
            seed = int(header.get("config", {}).get("seed", m["seed"]))
            run.jobs.append(Job(group, m["suite"], m["dataset"], seed, path, header))
            for raw in payload["rows"]:
                if not isinstance(raw, dict) or "Method" not in raw:
                    continue
                suite = str(raw.get("Suite") or m["suite"])
                notes = str(raw.get("Notes") or "")
                row = Row(
                    group=group,
                    suite=suite,
                    dataset=str(raw.get("Dataset") or m["dataset"]),
                    method=_method_name(raw, suite),
                    library=str(raw.get("Library") or ""),
                    seed=seed,
                    status=row_status(notes),
                    notes=notes,
                    data=raw,
                    source=f"{group}/{path.name}",
                )
                if _DETAIL_RE.search(row.method):
                    run.detail_rows.append(row)
                else:
                    run.rows.append(row)
        for vpath in sorted(group_dir.glob("versions_replica_*.txt")):
            pkgs, env = _parse_versions(vpath)
            run.versions[f"{group}/{vpath.name}"] = pkgs
            run.env_lines[f"{group}/{vpath.name}"] = env
    return run


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


@dataclass
class Stat:
    n: int
    mean: float
    std: float
    n_nonfinite: int = 0

    @property
    def rel_std(self) -> float:
        if self.n < 2 or self.mean == 0 or not math.isfinite(self.mean):
            return 0.0
        return self.std / abs(self.mean)


def stat(values: Iterable[Any]) -> Stat | None:
    nums = [v for v in (_num(x) for x in values) if v is not None]
    finite = [v for v in nums if math.isfinite(v)]
    nonfinite = len(nums) - len(finite)
    if not finite:
        if nonfinite:
            inf = [v for v in nums if math.isinf(v)]
            return Stat(0, inf[0] if inf else math.nan, 0.0, nonfinite)
        return None
    std = statistics.stdev(finite) if len(finite) > 1 else 0.0
    return Stat(len(finite), statistics.fmean(finite), std, nonfinite)


def fmt(x: float) -> str:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    a = abs(x)
    if a != 0 and (a >= 1e5 or a < 1e-3):
        return f"{x:.3e}"
    if a >= 100:
        return f"{x:.1f}"
    return f"{x:.4f}"


def _spec(x: float) -> str:
    a = abs(x)
    if a != 0 and (a >= 1e5 or a < 1e-3):
        return ".3e"
    if a >= 100:
        return ".1f"
    return ".4f"


def fmt_mean_std(mean: float, std: float) -> str:
    """``mean ± std`` with the std in the mean's format (fixed decimals or scientific)."""
    if not math.isfinite(mean):
        return fmt(mean)
    spec = _spec(mean)
    return f"{mean:{spec}} ± {std:{spec}}"


def fmt_pct(x: float) -> str:
    if math.isinf(x) or math.isnan(x):
        return "n/a"
    if x == 0:
        return "0"
    if x < 1e-4:
        return f"{x * 100:.1e}%"
    return f"{x * 100:.2f}%"


def direction(metric: str) -> str:
    if metric in NEUTRAL:
        return "none"
    if metric in HIGHER_IS_BETTER:
        return "max"
    if metric in ABS_IS_BETTER:
        return "abs"
    return "min"


def rel_diff(metric: str, a: float, b: float) -> float:
    """Relative difference of ``a`` (torchregress) from ``b`` (comparator)."""
    if not (math.isfinite(a) and math.isfinite(b)):
        return 0.0 if a == b else math.inf
    d = abs(a - b)
    if d == 0:
        return 0.0
    if metric in LOG_SCALE:
        return d / max(abs(b), 1.0)
    denom = max(abs(a), abs(b))
    return d / denom if denom > 0 else 0.0


def verdict(rd: float) -> str:
    if rd <= EXACT_TOL:
        return "exact"
    if rd <= CLOSE_TOL:
        return "close"
    return "differs"


def trunc(text: str, width: int = NOTE_WIDTH) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= width else text[: width - 1] + "…"


def md_escape(text: str) -> str:
    return text.replace("|", "\\|")


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

Key = tuple[str, str, str]  # suite, dataset, method


def index_rows(rows: Sequence[Row]) -> dict[Key, list[Row]]:
    out: dict[Key, list[Row]] = defaultdict(list)
    for r in rows:
        out[(r.suite, r.dataset, r.method)].append(r)
    for v in out.values():
        v.sort(key=lambda r: (r.seed, r.source))
    return out


def suite_metrics(suite: str, rows: Sequence[Row]) -> list[str]:
    present: set[str] = set()
    for r in rows:
        if r.status == "ok":
            present.update(k for k, v in r.data.items() if _num(v) is not None)
    wanted = SUITE_METRICS.get(suite, DEFAULT_METRICS)
    cols = [m for m in wanted if m in present]
    if not cols:
        cols = [m for m in DEFAULT_METRICS if m in present][:6]
    return cols


def main_metric(cols: Sequence[str]) -> str | None:
    for c in cols:
        if direction(c) != "none":
            return c
    return cols[0] if cols else None


@dataclass
class PairResult:
    suite: str
    dataset: str
    tr: str
    comp: str
    kind: str
    rule: PairRule
    values: dict[str, tuple[Stat, Stat]]  # metric -> (tr stat, comp stat) on common seeds
    per_metric: dict[str, float]  # metric -> relative difference of the seed means
    n_seeds: int

    @property
    def max_rel(self) -> float:
        return max(self.per_metric.values()) if self.per_metric else math.inf

    @property
    def verdict(self) -> str:
        return verdict(self.max_rel) if self.per_metric else "n/a"

    def z(self, metric: str) -> float | None:
        """|mean difference| in units of its standard error (None with < 2 seeds)."""
        a, b = self.values[metric]
        if a.n < 2 or b.n < 2:
            return None
        se = math.sqrt(a.std**2 / a.n + b.std**2 / b.n)
        d = abs(a.mean - b.mean)
        if se == 0:
            return 0.0 if d == 0 else math.inf
        return d / se

    @property
    def within_noise(self) -> str:
        if self.kind == "deterministic":
            return "-"
        zs = [self.z(m) for m in self.values]
        if not zs or any(z is None for z in zs):
            return "-"
        return "yes" if max(z for z in zs if z is not None) <= Z_NOISE else "no"


def evaluate_pairs(index: Mapping[Key, list[Row]]) -> list[PairResult]:
    by_sd: dict[tuple[str, str], dict[str, list[Row]]] = defaultdict(dict)
    for (suite, dataset, method), rows in index.items():
        by_sd[(suite, dataset)][method] = rows
    results: list[PairResult] = []
    seen: set[tuple[str, str, str, str]] = set()
    for (suite, dataset), methods in sorted(by_sd.items()):
        for rule in PAIR_RULES:
            if suite not in rule.suites:
                continue
            for tr_name in sorted(methods):
                m = re.fullmatch(rule.tr, tr_name)
                if not m:
                    continue
                groups = ("",) + m.groups()
                for tmpl in rule.comps:
                    comp = tmpl.format(*groups)
                    if comp not in methods or (suite, dataset, tr_name, comp) in seen:
                        continue
                    seen.add((suite, dataset, tr_name, comp))
                    results.append(
                        _compare(
                            suite, dataset, tr_name, comp, rule, methods[tr_name], methods[comp]
                        )
                    )
    return results


def _compare(
    suite: str,
    dataset: str,
    tr_name: str,
    comp: str,
    rule: PairRule,
    tr_rows: Sequence[Row],
    comp_rows: Sequence[Row],
) -> PairResult:
    tr_ok = {r.seed: r for r in tr_rows if r.status == "ok"}
    comp_ok = {r.seed: r for r in comp_rows if r.status == "ok"}
    seeds = sorted(set(tr_ok) & set(comp_ok))
    values: dict[str, tuple[Stat, Stat]] = {}
    per_metric: dict[str, float] = {}
    for metric in rule.metrics:
        sa = stat(tr_ok[s].data.get(metric) for s in seeds)
        sb = stat(comp_ok[s].data.get(metric) for s in seeds)
        if sa is None or sb is None:
            continue
        values[metric] = (sa, sb)
        per_metric[metric] = rel_diff(metric, sa.mean, sb.mean)
    return PairResult(
        suite, dataset, tr_name, comp, rule.kind, rule, values, per_metric, len(seeds)
    )


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def table(header: Sequence[str], rows: Iterable[Sequence[str]], align: Sequence[str]) -> list[str]:
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    out.extend("| " + " | ".join(r) + " |" for r in rows)
    return out


def anchor(text: str) -> str:
    """GitHub-style heading anchor."""
    return re.sub(r"[^a-z0-9 _-]", "", text.lower()).replace(" ", "-")


def synthetic_alarm(rows: Sequence[Row]) -> list[str]:
    """Rows with ``SyntheticFallback`` true on a dataset that is not synthetic by design."""
    try:
        from check_provenance import registry_sets

        synthetic, known = registry_sets()
    except Exception:  # pragma: no cover - harness not importable
        synthetic, known = frozenset(), frozenset()
    out: list[str] = []
    for r in rows:
        if r.status != "ok" or not r.data.get("SyntheticFallback"):
            continue
        if r.dataset in synthetic or r.dataset.startswith(EXTRA_SYNTHETIC_PREFIXES):
            continue
        reason = (
            "real dataset fell back to synthetic data"
            if r.dataset in known
            else "synthetic data on a dataset not registered as synthetic"
        )
        out.append(f"{r.source}: `{r.suite}/{r.method}` on `{r.dataset}` seed {r.seed}: {reason}")
    return sorted(out)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


@dataclass
class Options:
    run_tag: str
    harness_ref: str
    lib_ref: str
    date: str
    preamble: str = ""
    notes: tuple[str, ...] = ()


def build_report(run: Run, opts: Options) -> str:
    index = index_rows(run.rows)
    pairs = evaluate_pairs(index)
    suites_by_group: dict[str, set[str]] = defaultdict(set)
    for j in run.jobs:
        suites_by_group[j.group].add(j.suite)
    for r in run.rows:
        suites_by_group[r.group].add(r.suite)

    out: list[str] = []
    out += _header(run, opts)
    if opts.preamble.strip():
        out += [opts.preamble.strip(), ""]
    out += _summary(run, index, suites_by_group)
    out += _parity_section(pairs)
    flagged: list[tuple[str, str, str, str, float]] = []
    out += ["## Results by suite", ""]
    for group in sorted(suites_by_group):
        out += [f"### Group `{group}`", ""]
        for suite in sorted(suites_by_group[group]):
            out += _suite_section(group, suite, run, index, flagged)
    out += _spread_section(flagged)
    out += _appendix(run, index)
    text = "\n".join(out).rstrip() + "\n"
    return re.sub(r"\n{3,}", "\n\n", text)


def _header(run: Run, opts: Options) -> list[str]:
    out = [
        "# torchregress Parity Report",
        "",
        f"Generated {opts.date} by `tools/build_parity_report.py` from the release run "
        f"artifacts. Do not edit by hand: regenerate.",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Run tag | `{opts.run_tag}` |",
        f"| torchregress commit | `{opts.lib_ref}` |",
        f"| torchregress-harness commit | `{opts.harness_ref}` |",
        f"| Job groups | {', '.join(f'`{g}`' for g in sorted({j.group for j in run.jobs}))} |",
        f"| Job artifacts | {len(run.jobs)} |",
        f"| Result rows | {len(run.rows) + len(run.detail_rows)} "
        f"({len(run.detail_rows)} per-bin detail rows not tabulated) |",
        "",
    ]
    if opts.notes:
        out += ["**Notes:**", ""]
        out += [f"- {n}" for n in opts.notes]
        out += [""]
    out += _contents()
    out += _versions(run)
    out += _environment(run)
    out += _provenance(run)
    return out


def _contents() -> list[str]:
    return [
        "## Contents",
        "",
        "- [Comparator versions](#comparator-versions)",
        "- [Environment and configuration](#environment-and-configuration)",
        "- [Provenance](#provenance)",
        "- [Summary](#summary)",
        "- [Parity: torchregress vs comparators](#parity-torchregress-vs-comparators)",
        "- [Results by suite](#results-by-suite)",
        "- [Seed spread](#seed-spread)",
        "- [Appendix A: failure and skip ledger](#appendix-a-failure-and-skip-ledger)",
        "- [Appendix B: method descriptions](#appendix-b-method-descriptions)",
        "- [Appendix C: methodology](#appendix-c-methodology)",
        "",
    ]


def _versions(run: Run) -> list[str]:
    out = ["## Comparator versions", ""]
    if not run.versions:
        return out + ["No `versions_replica_*.txt` files in the run directory.", ""]
    # pkg -> version string -> set(groups), and installed count per group
    seen: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    n_rep: dict[str, int] = defaultdict(int)
    for key, pkgs in sorted(run.versions.items()):
        group = key.split("/", 1)[0]
        n_rep[group] += 1
        for name, version, commit in pkgs:
            label = version if version != "NOT INSTALLED" else "not installed"
            if commit:
                label = f"{label} @ `{commit}`"
            seen[name][label].add(group)
    rows: list[list[str]] = []
    n_diff = 0
    for name in sorted(seen, key=str.lower):
        vers = seen[name]
        installed = [v for v in vers if v != "not installed"]
        flag = ""
        if len(installed) > 1:
            flag = "**differs**"
            n_diff += 1
        elif installed and "not installed" in vers:
            flag = "partial"
        cells = []
        for v in sorted(vers):
            cells.append(f"{v} ({', '.join(sorted(vers[v]))})")
        rows.append([f"`{name}`", "; ".join(cells), flag])
    reps = ", ".join(f"{g}: {n}" for g, n in sorted(n_rep.items()))
    out += [
        f"Union of the `versions_replica_*.txt` files (replicas per group: {reps}). "
        f"Each version is followed by the groups that recorded it. "
        f"**differs**: more than one installed version across replicas "
        f"({n_diff} packages); partial: installed in some replicas only.",
        "",
    ]
    out += table(["Package", "Version (groups)", "Flag"], rows, ["l", "l", "l"])
    return out + [""]


def _environment(run: Run) -> list[str]:
    out = ["## Environment and configuration", ""]
    by_group: dict[str, list[Job]] = defaultdict(list)
    for j in run.jobs:
        by_group[j.group].append(j)
    rows = []
    for group in sorted(by_group):
        jobs = by_group[group]

        def vals(fn: Any, jobs: list[Job] = jobs) -> str:
            s = sorted({str(fn(j)) for j in jobs})
            return ", ".join(s)

        env_lines = sorted(
            {
                line
                for k, lines in run.env_lines.items()
                if k.startswith(group + "/")
                for line in lines
            }
        )
        rows.append(
            [
                f"`{group}`",
                str(len(jobs)),
                vals(lambda j: j.header.get("environment", {}).get("python", "?")),
                vals(lambda j: j.header.get("environment", {}).get("torch", "?")),
                vals(lambda j: j.header.get("environment", {}).get("cuda_available", "?")),
                vals(lambda j: j.header.get("config", {}).get("device", "?")),
                vals(lambda j: j.seed),
                vals(lambda j: j.header.get("config", {}).get("epochs", "?")),
                vals(lambda j: j.header.get("config", {}).get("batch_size", "?")),
                vals(lambda j: j.header.get("config", {}).get("lr", "?")),
                vals(lambda j: j.header.get("config", {}).get("warmup_epochs", "-")),
                vals(lambda j: j.header.get("config", {}).get("smoke", "?")),
                "; ".join(md_escape(e) for e in env_lines) or "-",
            ]
        )
    out += table(
        [
            "Group",
            "Jobs",
            "Python",
            "torch",
            "CUDA",
            "Device",
            "Seeds",
            "Epochs",
            "Batch",
            "LR",
            "Warmup",
            "Smoke",
            "Replica env line(s)",
        ],
        rows,
        ["l", "r", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l"],
    )
    stamps = sorted(
        str(j.header.get("timestamp", "")) for j in run.jobs if j.header.get("timestamp")
    )
    if stamps:
        out += ["", f"Job timestamps span {stamps[0]} to {stamps[-1]}."]
    return out + [""]


def _provenance(run: Run) -> list[str]:
    counts = defaultdict(int)
    for r in run.rows:
        counts[r.status] += 1
    n_cpu = sum(1 for r in run.rows if r.cpu_retry)
    n_synth = sum(1 for r in run.rows if r.data.get("SyntheticFallback"))
    alarm = synthetic_alarm(run.rows)
    out = [
        "## Provenance",
        "",
        f"- Tabulated rows: {len(run.rows)} (ok {counts['ok']}, failed {counts['failed']}, "
        f"skipped {counts['skipped']}); per-bin detail rows: {len(run.detail_rows)}.",
        f"- Rows that ran on CPU after a CUDA failure (`{CPU_MARK}` in the tables): {n_cpu}.",
        f"- Rows with `SyntheticFallback` true: {n_synth}, all on synthetic-by-design "
        f"datasets unless listed below.",
        "",
    ]
    if alarm:
        out += [
            f"> **SYNTHETIC FALLBACK ON REAL DATA: {len(alarm)} row(s).** These rows did not "
            "use the real dataset and must not be reported as real-data results:",
            "",
        ]
        out += [f"- {a}" for a in alarm]
    else:
        out += ["Synthetic-fallback check: **OK** (no real dataset fell back to synthetic data)."]
    return out + [""]


def _summary(
    run: Run, index: Mapping[Key, list[Row]], suites_by_group: Mapping[str, set[str]]
) -> list[str]:
    out = ["## Summary", ""]
    rows = []
    tot = defaultdict(int)
    for group in sorted(suites_by_group):
        for suite in sorted(suites_by_group[group]):
            jobs = [j for j in run.jobs if j.group == group and j.suite == suite]
            rs = [r for r in run.rows if r.group == group and r.suite == suite]
            c = defaultdict(int)
            for r in rs:
                c[r.status] += 1
            vals = {
                "jobs": len(jobs),
                "datasets": len({j.dataset for j in jobs}),
                "methods": len({r.method for r in rs}),
                "ok": c["ok"],
                "failed": c["failed"],
                "skipped": c["skipped"],
            }
            for k, v in vals.items():
                tot[k] += v
            rows.append(
                [f"`{group}`", f"[{suite}](#{anchor(f'{suite} ({group})')})"]
                + [str(vals[k]) for k in ("jobs", "datasets", "methods", "ok", "failed", "skipped")]
            )
    rows.append(
        ["**total**", ""]
        + [f"**{tot[k]}**" for k in ("jobs", "datasets", "methods", "ok", "failed", "skipped")]
    )
    out += table(
        ["Group", "Suite", "Jobs", "Datasets", "Methods", "Rows ok", "Failed", "Skipped"],
        rows,
        ["l", "l", "r", "r", "r", "r", "r", "r"],
    )
    return out + [""]


def _parity_section(pairs: Sequence[PairResult]) -> list[str]:
    out = ["## Parity: torchregress vs comparators", ""]
    out += [
        "Pairs come from the explicit rules in Appendix C, evaluated per dataset on the seeds "
        "where both rows are ok. Verdict on the largest relative difference of the seed means "
        f"over the pair's key metrics: exact ≤ {EXACT_TOL:g}, close ≤ {CLOSE_TOL:.0%}, otherwise "
        f"differs. *Within noise*: every key metric has |Δmean| ≤ {Z_NOISE:g} standard errors "
        "(z column). Pair kinds: *deterministic* pairs apply the same algorithm to the same "
        "fitted model or data and should agree to numerical precision; *matched* pairs train "
        "the same architecture / capacity independently, so they should agree within seed "
        "noise; *method* pairs share the method but not the model or training recipe, so a "
        "difference is not a defect by itself.",
        "",
    ]
    tally: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for p in pairs:
        tally[(p.kind, p.suite)][p.verdict] += 1
        tally[(p.kind, p.suite)]["noise:" + p.within_noise] += 1
    verdicts = ("exact", "close", "differs", "n/a", "noise:yes")
    rows = []
    for (kind, suite), c in sorted(tally.items()):
        rows.append(
            [kind, f"`{suite}`"]
            + [str(c[v]) for v in verdicts]
            + [str(sum(n for v, n in c.items() if not v.startswith("noise:")))]
        )
    for kind in KINDS:
        c = defaultdict(int)
        for (k, _), cc in tally.items():
            if k == kind:
                for v, n in cc.items():
                    c[v] += n
        if c:
            rows.append(
                [f"**{kind} total**", ""]
                + [f"**{c[v]}**" for v in verdicts]
                + [f"**{sum(n for v, n in c.items() if not v.startswith('noise:'))}**"]
            )
    out += ["### Verdict counts", ""]
    out += table(
        ["Kind", "Suite", "Exact", "Close", "Differs", "n/a", "Within noise", "Pairs"],
        rows,
        ["l", "l", "r", "r", "r", "r", "r", "r"],
    )
    out += [""]
    for kind in KINDS:
        sub = [p for p in pairs if p.kind == kind]
        if not sub:
            continue
        out += [f"### {kind.capitalize()} pairs", ""]
        rows = []
        for p in sorted(sub, key=lambda p: (p.suite, p.dataset, p.tr, p.comp)):
            metric_cells = []
            for metric in p.rule.metrics:
                if metric in p.values:
                    a, b = p.values[metric]
                    z = p.z(metric)
                    zs = f", z={z:.1f}" if z is not None and math.isfinite(z) else ""
                    metric_cells.append(
                        f"{metric} {fmt(a.mean)} vs {fmt(b.mean)} ({fmt_pct(p.per_metric[metric])}{zs})"
                    )
            v = p.verdict
            rows.append(
                [
                    f"`{p.suite}`",
                    md_escape(p.dataset),
                    f"`{md_escape(p.tr)}`",
                    f"`{md_escape(p.comp)}`",
                    str(p.n_seeds),
                    "<br>".join(metric_cells) or "no common ok seeds",
                    p.within_noise,
                    f"**{v}**" if v == "differs" else v,
                ]
            )
        out += table(
            [
                "Suite",
                "Dataset",
                "torchregress",
                "Comparator",
                "Seeds",
                "Key metrics: tr vs comp (rel. diff, z)",
                "Within noise",
                "Verdict",
            ],
            rows,
            ["l", "l", "l", "l", "r", "l", "l", "l"],
        )
        out += [""]
    return out


def _cell(s: Stat | None, bold: bool, spread: bool) -> str:
    if s is None:
        return "-"
    if s.n == 0:
        return f"non-finite ({s.n_nonfinite})"
    body = fmt(s.mean) if s.n < 2 else fmt_mean_std(s.mean, s.std)
    if bold:
        body = f"**{body}**"
    if s.n < 3:
        body += f" (n={s.n})"
    if s.n_nonfinite:
        body += f" ({s.n_nonfinite} non-finite)"
    if spread:
        body += f" {SPREAD_MARK}"
    return body


def _suite_section(
    group: str,
    suite: str,
    run: Run,
    index: Mapping[Key, list[Row]],
    flagged: list[tuple[str, str, str, str, float]],
) -> list[str]:
    rows = [r for r in run.rows if r.group == group and r.suite == suite]
    cols = suite_metrics(suite, rows)
    mm = main_metric(cols)
    datasets = sorted(
        {r.dataset for r in rows}
        | {j.dataset for j in run.jobs if j.group == group and j.suite == suite}
    )
    out = [f"#### {suite} ({group})", ""]
    dirs = ", ".join(
        f"{c} {'↓' if direction(c) == 'min' else '↑' if direction(c) == 'max' else '|x|↓' if direction(c) == 'abs' else '·'}"
        for c in cols
    )
    out += [
        f"Group `{group}`; {len(datasets)} dataset(s). Columns: {dirs} "
        f"(↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). "
        f"Seed-spread metric: {mm or '-'}.",
        "",
    ]
    for dataset in datasets:
        methods = sorted({r.method for r in rows if r.dataset == dataset})
        stats: dict[str, dict[str, Stat | None]] = {}
        train: dict[str, Stat | None] = {}
        cpu: dict[str, bool] = {}
        problems: list[str] = []
        for method in methods:
            mrows = index[(suite, dataset, method)]
            mrows = [r for r in mrows if r.group == group]
            ok = [r for r in mrows if r.status == "ok"]
            bad = [r for r in mrows if r.status != "ok"]
            if bad:
                by_note: dict[tuple[str, str], list[int]] = defaultdict(list)
                for r in bad:
                    by_note[(r.status, trunc(r.notes))].append(r.seed)
                for (status, note), seeds in sorted(by_note.items()):
                    problems.append(
                        f"- `{md_escape(method)}` ({status}, seed {', '.join(map(str, sorted(seeds)))}): {note}"
                    )
            if not ok:
                continue
            stats[method] = {c: stat(r.data.get(c) for r in ok) for c in cols}
            train[method] = stat(r.data.get("train_s") for r in ok)
            cpu[method] = any(r.cpu_retry for r in ok)
        best: dict[str, float] = {}
        for c in cols:
            d = direction(c)
            if d == "none":
                continue
            cand = [s[c].mean for s in stats.values() if s[c] is not None and s[c].n > 0]
            if len(cand) < 2:
                continue
            if d == "min":
                best[c] = min(cand)
            elif d == "max":
                best[c] = max(cand)
            else:
                best[c] = min(cand, key=abs)
        table_rows = []
        for method in sorted(stats):
            s = stats[method]
            cells = []
            for c in cols:
                sc = s[c]
                is_best = (
                    c in best
                    and sc is not None
                    and sc.n > 0
                    and (abs(sc.mean) if direction(c) == "abs" else sc.mean)
                    == (abs(best[c]) if direction(c) == "abs" else best[c])
                )
                spread = c == mm and sc is not None and sc.rel_std > SPREAD_TOL
                if spread and sc is not None:
                    flagged.append((suite, dataset, method, c, sc.rel_std))
                cells.append(_cell(sc, is_best, spread))
            lib = index[(suite, dataset, method)][0].library
            name = f"`{md_escape(method)}`" + (f" {CPU_MARK}" if cpu[method] else "")
            t = train[method]
            cells.append(fmt(t.mean) if t is not None and t.n else "-")
            table_rows.append([name, md_escape(lib)] + cells)
        out += [f"##### {suite} / {dataset}", ""]
        if table_rows:
            out += table(
                ["Method", "Library"] + cols + ["train_s"],
                table_rows,
                ["l", "l"] + ["r"] * (len(cols) + 1),
            )
        else:
            out += ["_No successful rows._"]
        if any(cpu.values()):
            out += ["", f"{CPU_MARK} ran on CPU after a CUDA failure (cpu retry)."]
        if problems:
            out += ["", "Failed / skipped:", ""] + problems
        out += [""]
    return out


def _spread_section(flagged: Sequence[tuple[str, str, str, str, float]]) -> list[str]:
    out = ["## Seed spread", ""]
    out += [
        f"Cells marked {SPREAD_MARK} have a relative standard deviation (std / |mean|) above "
        f"{SPREAD_TOL:.0%} on the suite's seed-spread metric: rank them with error bars only. "
        f"{len(flagged)} cells flagged.",
        "",
    ]
    per_suite: dict[str, int] = defaultdict(int)
    for s, *_ in flagged:
        per_suite[s] += 1
    if per_suite:
        out += table(
            ["Suite", "Flagged cells"],
            [[f"`{s}`", str(n)] for s, n in sorted(per_suite.items())],
            ["l", "r"],
        )
        out += ["", "Largest 25:", ""]
        top = sorted(flagged, key=lambda f: (-f[4], f[0], f[1], f[2]))[:25]
        out += table(
            ["Suite", "Dataset", "Method", "Metric", "Rel. std"],
            [[f"`{a}`", b, f"`{md_escape(c)}`", d, fmt_pct(e)] for a, b, c, d, e in top],
            ["l", "l", "l", "l", "r"],
        )
    return out + [""]


def _appendix(run: Run, index: Mapping[Key, list[Row]]) -> list[str]:
    out = ["## Appendix A: failure and skip ledger", ""]
    led: dict[tuple[str, str, str, str, str], list[tuple[str, int]]] = defaultdict(list)
    for r in run.rows + run.detail_rows:
        if r.status == "ok":
            continue
        led[(r.status, r.suite, r.method, r.group, trunc(r.notes))].append((r.dataset, r.seed))
    rows = []
    for (status, suite, method, group, note), items in sorted(led.items()):
        ds: dict[str, list[int]] = defaultdict(list)
        for d, s in items:
            ds[d].append(s)
        where = "; ".join(f"{d} ({','.join(map(str, sorted(s)))})" for d, s in sorted(ds.items()))
        rows.append(
            [
                status,
                f"`{suite}`",
                f"`{md_escape(method)}`",
                md_escape(where),
                str(len(items)),
                note,
            ]
        )
    n_f = sum(len(v) for k, v in led.items() if k[0] == "failed")
    n_s = sum(len(v) for k, v in led.items() if k[0] == "skipped")
    out += [f"{n_f} failed and {n_s} skipped rows (dataset (seeds)).", ""]
    if rows:
        out += table(
            ["Status", "Suite", "Method", "Dataset (seeds)", "Rows", "Reason"],
            rows,
            ["l"] * 4 + ["r", "l"],
        )
    out += ["", "## Appendix B: method descriptions", ""]
    out += ["First non-empty note per (suite, method) among ok rows, truncated.", ""]
    desc: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    for r in sorted(run.rows, key=lambda r: (r.suite, r.method, r.group, r.dataset, r.seed)):
        if r.status != "ok" or not r.notes.strip():
            continue
        base = re.sub(r" \[[^\]]+\]$", "", r.method)
        desc[r.suite].setdefault(base, (r.library, trunc(r.notes)))
    for suite in sorted(desc):
        out += [f"### {suite} (methods)", ""]
        out += table(
            ["Method", "Library", "Description"],
            [
                [f"`{md_escape(m)}`", md_escape(lib), n]
                for m, (lib, n) in sorted(desc[suite].items())
            ],
            ["l", "l", "l"],
        )
        out += [""]
    out += _methodology()
    return out


def _methodology() -> list[str]:
    out = [
        "## Appendix C: methodology",
        "",
        "- **Rows.** One row per (suite, dataset, method, seed) from the job JSONs. A row is "
        "*failed* when its `Notes` start with `failed:`, `known-bad:` or `error`, *skipped* "
        "when they start with `skipped:`, otherwise *ok*. Per-bin `distributional_bins` rows "
        "(`.../bin_k`) are counted but not tabulated. For scenario suites the scenario is "
        "appended to the method name (`Method [scenario]`).",
        "- **Means and spread.** Cells are the mean ± sample standard deviation (n−1) over the "
        "ok seeds with a finite value; `(n=k)` marks fewer than 3 seeds and `(k non-finite)` "
        "counts excluded inf/nan values. `train_s` is the mean training time in seconds.",
        f"- **Best value.** Bold marks the best mean per column among ≥2 methods: lowest for "
        f"error-type metrics, highest for {', '.join(sorted(HIGHER_IS_BETTER))}, smallest |x| "
        f"for {', '.join(sorted(ABS_IS_BETTER))}. Coverage, widths and other target-valued "
        "columns are not ranked (see the column legend in each suite).",
        f"- **Seed spread.** {SPREAD_MARK} when std / |mean| > {SPREAD_TOL:.0%} on the suite's "
        "first ranked column.",
        f"- **Parity verdicts.** For each pair and key metric the relative difference "
        f"|a−b| / max(|a|, |b|) of the seed means a, b (seeds where both rows are ok) is "
        f"computed; for log-scale "
        f"metrics ({', '.join(sorted(LOG_SCALE))}) it is |a−b| / max(|b|, 1) (nats). The "
        f"verdict uses the largest value: exact ≤ {EXACT_TOL:g}, close ≤ {CLOSE_TOL:.0%}, "
        f"differs above. z = |a−b| / sqrt(s_a²/n_a + s_b²/n_b); *within noise* when z ≤ "
        f"{Z_NOISE:g} for every key metric.",
        "",
        "### Pairing rules",
        "",
        "Method names are matched exactly (full-regex on the name without the suite prefix); "
        "comparator names are templates filled from the regex groups. No fuzzy matching.",
        "",
    ]
    rows = []
    for rule in PAIR_RULES:
        rows.append(
            [
                ", ".join(f"`{s}`" for s in rule.suites),
                f"`{md_escape(rule.tr)}`",
                ", ".join(f"`{md_escape(c)}`" for c in rule.comps),
                ", ".join(rule.metrics),
                rule.kind,
                md_escape(rule.why),
            ]
        )
    out += table(
        ["Suites", "torchregress (regex)", "Comparators", "Key metrics", "Kind", "Rationale"],
        rows,
        ["l"] * 6,
    )
    return out + [""]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--out", default="PARITY_REPORT.md", type=Path)
    p.add_argument("--run-tag", default="")
    p.add_argument("--harness-ref", default="")
    p.add_argument("--lib-ref", default="")
    p.add_argument("--date", default=dt.datetime.now(dt.UTC).date().isoformat())
    p.add_argument("--preamble", type=Path, default=None, help="Markdown inserted after the header")
    p.add_argument("--note", action="append", default=[], help="header note (repeatable)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run = load_run_dir(args.run_dir)
    if not run.jobs:
        print(f"no job JSONs under {args.run_dir}", file=sys.stderr)
        return 1
    opts = Options(
        run_tag=args.run_tag or args.run_dir.name,
        harness_ref=args.harness_ref or "?",
        lib_ref=args.lib_ref or "?",
        date=args.date,
        preamble=args.preamble.read_text(encoding="utf-8") if args.preamble else "",
        notes=tuple(args.note),
    )
    text = build_report(run, opts)
    args.out.write_text(text, encoding="utf-8")
    print(
        f"wrote {args.out}: {text.count(chr(10))} lines, {len(run.jobs)} jobs, {len(run.rows)} rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
