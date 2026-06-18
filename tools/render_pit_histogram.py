"""PIT histogram visualization for §13.

Retrains NormalizingFlow and NF_Ensemble_K5 on each distributional_bins
dataset using the suite's fit/predict methods, then plots side-by-side
PIT histograms that show the U-shape -> uniform-flat transformation
motivating PROPOSAL #10 (flow ensemble remedy for single-NF overconfidence).

Output
------
results/latest/pit_histogram_ensemble_remedy.png

Methodology
-----------
- Re-runs ``DistributionalBinsSuite.fit`` + ``predict`` for each
  ``(method, dataset)`` pair to obtain the same MC sample arrays that
  the harness previously used to derive the benchmark's
  PIT_Mean/Std/KS/CvM summary statistics.  The benchmark JSON stores
  only those summary statistics; we re-train here to recover the
  underlying raw PIT arrays needed for the histogram figure.
- PIT values per test point = mean(samples <= y_true) — the harness's
  ``compute_pit_values`` dispatches on the ``samples`` key for this.
- 60-epoch training matches the PARITY_REPORT.md benchmark config.
- Figure is a 1-row x 3-column grid: one column per dataset, both
  methods overlaid on each subplot with KS/CvM in the legend.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure harness and suites are importable from repo root
# (mirrors the pattern used by tools/run_all.py).
_repo_root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _repo_root)
sys.path.insert(0, str(Path(_repo_root) / "src"))

import matplotlib  # noqa: E402 — path bootstrap above must precede 3rd-party import

matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from harness.core import set_harness_seed  # noqa: E402
from harness.datasets import load_dataset  # noqa: E402
from harness.metrics import compute_pit_values, pit_uniformity_tests  # noqa: E402
from harness.runners import HarnessConfig  # noqa: E402
from suites._backbones import _carve_calibration_split  # noqa: E402
from suites.tabular.distributional_bins import DistributionalBinsSuite  # noqa: E402

# ---------------------------------------------------------------------------
# Config — must match PARITY_REPORT.md headline run (60 epochs, seed=42).
# ---------------------------------------------------------------------------

_DATASETS = ["california_housing", "diabetes", "synthetic_hetero"]
_METHODS = [("NormalizingFlow", "NF (single)"), ("NF_Ensemble_K5", "NF Ensemble K=5")]
_N_BINS = 20  # 0.05-wide bins — same resolution as PIT KS uses


def _fit_and_predict(
    method: str,
    splits: dict[str, np.ndarray],
    cfg: HarnessConfig,
) -> dict[str, np.ndarray]:
    """One fit+predict call on a pre-split dataset.

    Seeds the harness RNG once before ``fit``.  ``DistributionalBinsSuite.run``
    re-seeds at the start of every method iteration and does *not* re-seed
    before ``predict``; doing the same here preserves the RNG state at the
    end of ``fit`` so MC samples drawn by ``predict`` are bit-identical to
    the ``pit_ensemble_benchmark.json`` rows.
    """
    suite = DistributionalBinsSuite()
    set_harness_seed(cfg.seed)
    model_state = suite.fit(method, splits, cfg)
    return suite.predict(method, model_state, splits["X_test"], cfg)


def _load_and_split(dataset: str, cfg: HarnessConfig) -> dict[str, np.ndarray]:
    """Load dataset + carve calibration split the same way the suite does."""
    splits = load_dataset(dataset, standardize_targets=cfg.standardize_targets)
    return _carve_calibration_split(
        splits,
        cal_frac=DistributionalBinsSuite.CALIBRATION_FRAC,
        seed=cfg.seed,
        smoke=cfg.smoke,
        suite_name="distributional_bins",
    )


def _dataset_subtitle(dataset: str, n_test: int) -> str:
    return {
        "california_housing": f"California Housing (n_test={n_test})",
        "diabetes": f"Diabetes (n_test={n_test})",
        "synthetic_hetero": f"Synthetic Hetero (n_test={n_test})",
    }.get(dataset, f"{dataset} (n_test={n_test})")


def main() -> None:
    cfg = HarnessConfig(
        seed=42,
        epochs=60,
        batch_size=64,
        lr=1e-3,
        device="cpu",
        smoke=False,
        standardize_targets=True,
    )

    bins = np.linspace(0.0, 1.0, _N_BINS + 1)

    fig, axes = plt.subplots(1, len(_DATASETS), figsize=(5 * len(_DATASETS), 4.0))

    palette = {
        "NormalizingFlow": "#d62728",  # red — the under-calibrated baseline
        "NF_Ensemble_K5": "#1f77b4",  # blue — the PROPOSAL #10 remedy
    }

    for col, dataset in enumerate(_DATASETS):
        ax = axes[col]
        print(f"\n=== {dataset} ===")
        splits = _load_and_split(dataset, cfg)
        n_test = int(splits["X_test"].shape[0])

        for method, label in _METHODS:
            preds = _fit_and_predict(method, splits, cfg)
            y_true = np.asarray(splits["y_test"]).reshape(-1)
            pit = compute_pit_values(preds, y_true)
            stats = pit_uniformity_tests(pit)

            print(
                f"  {label:30s}  n={len(pit)}  "
                f"PIT_mean={stats['PIT_Mean']:.3f}  "
                f"PIT_std={stats['PIT_Std']:.3f}  "
                f"KS={stats['PIT_KS']:.3f}  CvM={stats['PIT_CvM']:.2f}"
            )

            ax.hist(
                pit,
                bins=bins,
                density=True,
                alpha=0.55,
                color=palette[method],
                edgecolor="black",
                linewidth=0.4,
                label=(
                    f"{label}\n"
                    f"  KS={stats['PIT_KS']:.3f}, "
                    f"CvM={stats['PIT_CvM']:.2f}, "
                    f"std={stats['PIT_Std']:.3f}"
                ),
            )

        # Ideal-calibration reference: density = 1.0 (uniform on [0, 1])
        ax.axhline(
            1.0,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Uniform (perfect)",
        )

        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("PIT value")
        if col == 0:
            ax.set_ylabel("Density")
        ax.set_title(_dataset_subtitle(dataset, n_test), fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper center", fontsize=8, framealpha=0.95)

    fig.suptitle(
        "PIT histograms: single NormalizingFlow (red) vs NF_Ensemble_K5 (blue)\n"
        "PROPOSAL #10 — Flow ensemble as remedy for single-NF overconfidence",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.93))

    out_path = Path("results/latest/pit_histogram_ensemble_remedy.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nSaved figure → {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
