#!/usr/bin/env python
"""Corner plots of predicted conditional densities from density_parity sample dumps.

    python tools/plot_density_parity.py DUMP_DIR --dataset mt_jura [--row 0] [--seed 0]

Overlays every method's samples for one test row (standardised target units),
one panel per target pair plus marginals on the diagonal, with the true target
marked.  Writes ``<DUMP_DIR>/<dataset>__seed<seed>__row<row>.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PAIRS = [
    ("torchregress_NSF", "sbi_NSF"),
    ("torchregress_MAF", "sbi_MAF"),
    ("torchregress_MDN", "sbi_MDN"),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--max-dims", type=int, default=4)
    args = ap.parse_args(argv)

    dumps = {}
    for f in sorted(args.dump_dir.glob(f"{args.dataset}__seed{args.seed}__*.npz")):
        dumps[f.stem.split("__", 2)[2]] = np.load(f)
    if not dumps:
        raise SystemExit(f"no dumps for {args.dataset} seed {args.seed} in {args.dump_dir}")
    y_true = next(iter(dumps.values()))["y_true"][args.row]
    D = min(len(y_true), args.max_dims)
    fig, axes = plt.subplots(len(PAIRS), D, figsize=(2.6 * D, 2.6 * len(PAIRS)), squeeze=False)
    for r, (a, b) in enumerate(PAIRS):
        for c in range(D):
            ax = axes[r][c]
            for name, color in ((a, "tab:blue"), (b, "tab:orange")):
                if name not in dumps:
                    continue
                s = dumps[name]["samples"][args.row]
                if c == 0 or D == 1:
                    ax.hist(
                        s[:, 0], bins=40, density=True, histtype="step", color=color, label=name
                    )
                else:
                    ax.scatter(s[:, 0], s[:, c], s=3, alpha=0.3, color=color, label=name)
            if c == 0 or D == 1:
                ax.axvline(y_true[0], color="k", lw=1)
                ax.set_xlabel("y0")
            else:
                ax.plot(y_true[0], y_true[c], "k*", ms=10)
                ax.set_xlabel("y0")
                ax.set_ylabel(f"y{c}")
            if c == 0:
                ax.legend(fontsize=6, loc="upper left")
    fig.suptitle(f"{args.dataset}: test row {args.row} (standardised targets; * = truth)")
    fig.tight_layout()
    out = args.dump_dir / f"{args.dataset}__seed{args.seed}__row{args.row}.png"
    fig.savefig(out, dpi=120)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
