#!/usr/bin/env python3
"""Gate G2 figure: the selection-bias gap each acquisition policy opens.

    python scripts/g2_figures.py \
        --gap results/processed/g2_coverage_gap.csv \
        --runs results/processed/g2_coverage_gap_runs.csv \
        --out-dir results/figures

  g2_gap_by_policy.png  selected-minus-pool coverage at the nominal 0.90 level,
                        steps 50-150, one panel per dataset, one row per arm:
                        the per-seed gaps behind each row and the mean with its
                        95% BCa interval on top, zero drawn.

This figure was first rendered in S05 without a committed script; S08 wrote
this file so that every figure the manuscript uses comes out of the repository.
It reads the committed tables and never recomputes the bootstrap, so the
interval drawn here is the interval in g2_coverage_gap.csv.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ARMS = [(1, "uniform random", "#999999"),
        (2, "greedy EI", "#1f4e9c"),
        (3, "greedy UCB", "#d95f02"),
        (4, "greedy max-variance", "#66a61e")]
SIZES = (8, 7, 6)


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": SIZES[0], "axes.titlesize": SIZES[0],
        "axes.labelsize": SIZES[0], "legend.fontsize": SIZES[1],
        "xtick.labelsize": SIZES[2], "ytick.labelsize": SIZES[2],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "lines.linewidth": 1.1,
        "legend.frameon": False, "axes.titlelocation": "left",
    })


def read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (TypeError, ValueError, KeyError):
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gap", default="results/processed/g2_coverage_gap.csv")
    ap.add_argument("--runs", default="results/processed/g2_coverage_gap_runs.csv")
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args()

    summary = read(Path(args.gap))
    per_run = read(Path(args.runs))
    datasets = sorted({r["dataset"] for r in summary})
    style()
    rng = np.random.default_rng(20261010)

    fig, axes = plt.subplots(1, len(datasets), figsize=(7.2, 2.4),
                             squeeze=False, sharey=True)
    for j, dataset in enumerate(datasets):
        ax = axes[0][j]
        ax.axvline(0.0, color="#999999", lw=0.6, zorder=0)
        for k, (arm, label, colour) in enumerate(ARMS):
            y = len(ARMS) - 1 - k
            gaps = [num(r, "gap") for r in per_run
                    if r["dataset"] == dataset and int(r["arm"]) == arm]
            gaps = [g for g in gaps if math.isfinite(g)]
            if gaps:
                ax.scatter(gaps, y + rng.uniform(-0.16, 0.16, size=len(gaps)),
                           s=2.5, color=colour, alpha=0.30, linewidths=0,
                           zorder=2)
            hit = [r for r in summary if r["dataset"] == dataset
                   and int(r["arm"]) == arm]
            if not hit:
                continue
            r = hit[0]
            m, lo, hi = num(r, "mean_gap"), num(r, "ci_lo"), num(r, "ci_hi")
            ax.errorbar([m], [y], xerr=[[m - lo], [hi - m]], fmt="o",
                        color=colour, ms=3.4, lw=1.2, capsize=2.0, zorder=3)
        ax.set_yticks(range(len(ARMS)))
        # sharey=True means one shared formatter: set the labels once and hide
        # them on the other panels, or the last assignment wins everywhere.
        if j == 0:
            ax.set_yticklabels([lab for _a, lab, _c in ARMS][::-1])
        ax.set_ylim(-0.6, len(ARMS) - 0.4)
        ax.tick_params(axis="y", length=0, labelleft=(j == 0))
        ax.locator_params(axis="x", nbins=5)
        ax.set_title(dataset, fontsize=SIZES[0])
        ax.set_xlabel("selected - pool coverage at 0.90", fontsize=SIZES[1])
    fig.suptitle("The gap an acquisition policy opens between what it sees and "
                 "what is out there", x=0.02, ha="left", fontsize=SIZES[0] + 1)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "g2_gap_by_policy.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"{len(summary)} (dataset, arm) cells, {len(per_run)} runs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
