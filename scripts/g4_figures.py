#!/usr/bin/env python3
"""Gate G4 figures: what the campaigns found, and what the comparisons came to.

    python scripts/g4_figures.py \
        --steps results/processed/evaluation-g2-greedy_steps.csv.gz \
                results/processed/evaluation-g3_steps.csv.gz \
                results/processed/evaluation-g4_steps.csv.gz \
        --primary results/processed/g4_primary.csv \
        --h4 results/processed/g4_randomization_cost.csv \
        --out-dir results/figures

Two figures, both read from committed tables and nothing else:

  g4_discovery_curves.png  best target found so far against step, one panel per
                           dataset, one line per arm with its seeds behind it.
                           The curve runs over QUERIED points only - the ten
                           initial points are identical across arms at a given
                           (dataset, seed), so including them would shift every
                           curve by the same constant and hide early separation.
  g4_paired_effects.png    the paired differences behind the gate, with their
                           95% BCa intervals and the zero line drawn. One panel
                           per (outcome, dataset) because the three outcomes do
                           not share units; the randomization cost (arm 5 minus
                           arm 2, H4) is the bottom row of every panel.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ARMS = [(2, "greedy EI", "#4d4d4d"),
        (5, "Boltzmann EI, no monitor", "#66a61e"),
        (6, "naive monitor", "#d95f02"),
        (7, "propensity-corrected", "#1f4e9c"),
        (8, "online calibration", "#7570b3"),
        (9, "localized conformal", "#e7298a")]
METRIC_LABEL = {"simple_regret": "simple regret (target units)",
                "topk_enrichment_10": "top-10 enrichment",
                "topk_enrichment_50": "top-50 enrichment",
                "discovery_auc": "discovery-curve AUC"}
METRIC_ORDER = ["simple_regret", "topk_enrichment_10", "topk_enrichment_50",
                "discovery_auc"]
SIZES = (8, 7, 6)   # base, annotation, ticks


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


def read_best_so_far(paths, arms) -> dict:
    """-> {(dataset, arm): {seed: array of best-so-far over steps 1..T}}."""
    wanted = set(arms)
    ys: dict[tuple, dict[int, dict[int, float]]] = {}
    for p in paths:
        path = Path(p)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    arm = int(r["arm"])
                    if arm not in wanted:
                        continue
                    step, seed = int(r["step"]), int(r["seed"])
                    y = float(r["y"])
                except (TypeError, ValueError, KeyError):
                    continue
                if math.isfinite(y):
                    ys.setdefault((r["dataset"], arm), {}).setdefault(seed, {})[step] = y
    out: dict[tuple, dict[int, np.ndarray]] = {}
    for key, per_seed in ys.items():
        for seed, by_step in per_seed.items():
            steps = sorted(by_step)
            vals = np.maximum.accumulate([by_step[t] for t in steps])
            out.setdefault(key, {})[seed] = (np.array(steps), vals)
    return out


def discovery_curves(curves: dict, datasets, out: Path) -> None:
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.2, 2.5), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, datasets):
        for arm, label, color in ARMS:
            per_seed = curves.get((dataset, arm))
            if not per_seed:
                continue
            grid = None
            stacked = []
            for _seed, (steps, vals) in sorted(per_seed.items()):
                ax.plot(steps, vals, color=color, alpha=0.07, linewidth=0.3,
                        zorder=1)
                if grid is None:
                    grid = steps
                if steps.size == grid.size:
                    stacked.append(vals)
            if stacked:
                ax.plot(grid, np.mean(stacked, axis=0), color=color,
                        linewidth=1.3, label=label, zorder=3)
        ax.set_title(dataset)
        ax.set_xlabel("step")
    axes[0].set_ylabel("best queried target\n(pool-standardized)")
    axes[-1].legend(loc="lower right", fontsize=SIZES[2] - 0.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def read_effects(paths) -> list[dict]:
    rows = []
    for p in paths:
        if not p or not Path(p).exists():
            continue
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                for k in ("mean_difference", "ci_lo", "ci_hi", "p_holm"):
                    try:
                        r[k] = float(r[k])
                    except (TypeError, ValueError):
                        r[k] = float("nan")
                rows.append(r)
    return rows


def paired_effects(rows: list[dict], datasets, out: Path) -> None:
    order, seen = [], set()
    for r in rows:
        if r["comparison"] not in seen:
            seen.add(r["comparison"])
            order.append(r["comparison"])
    labels = {c: c.replace("arm", "arm ").replace("_vs_", " vs ") for c in order}
    fig, axes = plt.subplots(len(METRIC_ORDER), len(datasets),
                             figsize=(7.2, 7.4), sharey=True)
    for i, metric in enumerate(METRIC_ORDER):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            ax.axvline(0.0, color="#999999", linewidth=0.6, zorder=1)
            for k, comp in enumerate(order):
                cell = [r for r in rows if r["dataset"] == dataset
                        and r["metric"] == metric and r["comparison"] == comp]
                if not cell:
                    continue
                r = cell[0]
                y = len(order) - 1 - k
                sig = math.isfinite(r["p_holm"]) and r["p_holm"] < 0.05
                color = "#1f4e9c" if sig else "#8c8c8c"
                if math.isfinite(r["ci_lo"]):
                    ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=color,
                            linewidth=1.2, zorder=2)
                ax.plot([r["mean_difference"]], [y], "o", color=color,
                        markersize=3.2, zorder=3)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([labels[c] for c in reversed(order)])
            ax.set_ylim(-0.6, len(order) - 0.4)
            if i == 0:
                ax.set_title(dataset)
            if i == len(METRIC_ORDER) - 1:
                ax.set_xlabel("paired difference")
            if j == 0:
                ax.set_ylabel(METRIC_LABEL[metric], fontsize=SIZES[1],
                              color="#333333", labelpad=6)
    handles = [plt.Line2D([], [], color="#1f4e9c", marker="o", markersize=3.2,
                          linewidth=1.2, label="Holm-adjusted p < 0.05"),
               plt.Line2D([], [], color="#8c8c8c", marker="o", markersize=3.2,
                          linewidth=1.2, label="not separated from zero")]
    axes[0, -1].legend(handles=handles, loc="lower right",
                       fontsize=SIZES[2] - 0.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", nargs="+", required=True)
    ap.add_argument("--primary", default="results/processed/g4_primary.csv")
    ap.add_argument("--h4", default="results/processed/g4_randomization_cost.csv")
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args()
    style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_effects([args.primary, args.h4])
    datasets = sorted({r["dataset"] for r in rows}) or None
    curves = read_best_so_far(args.steps, [a for a, _l, _c in ARMS])
    if datasets is None:
        datasets = sorted({k[0] for k in curves})

    discovery_curves(curves, datasets, out_dir / "g4_discovery_curves.png")
    paired_effects(rows, datasets, out_dir / "g4_paired_effects.png")
    print(f"{len(curves)} (dataset, arm) curve sets, {len(rows)} effect rows "
          f"-> {out_dir}/g4_discovery_curves.png, {out_dir}/g4_paired_effects.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
