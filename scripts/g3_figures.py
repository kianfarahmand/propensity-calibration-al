#!/usr/bin/env python3
"""Gate G3 figures: what each monitor reported, and whether it could report.

    python scripts/g3_figures.py results/processed/evaluation-g3_steps.csv.gz \
        --out-dir results/figures

Two figures, both read from the committed per-step table and nothing else:

  g3_estimator_vs_truth.png  the reading against the evaluator's C_t(0.90),
                             one panel per dataset. Suppressed steps are gaps
                             and flagged ticks, never interpolated over - a
                             monitor that did not answer must not look like a
                             monitor that answered correctly.
  g3_ess.png                 effective sample size by step and arm with the
                             floor drawn, because the floor is what decides
                             whether the corrected reading exists at all.
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

ARMS = {6: ("naive monitor", "#d95f02"),
        7: ("propensity-corrected", "#1f4e9c"),
        12: ("no leave-one-out refit", "#7fa6d9"),
        13: ("no weight clipping", "#9d7fb8")}
TRUTH_COLOR = "#4d4d4d"
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


def read_steps(paths) -> dict:
    """-> {(dataset, arm): {step: {"readings", "ess", "pool", "n"}}}.

    Aggregated at read time: the per-step table is a few hundred thousand rows
    and every panel wants the same across-seed summaries.
    """
    agg: dict[tuple, dict] = {}
    for p in paths:
        path = Path(p)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    dataset, arm, step = r["dataset"], int(r["arm"]), int(r["step"])
                    pool = float(r["pool_coverage"])
                except (TypeError, ValueError, KeyError):
                    continue
                try:
                    reading = float(r["reading"])
                except (TypeError, ValueError):
                    reading = float("nan")
                try:
                    ess = float(r["ess"])
                except (TypeError, ValueError):
                    ess = float("nan")
                answered = (str(r.get("suppressed")).strip().lower()
                            not in ("true", "1", "yes")) and math.isfinite(reading)
                cell = agg.setdefault((dataset, arm), {}).setdefault(
                    step, {"readings": [], "ess": [], "pool": [], "n": 0})
                cell["n"] += 1
                cell["pool"].append(pool)
                if answered:
                    cell["readings"].append(reading)
                if math.isfinite(ess):
                    cell["ess"].append(ess)
    return agg


def by_step(agg: dict, dataset: str, arm: int, steps: np.ndarray):
    """Across-seed mean reading over answered runs, and the answered fraction."""
    mean = np.full(steps.size, np.nan)
    frac = np.zeros(steps.size)
    cells = agg.get((dataset, arm), {})
    for i, t in enumerate(steps):
        c = cells.get(int(t))
        if not c or not c["n"]:
            continue
        frac[i] = len(c["readings"]) / c["n"]
        if c["readings"]:
            mean[i] = float(np.mean(c["readings"]))
    return mean, frac


def truth_by_step(agg: dict, dataset: str, arm: int, steps: np.ndarray):
    out = np.full(steps.size, np.nan)
    cells = agg.get((dataset, arm), {})
    for i, t in enumerate(steps):
        c = cells.get(int(t))
        if c and c["pool"]:
            out[i] = float(np.mean(c["pool"]))
    return out


def ess_by_step(agg: dict, dataset: str, arm: int, steps: np.ndarray):
    med = np.full(steps.size, np.nan)
    q1 = np.full(steps.size, np.nan)
    q3 = np.full(steps.size, np.nan)
    cells = agg.get((dataset, arm), {})
    for i, t in enumerate(steps):
        c = cells.get(int(t))
        if c and c["ess"]:
            med[i] = float(np.median(c["ess"]))
            q1[i] = float(np.percentile(c["ess"], 25))
            q3[i] = float(np.percentile(c["ess"], 75))
    return med, q1, q3


def figure_truth(agg, datasets, steps, out_path: Path, window, floor_frac=0.5):
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.1, 2.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, datasets, strict=True):
        ax.axvspan(window[0], window[1], color="0.93", zorder=0, lw=0)
        truth = truth_by_step(agg, dataset, 7, steps)
        ax.plot(steps, truth, color=TRUTH_COLOR, lw=1.4, zorder=1)
        for arm in (6, 7):
            _label, color = ARMS[arm]
            mean, frac = by_step(agg, dataset, arm, steps)
            drawn = np.where(frac > 0, mean, np.nan)
            ax.plot(steps, drawn, color=color, lw=1.0, zorder=3 if arm == 7 else 2,
                    alpha=1.0 if arm == 7 else 0.9)
            flagged = steps[(frac > 0) & (frac < floor_frac)]
            silent = steps[frac == 0]
            for xs, y in ((flagged, 0.722), (silent, 0.710)):
                if xs.size:
                    ax.plot(xs, np.full(xs.size, y), marker="|", ls="none",
                            color=color, ms=3, alpha=0.85)
        ax.set_title(dataset.replace("_", " "))
        ax.set_xlabel("step")
        ax.set_ylim(0.70, 1.005)
        ax.set_yticks([0.7, 0.8, 0.9, 1.0])
        ax.margins(x=0.02)
    axes[0].set_ylabel("coverage at 90% nominal")
    handles = [plt.Line2D([], [], color=TRUTH_COLOR, lw=1.4, label="pool truth"),
               plt.Line2D([], [], color=ARMS[7][1], lw=1.0, label=ARMS[7][0]),
               plt.Line2D([], [], color=ARMS[6][1], lw=1.0, label=ARMS[6][0]),
               plt.Line2D([], [], color=ARMS[7][1], marker="|", ls="none", ms=4,
                          label="most runs suppressed (upper row: no run answered)"),
               plt.Rectangle((0, 0), 1, 1, color="0.93",
                             label=f"scored window, steps {window[0]}-{window[1]}")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.17),
               ncol=3, handlelength=1.4, columnspacing=1.4)
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(out_path)
    plt.close(fig)


def figure_ess(agg, datasets, steps, out_path: Path, floor=10.0):
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.1, 2.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, datasets, strict=True):
        for arm in (6, 7, 12, 13):
            _label, color = ARMS[arm]
            med, q1, q3 = ess_by_step(agg, dataset, arm, steps)
            if not np.any(np.isfinite(med)):
                continue
            ax.plot(steps, med, color=color, lw=1.1 if arm == 7 else 0.8,
                    alpha=1.0 if arm == 7 else 0.8, zorder=3 if arm == 7 else 2)
            if arm == 7:
                ax.fill_between(steps, q1, q3, color=color, alpha=0.16, lw=0, zorder=1)
        ax.axhline(floor, color="#b2182b", ls="--", lw=0.8, zorder=4)
        ax.set_title(dataset.replace("_", " "))
        ax.set_xlabel("step")
        ax.set_yscale("log")
        ax.set_yticks([1, 2, 5, 10, 20, 30])
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.margins(x=0.02)
    axes[0].set_ylabel("effective sample size")
    axes[0].annotate("floor = 10", xy=(steps[len(steps) // 3], floor),
                     xytext=(0, 3), textcoords="offset points",
                     fontsize=SIZES[1], color="#b2182b")
    handles = [plt.Line2D([], [], color=ARMS[a][1], lw=1.1, label=ARMS[a][0])
               for a in (7, 6, 12, 13)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.13),
               ncol=4, handlelength=1.4, columnspacing=1.4)
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("steps_csv", nargs="+")
    ap.add_argument("--out-dir", default="results/figures")
    ap.add_argument("--window", type=int, nargs=2, default=(50, 150))
    args = ap.parse_args()

    style()
    agg = read_steps(args.steps_csv)
    if not agg:
        print("no usable rows")
        return 1
    datasets = sorted({k[0] for k in agg})
    steps = np.array(sorted({t for cells in agg.values() for t in cells}), dtype=int)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_truth(agg, datasets, steps, out_dir / "g3_estimator_vs_truth.png", args.window)
    figure_ess(agg, datasets, steps, out_dir / "g3_ess.png")
    n_cells = sum(len(c) for c in agg.values())
    print(f"{n_cells} (arm, step) cells -> {out_dir}/g3_estimator_vs_truth.png, "
          f"{out_dir}/g3_ess.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
