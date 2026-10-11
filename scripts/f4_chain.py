#!/usr/bin/env python3
"""Figure 4: what randomization costs and what the monitor buys back.

    python scripts/f4_chain.py results/processed/evaluation-g2-greedy.csv \
        results/processed/evaluation-g2-random.csv \
        results/processed/evaluation-g3.csv \
        results/processed/evaluation-g4.csv \
        --out-csv results/processed/f4_randomization_chain.csv \
        --out-dir results/figures

The paper's argument in one panel per dataset. The method needs a randomized
acquisition before it can estimate anything, and the estimate is then fed back
into the campaign, so the honest accounting is a chain of three paired
differences on the same runs:

  arm 5 - arm 2   what randomizing the acquisition costs, before any monitor
  arm 7 - arm 5   what the propensity-corrected monitor buys back on top of it
  arm 7 - arm 2   the net, against the greedy campaign a practitioner would run

Every link is a paired difference over the 30 (dataset, seed) pairs with a 95%
BCa interval from the same bootstrap the gates used - this script imports
`bca` and `read_runs` from g4_primary.py rather than restating them. The first
link is H4 and is also reported, Holm-corrected within its comparison, in
results/processed/g4_randomization_cost.csv; the third link is new here and is
descriptive, since no pre-registered rule tests arm 7 against arm 2.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g4_primary import ARM_NAMES, METRICS, bca, read_runs  # noqa: E402

# (focal, reference, short label, colour)
LINKS = [(5, 2, "randomize (5 - 2)", "#d95f02"),
         (7, 5, "monitor (7 - 5)", "#1f4e9c"),
         (7, 2, "net (7 - 2)", "#4d4d4d")]
METRIC_LABEL = {"simple_regret": "simple regret\n(lower is better)",
                "topk_enrichment_10": "top-10 enrichment",
                "topk_enrichment_50": "top-50 enrichment",
                "discovery_auc": "discovery-curve AUC"}
METRIC_ORDER = ["topk_enrichment_50", "topk_enrichment_10", "discovery_auc",
                "simple_regret"]
SIZES = (8, 7, 6)
COLS = ["dataset", "link", "arm", "arm_name", "reference_arm", "reference_name",
        "isolates", "metric", "higher_is_better", "n_pairs", "mean_arm",
        "mean_reference", "mean_difference", "ci_lo", "ci_hi", "p_raw",
        "excludes_zero", "favors_arm", "status"]
ISOLATES = {(5, 2): "the cost of randomizing the acquisition",
            (7, 5): "what the corrected monitor buys back",
            (7, 2): "the net against a greedy campaign"}
STATUS = {(5, 2): "pre-registered secondary (H4)", (7, 5): "pre-registered (G4)",
          (7, 2): "descriptive"}


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


def chain(runs, dataset, seeds, resamples, seed) -> list[dict]:
    rows = []
    for focal, other, label, _colour in LINKS:
        for metric, _family, better in METRICS:
            f_vals, o_vals, diffs = [], [], []
            for s in seeds:
                f, o = runs.get((dataset, focal, s)), runs.get((dataset, other, s))
                if f is None or o is None:
                    continue
                try:
                    fv, ov = float(f[metric]), float(o[metric])
                except (TypeError, ValueError, KeyError):
                    continue
                if not (math.isfinite(fv) and math.isfinite(ov)):
                    continue
                f_vals.append(fv)
                o_vals.append(ov)
                diffs.append(fv - ov)
            if not diffs:
                continue
            d = np.array(diffs, dtype=float)
            lo, hi, p = bca(d, resamples,
                            seed + focal * 1000 + other * 10
                            + METRIC_ORDER.index(metric))
            mean_d = float(d.mean())
            excludes = int(math.isfinite(lo) and math.isfinite(hi)
                           and (lo > 0.0 or hi < 0.0))
            rows.append({"dataset": dataset, "link": label, "arm": focal,
                         "arm_name": ARM_NAMES.get(focal, f"arm{focal}"),
                         "reference_arm": other,
                         "reference_name": ARM_NAMES.get(other, f"arm{other}"),
                         "isolates": ISOLATES[(focal, other)], "metric": metric,
                         "higher_is_better": int(better > 0), "n_pairs": int(d.size),
                         "mean_arm": float(np.mean(f_vals)),
                         "mean_reference": float(np.mean(o_vals)),
                         "mean_difference": mean_d, "ci_lo": lo, "ci_hi": hi,
                         "p_raw": p, "excludes_zero": excludes,
                         "favors_arm": int(better * mean_d > 0),
                         "status": STATUS[(focal, other)]})
    return rows


def figure(rows, datasets, out: Path) -> None:
    style()
    metrics = [m for m in METRIC_ORDER if any(r["metric"] == m for r in rows)]
    fig, axes = plt.subplots(len(metrics), len(datasets),
                             figsize=(7.2, 1.55 * len(metrics)), squeeze=False)
    for i, metric in enumerate(metrics):
        for j, dataset in enumerate(datasets):
            ax = axes[i][j]
            ax.axvline(0.0, color="#999999", lw=0.6, zorder=0)
            for k, (focal, other, label, colour) in enumerate(LINKS):
                hit = [r for r in rows if r["dataset"] == dataset
                       and r["metric"] == metric and r["arm"] == focal
                       and r["reference_arm"] == other]
                if not hit:
                    continue
                r = hit[0]
                y = len(LINKS) - 1 - k
                m, lo, hi = r["mean_difference"], r["ci_lo"], r["ci_hi"]
                if math.isfinite(lo) and math.isfinite(hi):
                    ax.errorbar([m], [y], xerr=[[m - lo], [hi - m]], fmt="o",
                                color=colour, ms=3.2, lw=1.1, capsize=2.0,
                                zorder=3)
                else:
                    # A degenerate difference: every pair identical, no interval.
                    ax.plot([m], [y], marker="x", color=colour, ms=4,
                            mew=1.0, zorder=3)
            ax.locator_params(axis="x", nbins=4)
            ax.set_yticks(range(len(LINKS)))
            ax.set_yticklabels([lab for _f, _o, lab, _c in LINKS][::-1]
                               if j == 0 else [""] * len(LINKS))
            ax.set_ylim(-0.6, len(LINKS) - 0.4)
            ax.tick_params(axis="y", length=0)
            if i == 0:
                ax.set_title(dataset, fontsize=SIZES[0])
            if j == 0:
                ax.set_ylabel(METRIC_LABEL.get(metric, metric), fontsize=SIZES[1])
            if i == len(metrics) - 1:
                ax.set_xlabel("paired difference (95% BCa)", fontsize=SIZES[1])
    fig.suptitle("What randomization costs, and what the monitor buys back",
                 x=0.02, ha="left", fontsize=SIZES[0] + 1)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs_csv", nargs="+")
    ap.add_argument("--out-csv",
                    default="results/processed/f4_randomization_chain.csv")
    ap.add_argument("--out-dir", default="results/figures")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261010)
    args = ap.parse_args()

    runs = read_runs(args.runs_csv)
    datasets = sorted({k[0] for k in runs})
    seeds = sorted({k[2] for k in runs})
    rows = []
    for dataset in datasets:
        rows += chain(runs, dataset, seeds, args.resamples, args.seed)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    floats = {"mean_arm", "mean_reference", "mean_difference", "ci_lo", "ci_hi",
              "p_raw"}
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        for r in rows:
            w.writerow([f"{r[c]:.6f}" if c in floats else r[c] for c in COLS])

    fig_path = Path(args.out_dir) / "f4_randomization_chain.png"
    figure(rows, datasets, fig_path)

    for dataset in datasets:
        bits = []
        for focal, other, label, _c in LINKS:
            hit = [r for r in rows if r["dataset"] == dataset
                   and r["metric"] == "topk_enrichment_50"
                   and r["arm"] == focal and r["reference_arm"] == other]
            if hit:
                r = hit[0]
                bits.append(f"{label} {r['mean_difference']:+.3f} "
                            f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]")
        print(f"  {dataset:<22} top-50 enrichment: " + "; ".join(bits))
    print(f"{len(rows)} rows -> {out}; figure -> {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
