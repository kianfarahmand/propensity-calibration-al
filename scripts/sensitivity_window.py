#!/usr/bin/env python3
"""POST-HOC: G3 re-measured at estimator settings nobody pre-registered.

    python scripts/sensitivity_window.py \
        --setting frozen,30,10,results/processed/evaluation-g3_steps.csv.gz \
        --setting window60,60,10,results/processed/sens-w60_steps.csv.gz \
        --setting window100,100,10,results/processed/sens-w100_steps.csv.gz \
        --setting floor3,30,3,results/processed/sens-f3_steps.csv.gz \
        --out-csv results/processed/sensitivity_window.csv \
        --out-dir results/figures

Recorded as entry 11 of preregistration/deviations.md, decided after seeing the
results. Nothing here changes a gate verdict: G3 is decided at the frozen
setting - window 30, weights clipped at the 95th percentile, ESS floor 10 - and
stays decided. The question this answers is narrower and fair to ask: the
corrected reading was suppressed on 40 to 61 per cent of window steps, so did
G3 fail because the propensity correction does not repay its variance, or
because thirty steps is too small a sample to find out?

These are re-RUNS, not re-scorings. An earlier attempt rebuilt the estimator
from the committed per-step log and was abandoned when it failed its own
reproduction check: under the primary specification each windowed residual is
recomputed under the CURRENT surrogate with the point held out
(PREREGISTRATION.md 7.1), and the log holds only the value each point had at
its own step, so a longer window cannot be simulated from it. Each setting here
is therefore a dispatch of arms 6 and 7 over the same three datasets and the
same thirty seeds with `--set estimator.window=...` or
`--set estimator.ess_floor=...`, and the comparison is computed by the same
`scripts/g3_error.py` that decided G3, on its own merged table.
"""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLS = ["setting", "window", "ess_floor", "preregistered", "status", "dataset",
        "support", "n_pairs", "mean_deviation_ipw", "mean_deviation_naive",
        "mean_difference", "ci_lo", "ci_hi", "excludes_zero", "favors_ipw",
        "suppression_rate_ipw", "mean_steps_answered_ipw",
        "median_ess_step100_ipw", "steps_table"]
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
    ap.add_argument("--setting", action="append", required=True,
                    help="label,window,ess_floor,steps_table")
    ap.add_argument("--out-csv", default="results/processed/sensitivity_window.csv")
    ap.add_argument("--out-dir", default="results/figures")
    ap.add_argument("--aux-dir", default="results/processed/sensitivity")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--steps", type=int, nargs=2, default=(50, 150))
    args = ap.parse_args()

    aux = Path(args.aux_dir)
    aux.mkdir(parents=True, exist_ok=True)
    settings, rows = [], []
    for spec in args.setting:
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) != 4:
            raise SystemExit(f"--setting wants label,window,floor,path; got {spec!r}")
        label, window, floor, steps_table = parts[0], int(parts[1]), float(parts[2]), parts[3]
        if not Path(steps_table).exists():
            print(f"  {label}: {steps_table} is not in the repository - skipped")
            continue
        err = aux / f"{label}_estimator_error.csv"
        sup = aux / f"{label}_suppression.csv"
        cmd = [sys.executable, "scripts/g3_error.py", steps_table,
               "--out-csv", str(err), "--out-runs", str(aux / f"{label}_runs.csv"),
               "--out-suppression", str(sup),
               "--steps", str(args.steps[0]), str(args.steps[1]),
               "--resamples", str(args.resamples)]
        print(f"\n--- {label} (window {window}, floor {floor:g}): "
              + " ".join(cmd[1:]), flush=True)
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0 or not err.exists():
            raise SystemExit(f"g3_error.py failed on {label}")
        settings.append((label, window, floor))
        error_rows = read(err)
        supp_rows = {(r["dataset"], int(r["arm"])): r for r in read(sup)}
        frozen = (window == 30 and floor == 10.0)
        for r in error_rows:
            if int(r["reference_arm"]) != 6 or int(r["arm"]) != 7:
                continue
            s = supp_rows.get((r["dataset"], 7), {})
            rows.append({
                "setting": label, "window": window, "ess_floor": floor,
                "preregistered": int(frozen),
                "status": ("pre-registered setting, as decided"
                           if frozen else "post-hoc sensitivity"),
                "dataset": r["dataset"], "support": r["support"],
                "n_pairs": int(r["n_pairs"]),
                "mean_deviation_ipw": num(r, "mean_deviation_arm"),
                "mean_deviation_naive": num(r, "mean_deviation_reference"),
                "mean_difference": num(r, "mean_difference"),
                "ci_lo": num(r, "ci_lo"), "ci_hi": num(r, "ci_hi"),
                "excludes_zero": int(r["excludes_zero"]),
                "favors_ipw": int(r["favors_focal"]),
                "suppression_rate_ipw": num(s, "suppression_rate") if s else float("nan"),
                "mean_steps_answered_ipw": num(s, "mean_steps_answered") if s else float("nan"),
                "median_ess_step100_ipw": num(s, "median_ess_step100") if s else float("nan"),
                "steps_table": steps_table})

    if not rows:
        raise SystemExit("no setting had a committed per-step table; nothing written")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    floats = {"mean_deviation_ipw", "mean_deviation_naive", "mean_difference",
              "ci_lo", "ci_hi", "suppression_rate_ipw",
              "mean_steps_answered_ipw", "median_ess_step100_ipw"}
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        for r in rows:
            w.writerow([f"{r[c]:.6f}" if c in floats else r[c] for c in COLS])
    print(f"\n{len(rows)} rows -> {out}")

    # The G3 rule, applied at each setting. It decides nothing here; it is
    # printed so the sensitivity can be read in the units of the gate it is
    # beside. "own" support is the support G3 was decided on.
    print("\nsetting      window  floor  datasets where arm 7 beats arm 6 "
          "(own support; G3 needs 2 of 3)   mean suppression")
    for label, window, floor in settings:
        cells = [r for r in rows if r["setting"] == label and r["support"] == "own"]
        won = [r["dataset"] for r in cells
               if r["excludes_zero"] == 1 and r["favors_ipw"] == 1]
        supp = float(np.mean([r["suppression_rate_ipw"] for r in cells])) if cells else float("nan")
        print(f"{label:<12} {window:>6}  {floor:>5.0f}  {len(won)} of {len(cells)}"
              f"  ({', '.join(won) if won else 'none'})"
              f"{'':>12}{supp:.3f}")

    datasets = sorted({r["dataset"] for r in rows})
    labels = [s[0] for s in settings]
    style()
    fig, axes = plt.subplots(2, len(datasets), figsize=(7.2, 4.2), squeeze=False)
    for j, dataset in enumerate(datasets):
        ax = axes[0][j]
        ax.axhline(0.0, color="#999999", lw=0.6, zorder=0)
        for i, label in enumerate(labels):
            hit = [r for r in rows if r["setting"] == label
                   and r["dataset"] == dataset and r["support"] == "own"]
            if not hit:
                continue
            r = hit[0]
            m, lo, hi = r["mean_difference"], r["ci_lo"], r["ci_hi"]
            colour = "#1f4e9c" if r["preregistered"] else "#d95f02"
            if math.isfinite(lo) and math.isfinite(hi):
                ax.errorbar([i], [m], yerr=[[m - lo], [hi - m]], fmt="o",
                            color=colour, ms=3.4, lw=1.1, capsize=2.0, zorder=3)
            else:
                ax.plot([i], [m], marker="x", color=colour, ms=4, zorder=3)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_title(dataset, fontsize=SIZES[0])
        if j == 0:
            ax.set_ylabel("arm 7 - arm 6\nmean |reading - truth|", fontsize=SIZES[1])
        ax2 = axes[1][j]
        vals = []
        for label in labels:
            hit = [r for r in rows if r["setting"] == label
                   and r["dataset"] == dataset and r["support"] == "own"]
            vals.append(hit[0]["suppression_rate_ipw"] if hit else float("nan"))
        ax2.bar(range(len(labels)), vals,
                color=["#1f4e9c" if lab == "frozen" else "#d95f02" for lab in labels],
                width=0.6)
        ax2.set_ylim(0.0, 1.0)
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=30, ha="right")
        if j == 0:
            ax2.set_ylabel("arm 7 suppression rate", fontsize=SIZES[1])
    fig.suptitle("Post-hoc: G3 re-measured at estimator settings nobody "
                 "pre-registered (blue is the frozen setting)",
                 x=0.02, ha="left", fontsize=SIZES[0] + 1)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig_path = Path(args.out_dir) / "sensitivity_window.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"figure -> {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
