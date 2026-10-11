#!/usr/bin/env python3
"""The arms no gate tests, described rather than tested.

    python scripts/descriptive_arms.py results/processed/evaluation-g2-greedy.csv \
        results/processed/evaluation-g2-random.csv \
        results/processed/evaluation-g3.csv \
        results/processed/evaluation-g4.csv \
        --out-csv results/processed/descriptive_arms.csv

Arms 10 (GP-Hedge) and 11 (one-off global uncertainty scaling) are in the grid
and in no gate; arms 12 and 13 are the two ablations of arm 7, tested by G3 on
estimator error but by nothing on campaign outcome. This script writes one row
per (dataset, arm, metric) with the arm's own mean and, for every arm except 7
itself, the paired difference against arm 7 with a 95% BCa interval.

Nothing here is corrected for multiplicity and nothing here changes a gate
verdict: every row carries status "descriptive", and the manuscript must report
them as observations. The interval is the same paired BCa the gates used -
`bca` and `read_runs` are imported from g4_primary.py, not restated - so a
reader can compare magnitudes with the pre-registered tables without
translating between two conventions.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics as stat
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g4_primary import ARM_NAMES, METRICS, bca, read_runs  # noqa: E402

REFERENCE = 7
COLS = ["dataset", "arm", "arm_name", "metric", "higher_is_better", "n_runs",
        "mean", "sd", "reference_arm", "reference_name", "mean_reference",
        "n_pairs", "mean_difference", "ci_lo", "ci_hi", "p_raw",
        "excludes_zero", "favors_arm", "rank_in_dataset", "status"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs_csv", nargs="+")
    ap.add_argument("--out-csv", default="results/processed/descriptive_arms.csv")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261010)
    args = ap.parse_args()

    runs = read_runs(args.runs_csv)
    datasets = sorted({k[0] for k in runs})
    arms = sorted({k[1] for k in runs})
    seeds = sorted({k[2] for k in runs})

    rows = []
    for dataset in datasets:
        for metric, _family, better in METRICS:
            own = {}
            for arm in arms:
                vals = []
                for s in seeds:
                    r = runs.get((dataset, arm, s))
                    if r is None:
                        continue
                    try:
                        v = float(r[metric])
                    except (TypeError, ValueError, KeyError):
                        continue
                    if math.isfinite(v):
                        vals.append(v)
                if vals:
                    own[arm] = vals
            # Rank 1 is the best arm on this dataset and metric.
            order = sorted(own, key=lambda a: -better * stat.mean(own[a]))
            rank = {a: i + 1 for i, a in enumerate(order)}
            for arm in arms:
                if arm not in own:
                    continue
                vals = own[arm]
                diffs, ref_vals = [], []
                if arm != REFERENCE:
                    for s in seeds:
                        f = runs.get((dataset, arm, s))
                        o = runs.get((dataset, REFERENCE, s))
                        if f is None or o is None:
                            continue
                        try:
                            fv, ov = float(f[metric]), float(o[metric])
                        except (TypeError, ValueError, KeyError):
                            continue
                        if math.isfinite(fv) and math.isfinite(ov):
                            diffs.append(fv - ov)
                            ref_vals.append(ov)
                if diffs:
                    d = np.array(diffs, dtype=float)
                    lo, hi, p = bca(d, args.resamples,
                                    args.seed + arm * 1000 + REFERENCE * 10
                                    + [m[0] for m in METRICS].index(metric))
                    mean_d = float(d.mean())
                    cell = {"reference_arm": REFERENCE,
                            "reference_name": ARM_NAMES.get(REFERENCE, "arm7"),
                            "mean_reference": float(np.mean(ref_vals)),
                            "n_pairs": int(d.size), "mean_difference": mean_d,
                            "ci_lo": lo, "ci_hi": hi, "p_raw": p,
                            "excludes_zero": int(math.isfinite(lo) and math.isfinite(hi)
                                                 and (lo > 0.0 or hi < 0.0)),
                            "favors_arm": int(better * mean_d > 0)}
                else:
                    cell = {"reference_arm": "", "reference_name": "",
                            "mean_reference": float("nan"), "n_pairs": 0,
                            "mean_difference": float("nan"), "ci_lo": float("nan"),
                            "ci_hi": float("nan"), "p_raw": float("nan"),
                            "excludes_zero": "", "favors_arm": ""}
                rows.append({"dataset": dataset, "arm": arm,
                             "arm_name": ARM_NAMES.get(arm, f"arm{arm}"),
                             "metric": metric, "higher_is_better": int(better > 0),
                             "n_runs": len(vals), "mean": float(np.mean(vals)),
                             "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
                             "rank_in_dataset": rank[arm],
                             "status": "descriptive", **cell})

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    floats = {"mean", "sd", "mean_reference", "mean_difference", "ci_lo",
              "ci_hi", "p_raw"}
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        for r in rows:
            w.writerow([f"{r[c]:.6f}" if c in floats else r[c] for c in COLS])

    # The finding S07 flagged: GP-Hedge recovers far more of the pool's true
    # top 50 than any single-acquisition arm while winning no gate.
    print(f"{len(rows)} rows -> {out}")
    for dataset in datasets:
        cells = [r for r in rows if r["dataset"] == dataset
                 and r["metric"] == "topk_enrichment_50"]
        cells.sort(key=lambda r: -r["mean"])
        best = ", ".join(f"arm {r['arm']} {r['arm_name']} {r['mean']:.3f}"
                         for r in cells[:3])
        rest = [r["mean"] for r in cells[1:]]
        print(f"  {dataset:<22} top-50 enrichment, best three: {best} "
              f"(the other {len(rest)} arms span "
              f"{min(rest):.3f}-{max(rest):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
