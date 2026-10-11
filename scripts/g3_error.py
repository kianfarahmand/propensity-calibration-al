#!/usr/bin/env python3
"""Gate G3: how far each monitor's reading sits from the evaluator's truth.

    python scripts/g3_error.py results/processed/evaluation-g3_steps.csv.gz \
        --out-csv results/processed/g3_estimator_error.csv \
        --out-runs results/processed/g3_estimator_error_runs.csv \
        --out-suppression results/processed/g3_suppression.csv

This is a merge, not a re-run. `reading` is what the monitor told the campaign
at step t, `pool_coverage` is the evaluator's C_t(alpha) over the unlabeled
pool under the same surrogate at the same step, and `suppressed` records
whether the ESS floor let the monitor answer at all (PREREGISTRATION.md
Sections 7 and 11).

Per run the error is the mean absolute deviation of the reading from the truth
over the pre-registered step window, and it is computed two ways:

  own     - over the steps that arm answered, which is what the campaign would
            actually have seen, and what `mean_absolute_deviation` in the
            per-run table carries;
  common  - over the steps where BOTH arms of the comparison answered. A
            monitor that answers less often is not a more accurate monitor, so
            the own-support comparison is between two different step sets and
            the common-support one is not.

Comparisons are paired on (dataset, seed) - every arm at a given seed starts
from the same initialization - and the interval is a 95% bias-corrected and
accelerated bootstrap over runs, never over iterations (Section 12).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import math
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap

ARM_NAMES = {6: "boltz_ei_naive", 7: "boltz_ei_ipw",
             12: "ipw_no_loo", 13: "ipw_no_clip"}

# (focal arm, reference arm, what the difference isolates)
COMPARISONS = [(7, 6, "propensity correction"),
               (7, 12, "ablation: leave-one-out refit"),
               (7, 13, "ablation: weight clipping")]


def _is_true(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def read_steps(path: Path) -> dict[tuple, dict]:
    """-> {(dataset, arm, seed): {step: (reading, ess, answered, pool_coverage)}}."""
    opener = gzip.open if path.suffix == ".gz" else open
    rows: dict[tuple, tuple] = {}
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                key = (r["dataset"], int(r["arm"]), int(r["seed"]), int(r["step"]))
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
            # A suppressed reading is NaN and NaN is not equal to itself, so the
            # answered/not-answered test is made once, here, and never repeated
            # as a float comparison downstream.
            answered = (not _is_true(r.get("suppressed"))) and math.isfinite(reading)
            rows.setdefault(key[:3], {})[key[3]] = (reading, ess, answered, pool)
    return rows


def answered_steps(run: dict, lo: int, hi: int) -> set[int]:
    return {t for t, (_r, _e, ok, _p) in run.items() if lo <= t <= hi and ok}


def deviation(run: dict, steps) -> tuple[float, int]:
    vals = [abs(run[t][0] - run[t][3]) for t in sorted(steps) if t in run]
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(vals)


def bca(values: np.ndarray, n_resamples: int, seed: int) -> tuple[float, float]:
    if values.size < 3 or np.allclose(values, values[0]):
        return float("nan"), float("nan")
    res = bootstrap((values,), np.mean, n_resamples=n_resamples, method="BCa",
                    confidence_level=0.95, random_state=np.random.default_rng(seed))
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("steps_csv", nargs="+")
    ap.add_argument("--out-csv", default="results/processed/g3_estimator_error.csv")
    ap.add_argument("--out-runs", default="results/processed/g3_estimator_error_runs.csv")
    ap.add_argument("--out-suppression", default="results/processed/g3_suppression.csv")
    ap.add_argument("--steps", type=int, nargs=2, default=(50, 150))
    ap.add_argument("--ess-at", type=int, nargs="+", default=(50, 100, 150))
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261010)
    args = ap.parse_args()
    lo, hi = args.steps

    rows: dict[tuple, dict] = {}
    for path in args.steps_csv:
        for key, run in read_steps(Path(path)).items():
            rows.setdefault(key, {}).update(run)
    if not rows:
        print("no usable rows")
        return 1

    datasets = sorted({k[0] for k in rows})
    arms = sorted({k[1] for k in rows})
    seeds = sorted({k[2] for k in rows})
    empty: dict[int, tuple] = {}

    # --- per arm: how often it answered, and at what effective sample size ---
    sup_path = Path(args.out_suppression)
    sup_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sup_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "arm", "arm_name", "n_runs", "step_lo", "step_hi",
                    "suppression_rate", "mean_steps_answered", "mean_deviation_own"]
                   + [f"median_ess_step{t}" for t in args.ess_at])
        for dataset in datasets:
            for arm in arms:
                rates, answered_n, devs = [], [], []
                ess_at = {t: [] for t in args.ess_at}
                for seed in seeds:
                    run = rows.get((dataset, arm, seed), empty)
                    window = [t for t in run if lo <= t <= hi]
                    if not window:
                        continue
                    ok = answered_steps(run, lo, hi)
                    rates.append(1.0 - len(ok) / len(window))
                    answered_n.append(len(ok))
                    d, _n = deviation(run, ok)
                    if math.isfinite(d):
                        devs.append(d)
                    for t in args.ess_at:
                        cell = run.get(t)
                        if cell is not None and math.isfinite(cell[1]):
                            ess_at[t].append(cell[1])
                if not rates:
                    continue
                w.writerow([dataset, arm, ARM_NAMES.get(arm, f"arm{arm}"), len(rates),
                            lo, hi, f"{np.mean(rates):.6f}", f"{np.mean(answered_n):.3f}",
                            f"{np.mean(devs):.6f}" if devs else "nan"]
                           + [f"{np.median(v):.4f}" if v else "nan"
                              for v in (ess_at[t] for t in args.ess_at)])

    # --- paired comparisons, own support and common support ---
    runs_path = Path(args.out_runs)
    per_run: list[tuple] = []
    summary: list[tuple] = []
    for dataset in datasets:
        for focal, other, isolates in COMPARISONS:
            if focal not in arms or other not in arms:
                continue
            for support in ("own", "common"):
                diffs, focal_devs, other_devs, keep = [], [], [], []
                for seed in seeds:
                    f_run = rows.get((dataset, focal, seed), empty)
                    o_run = rows.get((dataset, other, seed), empty)
                    f_ok = answered_steps(f_run, lo, hi)
                    o_ok = answered_steps(o_run, lo, hi)
                    if support == "common":
                        f_steps = o_steps = f_ok & o_ok
                    else:
                        f_steps, o_steps = f_ok, o_ok
                    df, nf = deviation(f_run, f_steps)
                    do, no = deviation(o_run, o_steps)
                    if not (math.isfinite(df) and math.isfinite(do)):
                        continue
                    diffs.append(df - do)
                    focal_devs.append(df)
                    other_devs.append(do)
                    keep.append(seed)
                    per_run.append((dataset, f"arm{focal}_vs_arm{other}", support, seed,
                                    nf, no, df, do, df - do))
                if not diffs:
                    continue
                d = np.array(diffs, dtype=float)
                ci_lo, ci_hi = bca(d, args.resamples, args.seed + focal * 100 + other)
                excludes = int(math.isfinite(ci_lo) and (ci_lo > 0 or ci_hi < 0))
                summary.append((dataset, focal, ARM_NAMES.get(focal, f"arm{focal}"),
                                other, ARM_NAMES.get(other, f"arm{other}"), isolates,
                                support, len(d), lo, hi,
                                float(np.mean(focal_devs)), float(np.mean(other_devs)),
                                float(d.mean()), ci_lo, ci_hi, excludes,
                                int(excludes and d.mean() < 0)))

    with open(runs_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "comparison", "support", "seed",
                    "n_steps_focal", "n_steps_other",
                    "deviation_focal", "deviation_other", "difference"])
        for r in sorted(per_run, key=lambda t: (t[0], t[1], t[2], t[3])):
            w.writerow(list(r[:6]) + [f"{v:.6f}" for v in r[6:]])

    out_csv = Path(args.out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "arm", "arm_name", "reference_arm", "reference_name",
                    "isolates", "support", "n_pairs", "step_lo", "step_hi",
                    "mean_deviation_arm", "mean_deviation_reference",
                    "mean_difference", "ci_lo", "ci_hi", "excludes_zero",
                    "favors_focal"])
        for r in summary:
            w.writerow(list(r[:10]) + [f"{v:.6f}" for v in r[10:15]] + list(r[15:]))

    # The gate rule, Section 13: arm 7 below arm 6 with the paired interval
    # excluding zero on at least 2 of 3 datasets. Printed, not decided here.
    primary = [r for r in summary if r[1] == 7 and r[3] == 6 and r[6] == "own"]
    wins = sum(r[16] for r in primary)
    n_rows = sum(len(v) for v in rows.values())
    print(f"{n_rows} step rows, {len(rows)} runs, {len(datasets)} datasets, arms {arms}")
    for r in primary:
        print(f"  {r[0]:<22} arm7 {r[10]:.4f} vs arm6 {r[11]:.4f} "
              f"diff {r[12]:+.4f} [{r[13]:+.4f}, {r[14]:+.4f}]"
              f"{'  favors arm 7' if r[16] else ''}")
    print(f"G3 primary: arm 7 better on {wins} of {len(primary)} datasets "
          f"(rule needs 2 of 3) -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
