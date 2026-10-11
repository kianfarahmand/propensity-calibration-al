#!/usr/bin/env python3
"""Gate G2: coverage at the selected point minus pool-marginal coverage.

    python scripts/g2_gap.py results/processed/evaluation-g2-*_steps.csv.gz \
        --out-csv results/processed/g2_coverage_gap.csv \
        --out-runs results/processed/g2_coverage_gap_runs.csv

This is a merge, not a re-run. Both quantities are already in the per-step
table: `mu`, `sd` and `y` are the pre-observation prediction at the point the
campaign chose and the label that came back, and `pool_coverage` is the
evaluator's C_t(alpha) over the unlabeled pool under the same surrogate at the
same step (PREREGISTRATION.md Sections 1 and 11).

Per run, the gap is the mean over the pre-registered step window of

    1{ |y_t - mu_t| <= q_alpha * sd_t }  -  C_t(alpha)

The first term is a single Bernoulli draw per step, so a per-run gap is a noisy
quantity and only its average over runs is interpretable; the interval is taken
over runs and never over iterations (PREREGISTRATION.md Section 12), with
scipy's bias-corrected and accelerated bootstrap.
"""
from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap, norm

ARM_NAMES = {1: "random", 2: "greedy_ei", 3: "greedy_ucb", 4: "greedy_maxvar"}


def read_steps(path: Path) -> dict[tuple, list[tuple[float, float, float, float]]]:
    """-> {(dataset, arm, seed): [(mu, sd, y, pool_coverage), ...]} inside the window."""
    opener = gzip.open if path.suffix == ".gz" else open
    rows: dict[tuple, list] = {}
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                step = int(r["step"])
                vals = (float(r["mu"]), float(r["sd"]), float(r["y"]),
                        float(r["pool_coverage"]))
            except (TypeError, ValueError, KeyError):
                continue
            rows.setdefault((r["dataset"], int(r["arm"]), int(r["seed"]), step), vals)
    return rows


def per_run_gaps(rows: dict, alpha: float, lo: int, hi: int) -> dict[tuple, dict]:
    q = float(norm.ppf(0.5 * (1.0 + alpha)))
    acc: dict[tuple, dict] = {}
    for (dataset, arm, seed, step), (mu, sd, y, pool) in rows.items():
        if not (lo <= step <= hi):
            continue
        hit = float(abs(y - mu) <= q * max(sd, 1e-300))
        a = acc.setdefault((dataset, arm, seed), {"sel": [], "pool": []})
        a["sel"].append(hit)
        a["pool"].append(pool)
    return {k: {"n_steps": len(v["sel"]),
                "selected_coverage": float(np.mean(v["sel"])),
                "pool_coverage": float(np.mean(v["pool"])),
                "gap": float(np.mean(v["sel"]) - np.mean(v["pool"]))}
            for k, v in acc.items()}


def bca(values: np.ndarray, n_resamples: int, seed: int) -> tuple[float, float]:
    if values.size < 3 or np.allclose(values, values[0]):
        return float("nan"), float("nan")
    res = bootstrap((values,), np.mean, n_resamples=n_resamples, method="BCa",
                    confidence_level=0.95, random_state=np.random.default_rng(seed))
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("steps_csv", nargs="+")
    ap.add_argument("--out-csv", default="results/processed/g2_coverage_gap.csv")
    ap.add_argument("--out-runs", default="results/processed/g2_coverage_gap_runs.csv")
    ap.add_argument("--alpha", type=float, default=0.90)
    ap.add_argument("--steps", type=int, nargs=2, default=(50, 150))
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261010)
    args = ap.parse_args()

    rows: dict[tuple, tuple] = {}
    for path in args.steps_csv:
        rows.update(read_steps(Path(path)))
    runs = per_run_gaps(rows, args.alpha, *args.steps)
    if not runs:
        print("no usable rows")
        return 1

    out_runs = Path(args.out_runs)
    out_runs.parent.mkdir(parents=True, exist_ok=True)
    with open(out_runs, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "arm", "arm_name", "seed", "n_steps",
                    "selected_coverage", "pool_coverage", "gap"])
        for (dataset, arm, seed), v in sorted(runs.items(), key=lambda kv: str(kv[0])):
            w.writerow([dataset, arm, ARM_NAMES.get(arm, f"arm{arm}"), seed, v["n_steps"],
                        f"{v['selected_coverage']:.6f}", f"{v['pool_coverage']:.6f}",
                        f"{v['gap']:.6f}"])

    groups: dict[tuple, list[dict]] = {}
    for (dataset, arm, _seed), v in runs.items():
        groups.setdefault((dataset, arm), []).append(v)

    out_csv = Path(args.out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "arm", "arm_name", "n_runs", "alpha", "step_lo", "step_hi",
                    "mean_selected_coverage", "mean_pool_coverage", "mean_gap",
                    "ci_lo", "ci_hi", "excludes_zero"])
        for (dataset, arm), vs in sorted(groups.items(), key=lambda kv: str(kv[0])):
            gaps = np.array([v["gap"] for v in vs], dtype=float)
            lo, hi = bca(gaps, args.resamples, args.seed + arm)
            w.writerow([dataset, arm, ARM_NAMES.get(arm, f"arm{arm}"), gaps.size,
                        args.alpha, args.steps[0], args.steps[1],
                        f"{np.mean([v['selected_coverage'] for v in vs]):.6f}",
                        f"{np.mean([v['pool_coverage'] for v in vs]):.6f}",
                        f"{gaps.mean():.6f}", f"{lo:.6f}", f"{hi:.6f}",
                        int(np.isfinite(lo) and (lo > 0 or hi < 0))])
    print(f"{len(runs)} runs, {len(groups)} (dataset, arm) cells -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
