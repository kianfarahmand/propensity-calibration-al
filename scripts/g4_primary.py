#!/usr/bin/env python3
"""Gate G4: does the controller that consumes the corrected reading win campaigns?

    python scripts/g4_primary.py results/processed/evaluation-g2-greedy.csv \
        results/processed/evaluation-g2-random.csv \
        results/processed/evaluation-g3.csv \
        results/processed/evaluation-g4.csv \
        --out-csv results/processed/g4_primary.csv \
        --out-h4 results/processed/g4_randomization_cost.csv

This is a merge over the committed per-run tables, not a re-run. Every arm is
seeded from (dataset, seed) alone, so the arms measured by S05 (1-4) and S06
(6, 7, 12, 13) are read from their tables rather than dispatched again.

Co-primary outcomes, PREREGISTRATION.md Section 11, three of them:

  simple_regret       at step 150, lower is better
  topk_enrichment     fraction of the pool's true top-k recovered, k = 10 and
                      k = 50, higher is better
  discovery_auc       normalized discovery-curve AUC over 150 steps, higher is
                      better

Section 11 lists top-k enrichment as ONE co-primary outcome reported at two k,
and Section 12 corrects "the three co-primary outcomes" by Holm within each
(comparison, dataset). The frozen plan does not say how the two k collapse into
one outcome. This script takes the conservative reading - the outcome's p-value
is the LARGER of the two, and it counts for the focal arm only if both k point
the same way - and preregistration/deviations.md records that choice, made
before any G4 comparison was computed. Both k are reported in every row.

Pairing is on (dataset, seed) and the interval is a 95% bias-corrected and
accelerated bootstrap over runs with 10000 resamples, never over iterations
(Section 12). The BCa p-value is the achieved significance level of the same
bootstrap: the smallest alpha at which the BCa interval excludes zero, which is
why the interval and the p-value can never disagree.

Gate rule, Section 13: arm 7 beats arms 5, 6, 8 AND 9 on at least 2 of the 3
co-primary outcomes after Holm correction, on at least 2 of 3 datasets. The
script prints it; it does not decide it.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

# (metric column, outcome family, +1 if larger is better)
METRICS = [("simple_regret", "simple_regret", -1),
           ("topk_enrichment_10", "topk_enrichment", +1),
           ("topk_enrichment_50", "topk_enrichment", +1),
           ("discovery_auc", "discovery_auc", +1)]
FAMILIES = ["simple_regret", "topk_enrichment", "discovery_auc"]

ARM_NAMES = {1: "uniform_random", 2: "greedy_ei", 3: "greedy_ucb",
             4: "greedy_maxvar", 5: "boltz_ei_none", 6: "boltz_ei_naive",
             7: "boltz_ei_ipw", 8: "online_calibration", 9: "locbo",
             10: "gp_hedge", 11: "global_scaling", 12: "ipw_no_loo",
             13: "ipw_no_clip"}

# (focal, reference, what the difference isolates)
COMPARISONS = [(7, 5, "the monitor, against randomization alone"),
               (7, 6, "the propensity correction in the control loop"),
               (7, 8, "against online calibration"),
               (7, 9, "against localized online conformal")]
H4 = [(5, 2, "the cost of randomizing the acquisition")]


def read_runs(paths) -> dict[tuple, dict]:
    """-> {(dataset, arm, seed): row}. Refuses to merge a duplicated key."""
    runs: dict[tuple, dict] = {}
    for p in paths:
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    key = (r["dataset"], int(r["arm"]), int(r["seed"]))
                except (TypeError, ValueError, KeyError):
                    continue
                if key in runs:
                    raise SystemExit(f"duplicate run {key} in {p}")
                runs[key] = r
    return runs


def bca(d: np.ndarray, n_resamples: int, seed: int) -> tuple[float, float, float]:
    """95% BCa interval for the mean of d, and the two-sided BCa p-value.

    The p-value is the achieved significance level of the same bootstrap: the
    alpha whose BCa endpoint lands exactly on zero. Inverting the BCa endpoint
    equation at u = F*(0) gives z_alpha = (w - z0) / (1 + a (w - z0)) - z0 with
    w = Phi^-1(u), so no second resampling and no second method is involved.
    """
    n = d.size
    if n < 3 or np.allclose(d, d[0]):
        # A tie at zero is a tie, never a win (S06's note, carried forward).
        return float("nan"), float("nan"), 1.0
    rng = np.random.default_rng(seed)
    theta = float(d.mean())
    boot = d[rng.integers(0, n, size=(n_resamples, n))].mean(axis=1)
    prop_less = float(np.mean(boot < theta))
    if prop_less <= 0.0 or prop_less >= 1.0:
        return float("nan"), float("nan"), 1.0
    z0 = float(norm.ppf(prop_less))
    jack = (d.sum() - d) / (n - 1)
    dev = jack.mean() - jack
    denom = 6.0 * (float((dev ** 2).sum()) ** 1.5)
    a = float((dev ** 3).sum()) / denom if denom > 0 else 0.0

    def endpoint(alpha: float) -> float:
        t = z0 + float(norm.ppf(alpha))
        adj = z0 + t / (1.0 - a * t)
        return float(np.quantile(boot, float(norm.cdf(adj))))

    lo, hi = endpoint(0.025), endpoint(0.975)
    u = float(np.mean(boot < 0.0)) + 0.5 * float(np.mean(boot == 0.0))
    u = min(max(u, 1.0 / (2 * n_resamples)), 1.0 - 1.0 / (2 * n_resamples))
    w = float(norm.ppf(u))
    za = (w - z0) / (1.0 + a * (w - z0)) - z0
    alpha = float(norm.cdf(za))
    return lo, hi, float(2.0 * min(alpha, 1.0 - alpha))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values over the three co-primary outcomes."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m, running, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def compare(runs, dataset, focal, other, isolates, seeds, args) -> list[dict]:
    rows = []
    for metric, family, better in METRICS:
        diffs, f_vals, o_vals = [], [], []
        for s in seeds:
            f = runs.get((dataset, focal, s))
            o = runs.get((dataset, other, s))
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
        lo, hi, p = bca(d, args.resamples,
                        args.seed + focal * 1000 + other * 10
                        + METRICS.index((metric, family, better)))
        mean_d = float(d.mean())
        rows.append({"dataset": dataset, "comparison": f"arm{focal}_vs_arm{other}",
                     "arm": focal, "arm_name": ARM_NAMES.get(focal, f"arm{focal}"),
                     "reference_arm": other,
                     "reference_name": ARM_NAMES.get(other, f"arm{other}"),
                     "isolates": isolates, "metric": metric, "outcome": family,
                     "higher_is_better": int(better > 0), "n_pairs": int(d.size),
                     "mean_arm": float(np.mean(f_vals)),
                     "mean_reference": float(np.mean(o_vals)),
                     "mean_difference": mean_d, "ci_lo": lo, "ci_hi": hi,
                     "p_raw": p, "favors_arm": int(better * mean_d > 0)})
    # One Holm family per (comparison, dataset): the three co-primary outcomes,
    # top-k enrichment entering once at its less favourable k.
    fam_p = {}
    for fam in FAMILIES:
        ps = [r["p_raw"] for r in rows if r["outcome"] == fam]
        if ps:
            fam_p[fam] = max(ps)
    adj = holm(fam_p) if fam_p else {}
    for r in rows:
        members = [m for m in rows if m["outcome"] == r["outcome"]]
        r["outcome_p"] = fam_p.get(r["outcome"], float("nan"))
        r["p_holm"] = adj.get(r["outcome"], float("nan"))
        r["outcome_favors_arm"] = int(all(m["favors_arm"] for m in members))
        r["outcome_wins"] = int(r["outcome_favors_arm"] and r["p_holm"] < args.alpha)
    return rows


def write(path: Path, rows: list[dict]) -> None:
    cols = ["dataset", "comparison", "arm", "arm_name", "reference_arm",
            "reference_name", "isolates", "metric", "outcome",
            "higher_is_better", "n_pairs", "mean_arm", "mean_reference",
            "mean_difference", "ci_lo", "ci_hi", "p_raw", "outcome_p",
            "p_holm", "favors_arm", "outcome_favors_arm", "outcome_wins"]
    floats = {"mean_arm", "mean_reference", "mean_difference", "ci_lo", "ci_hi",
              "p_raw", "outcome_p", "p_holm"}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([f"{r[c]:.6f}" if c in floats else r[c] for c in cols])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs_csv", nargs="+")
    ap.add_argument("--out-csv", default="results/processed/g4_primary.csv")
    ap.add_argument("--out-h4",
                    default="results/processed/g4_randomization_cost.csv")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--expect-rows", type=int, default=0)
    args = ap.parse_args()

    runs = read_runs(args.runs_csv)
    datasets = sorted({k[0] for k in runs})
    arms = sorted({k[1] for k in runs})
    seeds = sorted({k[2] for k in runs})
    # Arms that never draw from the Boltzmann sampler log no c at all, so the
    # check is that no TWO values of c are mixed, not that every run carries one.
    cs = {r.get("c") for r in runs.values()} - {None, ""}
    if len(cs) > 1:
        raise SystemExit(f"several c values in the analysis set: {sorted(cs)}")
    if args.expect_rows and len(runs) != args.expect_rows:
        raise SystemExit(f"{len(runs)} runs, expected {args.expect_rows}")
    print(f"{len(runs)} runs, {len(datasets)} datasets, {len(arms)} arms "
          f"{arms}, {len(seeds)} seeds, c={sorted(cs)}")

    primary, h4 = [], []
    for dataset in datasets:
        for focal, other, isolates in COMPARISONS:
            if focal in arms and other in arms:
                primary += compare(runs, dataset, focal, other, isolates, seeds, args)
        for focal, other, isolates in H4:
            if focal in arms and other in arms:
                h4 += compare(runs, dataset, focal, other, isolates, seeds, args)
    write(Path(args.out_csv), primary)
    write(Path(args.out_h4), h4)

    # The gate, printed per dataset: an outcome counts only if arm 7 beats
    # every one of arms 5, 6, 8 and 9 on it.
    refs = sorted({c[1] for c in COMPARISONS})
    datasets_passing = 0
    for dataset in datasets:
        won = []
        for fam in FAMILIES:
            cells = [r for r in primary if r["dataset"] == dataset
                     and r["outcome"] == fam and r["reference_arm"] in refs]
            beaten = {r["reference_arm"] for r in cells if r["outcome_wins"]}
            if beaten == set(refs):
                won.append(fam)
        datasets_passing += int(len(won) >= 2)
        print(f"  {dataset:<22} outcomes where arm 7 beats all of {refs}: "
              f"{', '.join(won) if won else 'none'}")
    print(f"G4: {datasets_passing} of {len(datasets)} datasets with >=2 of 3 "
          f"co-primary outcomes won against all four references "
          f"(rule needs 2 of 3) -> {Path(args.out_csv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
