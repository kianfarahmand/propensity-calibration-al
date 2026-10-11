#!/usr/bin/env python3
"""The analysis set, counted from the committed tables before anything is read off them.

    python scripts/run_inventory.py \
        --runs results/processed/evaluation-g2-greedy.csv \
               results/processed/evaluation-g2-random.csv \
               results/processed/evaluation-g3.csv \
               results/processed/evaluation-g4.csv \
        --steps results/processed/evaluation-g2-greedy_steps.csv.gz \
                results/processed/evaluation-g2-random_steps.csv.gz \
                results/processed/evaluation-g3_steps.csv.gz \
                results/processed/evaluation-g4_steps.csv.gz \
        --out results/processed/run_inventory.csv

Standard library only, so it also runs in a collect job with no conda
environment. It asserts the grid rather than describing it: the full factorial
of 13 arms x 3 datasets x 30 seeds, no run counted twice, one value of c, and a
per-step table covering exactly the runs the per-run tables contain, 150 steps
each. A violated assertion is a non-zero exit, not a warning - the manuscript
states the grid from this table, so the table has to be the grid.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import statistics as st
from pathlib import Path

ARM_NAMES = {1: "uniform_random", 2: "greedy_ei", 3: "greedy_ucb",
             4: "greedy_maxvar", 5: "boltz_ei_none", 6: "boltz_ei_naive",
             7: "boltz_ei_ipw", 8: "online_calibration", 9: "locbo",
             10: "gp_hedge", 11: "global_scaling", 12: "ipw_no_loo",
             13: "ipw_no_clip"}

# Which session measured which table. The arms are seeded from (dataset, seed)
# alone, so an arm measured by an earlier session is read, never re-run.
SESSION_OF = {"evaluation-g2-greedy": "S05", "evaluation-g2-random": "S05",
              "evaluation-g3": "S06", "evaluation-g4": "S07"}

COLS = ["dataset", "arm", "arm_name", "session", "source_table", "n_seeds",
        "seed_min", "seed_max", "seeds_complete", "n_steps_min", "n_steps_max",
        "steps_complete", "c", "pool_size", "wall_seconds_mean",
        "wall_seconds_total"]


def label_of(path: str) -> str:
    return Path(path).name.split(".")[0].replace("_steps", "")


def num(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def read_runs(paths) -> dict:
    runs: dict = {}
    for p in paths:
        label = label_of(p)
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    key = (r["dataset"], int(r["arm"]), int(r["seed"]))
                except (TypeError, ValueError, KeyError):
                    continue
                if key in runs:
                    raise SystemExit(f"duplicate run {key}: {runs[key]['_source']} and {label}")
                r["_source"] = label
                runs[key] = r
    return runs


def read_steps(paths) -> tuple[dict, dict]:
    """-> ({(dataset, arm, seed): n_steps}, {dataset: pool_size})."""
    counts: dict = {}
    pools: dict = {}
    for p in paths:
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    key = (r["dataset"], int(r["arm"]), int(r["seed"]))
                except (TypeError, ValueError, KeyError):
                    continue
                counts[key] = counts.get(key, 0) + 1
                ps = num(r.get("pool_size"))
                if ps is not None:
                    pools[r["dataset"]] = max(pools.get(r["dataset"], 0), int(ps))
    return counts, pools


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--steps", nargs="+", required=True)
    ap.add_argument("--out", default="results/processed/run_inventory.csv")
    ap.add_argument("--expect-runs", type=int, default=1170)
    ap.add_argument("--expect-arms", type=int, default=13)
    ap.add_argument("--expect-datasets", type=int, default=3)
    ap.add_argument("--expect-seeds", type=int, default=30)
    ap.add_argument("--expect-steps", type=int, default=150)
    args = ap.parse_args()

    runs = read_runs(args.runs)
    counts, pools = read_steps(args.steps)

    datasets = sorted({k[0] for k in runs})
    arms = sorted({k[1] for k in runs})
    seeds = sorted({k[2] for k in runs})
    problems = []
    if len(runs) != args.expect_runs:
        problems.append(f"{len(runs)} runs, expected {args.expect_runs}")
    if len(arms) != args.expect_arms:
        problems.append(f"{len(arms)} arms {arms}, expected {args.expect_arms}")
    if len(datasets) != args.expect_datasets:
        problems.append(f"{len(datasets)} datasets {datasets}, expected {args.expect_datasets}")
    if len(seeds) != args.expect_seeds:
        problems.append(f"{len(seeds)} seeds, expected {args.expect_seeds}")
    missing = [(d, a, s) for d in datasets for a in arms for s in seeds
               if (d, a, s) not in runs]
    if missing:
        problems.append(f"{len(missing)} cells of the factorial missing, first {missing[:3]}")

    # Arms that never draw from the Boltzmann sampler log no c, so the check is
    # that no two values of c are mixed, not that every run carries one.
    cs = {r.get("c") for r in runs.values()} - {None, ""}
    if len(cs) > 1:
        problems.append(f"several c values in the analysis set: {sorted(cs)}")

    only_steps = sorted(set(counts) - set(runs))
    only_runs = sorted(set(runs) - set(counts))
    if only_steps:
        problems.append(f"{len(only_steps)} runs in the per-step tables only, "
                        f"first {only_steps[:3]}")
    if only_runs:
        problems.append(f"{len(only_runs)} runs with no logged steps, first {only_runs[:3]}")
    wrong = sorted(k for k, n in counts.items() if n != args.expect_steps)
    if wrong:
        problems.append(f"{len(wrong)} runs not logging {args.expect_steps} steps, "
                        f"first {[(k, counts[k]) for k in wrong[:3]]}")

    rows = []
    for dataset in datasets:
        for arm in arms:
            group = [(s, runs[(dataset, arm, s)]) for s in seeds
                     if (dataset, arm, s) in runs]
            if not group:
                continue
            present = [s for s, _ in group]
            n_steps = [counts.get((dataset, arm, s), 0) for s in present]
            wall = [v for v in (num(r.get("wall_seconds")) for _, r in group) if v is not None]
            source = sorted({r["_source"] for _, r in group})
            c_vals = sorted({r.get("c") for _, r in group} - {None, ""})
            rows.append({
                "dataset": dataset, "arm": arm,
                "arm_name": group[0][1].get("arm_name") or ARM_NAMES.get(arm, f"arm{arm}"),
                "session": "+".join(sorted({SESSION_OF.get(s, "?") for s in source})),
                "source_table": "+".join(source),
                "n_seeds": len(present), "seed_min": min(present), "seed_max": max(present),
                "seeds_complete": int(sorted(present) == list(range(min(present), max(present) + 1))
                                      and len(present) == args.expect_seeds),
                "n_steps_min": min(n_steps), "n_steps_max": max(n_steps),
                "steps_complete": int(all(n == args.expect_steps for n in n_steps)),
                "c": c_vals[0] if c_vals else "",
                "pool_size": pools.get(dataset, ""),
                "wall_seconds_mean": st.mean(wall) if wall else float("nan"),
                "wall_seconds_total": sum(wall) if wall else float("nan"),
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    floats = {"wall_seconds_mean", "wall_seconds_total"}
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        for r in rows:
            w.writerow([f"{r[c]:.3f}" if c in floats else r[c] for c in COLS])

    total_wall = sum(r["wall_seconds_total"] for r in rows
                     if r["wall_seconds_total"] == r["wall_seconds_total"])
    print(f"{len(runs)} runs = {len(arms)} arms x {len(datasets)} datasets x "
          f"{len(seeds)} seeds, c={sorted(cs)}, "
          f"{sum(counts.values())} logged steps, "
          f"{total_wall / 3600.0:.2f} runner-hours -> {out}")
    for dataset in datasets:
        print(f"  {dataset:<22} pool {pools.get(dataset, '?')}")
    if problems:
        for p in problems:
            print(f"  FAIL {p}")
        raise SystemExit("the analysis set is not what the pre-registration says it is")
    print("  the grid is complete and no run is counted twice")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
