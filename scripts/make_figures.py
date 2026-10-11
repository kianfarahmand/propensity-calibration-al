#!/usr/bin/env python3
"""One entry point that regenerates every table and figure the manuscript cites.

    python scripts/make_figures.py                      # everything
    python scripts/make_figures.py --only g2 f4         # a subset
    python scripts/make_figures.py --resamples 200      # a fast smoke pass

This is a merge over the committed per-run and per-step tables and never a
re-run of a campaign: `results/processed/evaluation-*.csv` and their
`_steps.csv.gz` companions are the inputs, and every number in the paper comes
out of this script. The arguments below are the ones the manuscript's numbers
were produced with - a step whose arguments change is a step whose output
changes, which is why they live here rather than in a shell history.

Each step reports whether its outputs came out NEW, UNCHANGED or CHANGED
against what is already committed. On a clean checkout of a finished session
every output should be UNCHANGED: the bootstraps are seeded, so a CHANGED row
means the inputs or the code moved, and is worth stopping for.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

RUNS = ["evaluation-g2-greedy.csv", "evaluation-g2-random.csv",
        "evaluation-g3.csv", "evaluation-g4.csv"]
G2_STEPS = ["evaluation-g2-greedy_steps.csv.gz", "evaluation-g2-random_steps.csv.gz"]
G3_STEPS = ["evaluation-g3_steps.csv.gz"]
G4_STEPS = ["evaluation-g2-greedy_steps.csv.gz", "evaluation-g3_steps.csv.gz",
            "evaluation-g4_steps.csv.gz"]
ALL_STEPS = ["evaluation-g2-greedy_steps.csv.gz", "evaluation-g2-random_steps.csv.gz",
             "evaluation-g3_steps.csv.gz", "evaluation-g4_steps.csv.gz"]
EXPECT_RUNS = 1170


def digest(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plan(args) -> list[tuple[str, list[str], list[str]]]:
    p, f = args.processed, args.figures
    runs = [f"{p}/{n}" for n in RUNS]
    resamples = ["--resamples", str(args.resamples)]
    return [
        ("inventory",
         ["scripts/run_inventory.py", "--runs", *runs,
          "--steps", *[f"{p}/{n}" for n in ALL_STEPS],
          "--out", f"{p}/run_inventory.csv",
          "--expect-runs", str(args.expect_runs)],
         [f"{p}/run_inventory.csv"]),
        ("g2",
         ["scripts/g2_gap.py", *[f"{p}/{n}" for n in G2_STEPS],
          "--out-csv", f"{p}/g2_coverage_gap.csv",
          "--out-runs", f"{p}/g2_coverage_gap_runs.csv",
          "--alpha", "0.90", "--steps", "50", "150", *resamples],
         [f"{p}/g2_coverage_gap.csv", f"{p}/g2_coverage_gap_runs.csv"]),
        ("g2-figure",
         ["scripts/g2_figures.py", "--gap", f"{p}/g2_coverage_gap.csv",
          "--runs", f"{p}/g2_coverage_gap_runs.csv", "--out-dir", f],
         [f"{f}/g2_gap_by_policy.png"]),
        ("g3",
         ["scripts/g3_error.py", *[f"{p}/{n}" for n in G3_STEPS],
          "--out-csv", f"{p}/g3_estimator_error.csv",
          "--out-runs", f"{p}/g3_estimator_error_runs.csv",
          "--out-suppression", f"{p}/g3_suppression.csv",
          "--steps", "50", "150", *resamples],
         [f"{p}/g3_estimator_error.csv", f"{p}/g3_estimator_error_runs.csv",
          f"{p}/g3_suppression.csv"]),
        ("g3-figures",
         ["scripts/g3_figures.py", *[f"{p}/{n}" for n in G3_STEPS],
          "--out-dir", f, "--window", "50", "150"],
         [f"{f}/g3_estimator_vs_truth.png", f"{f}/g3_ess.png"]),
        ("g4",
         ["scripts/g4_primary.py", *runs,
          "--out-csv", f"{p}/g4_primary.csv",
          "--out-h4", f"{p}/g4_randomization_cost.csv",
          "--alpha", "0.05", *resamples,
          "--expect-rows", str(args.expect_runs)],
         [f"{p}/g4_primary.csv", f"{p}/g4_randomization_cost.csv"]),
        ("g4-figures",
         ["scripts/g4_figures.py", "--steps", *[f"{p}/{n}" for n in G4_STEPS],
          "--primary", f"{p}/g4_primary.csv",
          "--h4", f"{p}/g4_randomization_cost.csv", "--out-dir", f],
         [f"{f}/g4_discovery_curves.png", f"{f}/g4_paired_effects.png"]),
        ("f4",
         ["scripts/f4_chain.py", *runs,
          "--out-csv", f"{p}/f4_randomization_chain.csv", "--out-dir", f,
          *resamples],
         [f"{p}/f4_randomization_chain.csv",
          f"{f}/f4_randomization_chain.png"]),
        ("sensitivity",
         ["scripts/sensitivity_window.py", f"{p}/evaluation-g3_steps.csv.gz",
          "--windows", "30", "60", "100", "--floors", "10", "5", "3", "0",
          "--out-csv", f"{p}/sensitivity_window.csv", "--out-dir", f,
          *resamples],
         [f"{p}/sensitivity_window.csv", f"{f}/sensitivity_window.png"]),
        ("descriptive",
         ["scripts/descriptive_arms.py", *runs,
          "--out-csv", f"{p}/descriptive_arms.csv", *resamples],
         [f"{p}/descriptive_arms.csv"]),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--processed", default="results/processed")
    ap.add_argument("--figures", default="results/figures")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--expect-runs", type=int, default=EXPECT_RUNS)
    ap.add_argument("--only", nargs="+", default=None,
                    help="run only these step names")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    steps = plan(args)
    if args.list:
        for name, argv, outs in steps:
            print(f"{name}\n    python {' '.join(argv)}\n    -> {', '.join(outs)}")
        return 0
    if args.only:
        unknown = set(args.only) - {n for n, _a, _o in steps}
        if unknown:
            raise SystemExit(f"unknown step(s): {sorted(unknown)}")
        steps = [s for s in steps if s[0] in args.only]

    Path(args.figures).mkdir(parents=True, exist_ok=True)
    failures, report = [], []
    for name, argv, outs in steps:
        before = {o: digest(Path(o)) for o in outs}
        print(f"\n=== {name}: python {' '.join(argv)}", flush=True)
        t0 = time.time()
        proc = subprocess.run([sys.executable, *argv], check=False)
        took = time.time() - t0
        states = []
        for o in outs:
            after = digest(Path(o))
            if after is None:
                states.append(f"{Path(o).name} MISSING")
            elif before[o] is None:
                states.append(f"{Path(o).name} NEW")
            elif before[o] == after:
                states.append(f"{Path(o).name} UNCHANGED")
            else:
                states.append(f"{Path(o).name} CHANGED")
        if proc.returncode != 0 or any(s.endswith("MISSING") for s in states):
            failures.append(name)
        report.append((name, proc.returncode, took, states))
        print(f"=== {name}: exit {proc.returncode}, {took:.1f}s, "
              + "; ".join(states), flush=True)

    print("\n| step | exit | seconds | outputs |")
    print("|---|---|---|---|")
    for name, code, took, states in report:
        print(f"| {name} | {code} | {took:.1f} | " + "<br>".join(states) + " |")
    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        return 1
    print("\nevery table and figure regenerated from the committed tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
