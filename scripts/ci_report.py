#!/usr/bin/env python3
"""Summarize a merged results table as markdown, for posting back through the API.

    python scripts/ci_report.py results/processed/tuning-c1.0.csv --title "..."

Standard library only: this runs in the collect job, which has no conda
environment. It reports what the dispatcher needs to act on - how long a step
took, how many runs landed, and the per-group means of the pre-registered
outcomes - because the job log and the build artifacts are unreachable from
outside the runner.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import statistics as st
from pathlib import Path

OUTCOMES = ["mean_absolute_deviation", "suppression_rate", "mean_ess",
            "simple_regret", "topk_enrichment_10", "discovery_auc", "wall_seconds"]


def read_csv(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def num(row: dict, key: str):
    try:
        v = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def agg(rows: list[dict], key: str):
    vals = [v for v in (num(r, key) for r in rows) if v is not None]
    return st.mean(vals) if vals else float("nan")


def fmt(x) -> str:
    return "-" if x != x else (f"{x:.4g}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs_csv")
    ap.add_argument("--title", default="experiment run")
    ap.add_argument("--ess-step", type=int, default=100)
    args = ap.parse_args()

    runs_path = Path(args.runs_csv)
    runs = read_csv(runs_path)
    steps_path = runs_path.with_name(runs_path.stem + "_steps.csv")
    if not steps_path.exists():
        steps_path = steps_path.with_suffix(".csv.gz")
    steps = read_csv(steps_path) if steps_path.exists() else []

    out = [f"### {args.title}", "", f"{len(runs)} runs, {len(steps)} logged steps.", ""]

    sec = [v for v in (num(s, "seconds") for s in steps) if v is not None]
    if sec:
        out += ["| step seconds | value |", "|---|---|",
                f"| median | {st.median(sec):.4g} |",
                f"| mean | {st.mean(sec):.4g} |",
                f"| p95 | {sorted(sec)[int(0.95 * (len(sec) - 1))]:.4g} |",
                f"| max | {max(sec):.4g} |", ""]
    wall = [v for v in (num(r, "wall_seconds") for r in runs) if v is not None]
    if wall:
        out += [f"Per-run wall seconds: median {st.median(wall):.4g}, max {max(wall):.4g}.", ""]

    ess_at = {}
    for s in steps:
        if str(s.get("step")) == str(args.ess_step):
            v = num(s, "ess")
            if v is not None:
                ess_at.setdefault((s.get("dataset"), s.get("arm")), []).append(v)

    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        groups.setdefault((r.get("dataset"), r.get("arm"), r.get("arm_name")), []).append(r)

    out += ["| dataset | arm | n | " + " | ".join(OUTCOMES) +
            f" | median ESS @{args.ess_step} |",
            "|---" * (4 + len(OUTCOMES)) + "|"]
    for (ds, arm, name), rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        e = ess_at.get((ds, arm), [])
        out.append(f"| {ds} | {arm} {name} | {len(rows)} | " +
                   " | ".join(fmt(agg(rows, k)) for k in OUTCOMES) +
                   f" | {fmt(st.median(e)) if e else '-'} |")
    out.append("")

    if len(runs) <= 60 and runs:
        out += ["<details><summary>per-run table</summary>", "", "```csv",
                ",".join(runs[0].keys())]
        out += [",".join(str(r.get(k, "")) for k in runs[0].keys()) for r in runs]
        out += ["```", "", "</details>", ""]

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
