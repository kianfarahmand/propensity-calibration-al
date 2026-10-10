#!/usr/bin/env python3
"""Merge raw per-run JSON records into one tidy table.

    python scripts/merge_results.py results/raw/smoke results/processed/smoke.csv

One row per (dataset, arm, seed) with the pre-registered outcomes, plus a
companion per-step table so the ESS, lambda and coverage trajectories survive.
Raw records are build artifacts; these tables are repository content.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

RUN_FIELDS = ["dataset", "arm", "arm_name", "seed", "monitor", "controller",
              "randomized", "alpha", "n_steps", "wall_seconds"]
SUMMARY_FIELDS = ["simple_regret", "topk_enrichment_10", "topk_enrichment_50",
                  "discovery_auc", "mean_absolute_deviation", "suppression_rate",
                  "mean_ess", "final_lambda"]
STEP_FIELDS = ["dataset", "arm", "seed", "step", "reading", "ess", "suppressed",
               "lambda", "propensity", "pool_size", "best_y", "policy", "seconds",
               "pool_coverage"]


def load_records(raw_dir: Path) -> list[dict]:
    records = []
    for path in sorted(raw_dir.glob("*.json")):
        with open(path, encoding="utf-8") as fh:
            records.append(json.load(fh))
    return records


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    raw_dir, out_csv = Path(sys.argv[1]), Path(sys.argv[2])
    records = load_records(raw_dir)
    if not records:
        print(f"no records in {raw_dir}")
        return 1
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RUN_FIELDS + SUMMARY_FIELDS)
        writer.writeheader()
        for r in records:
            row = {k: r.get(k) for k in RUN_FIELDS}
            row.update({k: r.get("summary", {}).get(k) for k in SUMMARY_FIELDS})
            writer.writerow(row)

    steps_csv = out_csv.with_name(out_csv.stem + "_steps.csv")
    with open(steps_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=STEP_FIELDS)
        writer.writeheader()
        for r in records:
            truth = {e["step"]: e.get("pool_coverage") for e in r.get("evaluation", [])}
            for s in r["trajectory"]:
                row = {k: s.get(k) for k in STEP_FIELDS if k in s}
                row.update({"dataset": r["dataset"], "arm": r["arm"], "seed": r["seed"],
                            "pool_coverage": truth.get(s["step"])})
                writer.writerow(row)

    print(json.dumps({"runs": len(records), "runs_table": str(out_csv),
                      "steps_table": str(steps_csv)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
