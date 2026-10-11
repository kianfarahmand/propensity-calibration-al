#!/usr/bin/env python3
"""How large an advantage gate G4 rules out.

    python scripts/g4_bounds.py \
        --primary results/processed/g4_primary.csv \
        --chain results/processed/f4_randomization_chain.csv \
        --out results/processed/g4_bounds.csv

A merge over committed tables, never a re-run. Gate G4 asks whether arm 7 beats
arms 5, 6, 8 and 9. It does not. This script reads the same pre-registered BCa
intervals the gate read and reports what they bound: for each dataset and
metric, the largest advantage arm 7 could hold over every one of the four
references at once and still be consistent with the interval. The binding
reference is the one whose interval is tightest from above.

No new test is performed and no new decision rule is applied. The upper
endpoints are the ones already committed in g4_primary.csv.
"""

import argparse
import csv
from pathlib import Path

REFERENCES = ("5", "6", "8", "9")
CHAIN_LINK = "randomize (5 - 2)"
FIELDS = (
    "dataset",
    "metric",
    "higher_is_better",
    "n_references",
    "binding_reference_arm",
    "binding_reference_name",
    "bound_advantage",
    "randomization_cost",
    "bound_covers_cost",
)


def advantage_upper(row: dict) -> float | None:
    """Upper endpoint of arm 7 minus the reference, oriented so larger is better."""
    lo, hi = row["ci_lo"], row["ci_hi"]
    if lo == "nan" or hi == "nan":
        return None
    if row["higher_is_better"] == "1":
        return float(hi)
    return -float(lo)


def read_bounds(primary: Path) -> dict:
    out: dict = {}
    with primary.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["reference_arm"] not in REFERENCES:
                continue
            upper = advantage_upper(row)
            if upper is None:
                continue
            key = (row["dataset"], row["metric"])
            entry = out.setdefault(key, {"hib": row["higher_is_better"], "refs": {}})
            entry["refs"][row["reference_arm"]] = (upper, row["reference_name"])
    return out


def read_costs(chain: Path) -> dict:
    out: dict = {}
    with chain.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["link"] != CHAIN_LINK or row["mean_difference"] == "nan":
                continue
            diff = float(row["mean_difference"])
            cost = -diff if row["higher_is_better"] == "1" else diff
            out[(row["dataset"], row["metric"])] = cost
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--primary", default="results/processed/g4_primary.csv")
    ap.add_argument("--chain", default="results/processed/f4_randomization_chain.csv")
    ap.add_argument("--out", default="results/processed/g4_bounds.csv")
    args = ap.parse_args()

    bounds = read_bounds(Path(args.primary))
    costs = read_costs(Path(args.chain))
    if not bounds:
        raise SystemExit(f"no usable rows in {args.primary}")

    rows = []
    for (dataset, metric), entry in sorted(bounds.items()):
        refs = entry["refs"]
        arm = min(refs, key=lambda a: refs[a][0])
        bound, name = refs[arm]
        cost = costs.get((dataset, metric))
        covers = "" if cost is None else int(bound >= cost)
        rows.append({
            "dataset": dataset,
            "metric": metric,
            "higher_is_better": entry["hib"],
            "n_references": len(refs),
            "binding_reference_arm": arm,
            "binding_reference_name": name,
            "bound_advantage": f"{bound:.6f}",
            "randomization_cost": "" if cost is None else f"{cost:.6f}",
            "bound_covers_cost": covers,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} rows -> {out}")
    for row in rows:
        print(f"  {row['dataset']:22s} {row['metric']:20s} "
              f"bound {row['bound_advantage']:>10s} "
              f"vs arm {row['binding_reference_arm']} "
              f"cost {row['randomization_cost'] or 'na':>10s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
