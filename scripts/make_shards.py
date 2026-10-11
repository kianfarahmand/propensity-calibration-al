#!/usr/bin/env python3
"""Emit the shard matrix for .github/workflows/experiments.yml.

GitHub Actions caps a job at six hours, so the grid is cut by dataset, arm and
seed block and each shard becomes one job. The block size lives in the grid
config, not here.

    python scripts/make_shards.py configs/grid_eval.yaml
    python scripts/make_shards.py configs/grid_eval.yaml --dataset heusler_magnetic --arm 7
    python scripts/make_shards.py configs/grid_eval.yaml --arm 1,2,3,4

Filtering happens here rather than in the workflow, because a matrix job cannot
skip itself after it has started: an `exit 0` inside a step ends that step, not
the job.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcal.config import load_yaml  # noqa: E402


def parse_arms(spec) -> list[int] | None:
    """--arm 7, or --arm 1,2,3,4 for one dispatch over several arms."""
    if spec in (None, ""):
        return None
    return [int(a) for a in str(spec).replace(" ", "").split(",") if a != ""]


def shards(grid: dict, dataset: str | None = None, arm=None,
           seeds: str | None = None) -> list[dict]:
    spec = grid["seeds"]
    start, count = int(spec["start"]), int(spec["count"])
    block = int(grid.get("shard", {}).get("seed_block_size", count)) or count
    datasets = [d for d in grid["datasets"] if dataset in (None, "", d)]
    wanted = parse_arms(arm)
    arms = [int(a) for a in grid["arms"] if wanted is None or int(a) in wanted]
    if seeds:
        # An explicit seed spec overrides the layout and gives one shard per
        # (dataset, arm). This is what the S05 timing probe dispatches on.
        blocks = [(seeds, seeds.replace(":", "-").replace(",", "."))]
    else:
        blocks = []
        for lo in range(start, start + count, block):
            hi = min(lo + block, start + count)
            blocks.append((f"{lo}:{hi}", f"{lo}-{hi}"))
    out = []
    for name in datasets:
        for a in arms:
            for spec_str, tag in blocks:
                out.append({"dataset": name, "arm": a, "seeds": spec_str,
                            "name": f"{name}-a{a:02d}-s{tag}"})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--arm", default=None)
    parser.add_argument("--seeds", default=None, help="0:1 or 0,3,7; overrides the blocks")
    args = parser.parse_args()
    grid = load_yaml(args.config)
    matrix = shards(grid, dataset=args.dataset or None, arm=args.arm,
                    seeds=args.seeds or None)
    if not matrix:
        print(json.dumps({"include": []}))
        return 1
    print(json.dumps({"include": matrix}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
