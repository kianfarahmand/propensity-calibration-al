#!/usr/bin/env python3
"""Emit the shard matrix for .github/workflows/experiments.yml.

GitHub Actions caps a job at six hours, so the grid is cut by dataset, arm and
seed block and each shard becomes one job. The block size lives in the grid
config, not here.

    python scripts/make_shards.py configs/grid_eval.yaml > shards.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcal.config import load_yaml  # noqa: E402


def shards(grid: dict) -> list[dict]:
    seeds = grid["seeds"]
    start, count = int(seeds["start"]), int(seeds["count"])
    block = int(grid.get("shard", {}).get("seed_block_size", count)) or count
    out = []
    for dataset in grid["datasets"]:
        for arm in grid["arms"]:
            for lo in range(start, start + count, block):
                hi = min(lo + block, start + count)
                out.append({"dataset": dataset, "arm": int(arm),
                            "seeds": f"{lo}:{hi}",
                            "name": f"{dataset}-a{int(arm):02d}-s{lo}-{hi}"})
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    grid = load_yaml(sys.argv[1])
    print(json.dumps({"include": shards(grid)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
