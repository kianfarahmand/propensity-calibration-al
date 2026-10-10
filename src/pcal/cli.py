"""Command line interface.

    pcal pool  --dataset matbench_steels
    pcal run   --config configs/smoke.yaml
    pcal run   --config configs/grid_eval.yaml --dataset heusler_magnetic --arm 7 --seeds 0:5

`pool` is the data check of the S04 procedure: build a pool and assert the
pre-registered invariants on real data. `run` executes campaigns and writes one
JSON record per (dataset, arm, seed) into the grid's out_dir.

Nothing here decides anything scientific. Every constant comes from configs/,
every override is echoed into the record it affected, and a run whose arm needs
the tuning constant c refuses to start while c is null.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .config import apply_overrides, arm_config, load_policies, load_yaml
from .data import build_pool, check_pool
from .loop import run_campaign
from .metrics import summarize_run


def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def _seed_list(spec: str | None, grid: dict) -> list[int]:
    if spec:
        if ":" in spec:
            lo, hi = spec.split(":", 1)
            return list(range(int(lo), int(hi)))
        return [int(s) for s in spec.split(",")]
    seeds = grid["seeds"]
    return list(range(int(seeds["start"]), int(seeds["start"]) + int(seeds["count"])))


def cmd_pool(args) -> int:
    report = check_pool(args.dataset, cache_dir=args.cache_dir)
    print(json.dumps(report, indent=2, default=_jsonable))
    return 0 if report["ok"] else 1


def cmd_run(args) -> int:
    grid = load_yaml(args.config)
    policies = apply_overrides(load_policies(), grid.get("overrides"))
    datasets = [args.dataset] if args.dataset else list(grid["datasets"])
    arms = [int(args.arm)] if args.arm else [int(a) for a in grid["arms"]]
    seeds = _seed_list(args.seeds, grid)
    out_dir = Path(args.out or grid["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    n_steps = int(args.n_steps or grid["n_steps"])
    n_init = int(grid["n_init"])
    written, step_times = [], []

    for dataset in datasets:
        pool = build_pool(dataset, cache_dir=args.cache_dir)
        print(json.dumps({"pool": pool.summary()}, default=_jsonable), flush=True)
        for arm_id in arms:
            arm = arm_config(policies, arm_id)
            for seed in seeds:
                started = time.time()
                record = run_campaign(pool, arm, seed, arm["resolved"],
                                      n_steps=n_steps, n_init=n_init)
                record["summary"] = summarize_run(record, pool)
                record["wall_seconds"] = time.time() - started
                per_step = [s["seconds"] for s in record["trajectory"]]
                step_times.extend(per_step)
                name = f"{dataset}__arm{arm_id:02d}__seed{seed:03d}.json"
                path = out_dir / name
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(record, fh, default=_jsonable)
                written.append(str(path))
                print(json.dumps({
                    "dataset": dataset, "arm": arm_id, "arm_name": arm["name"],
                    "seed": seed, "wall_seconds": round(record["wall_seconds"], 2),
                    "median_step_seconds": round(float(np.median(per_step)), 4) if per_step else None,
                    "summary": record["summary"],
                }, default=_jsonable), flush=True)

    if step_times:
        print(json.dumps({
            "runs": len(written),
            "steps": len(step_times),
            "median_step_seconds": round(float(np.median(step_times)), 4),
            "mean_step_seconds": round(float(np.mean(step_times)), 4),
            "max_step_seconds": round(float(np.max(step_times)), 4),
        }), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pcal", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_pool = sub.add_parser("pool", help="build one pool and check the invariants")
    p_pool.add_argument("--dataset", required=True)
    p_pool.add_argument("--cache-dir", default=".cache/pcal")
    p_pool.set_defaults(func=cmd_pool)

    p_run = sub.add_parser("run", help="run campaigns from a grid config")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--dataset", default=None)
    p_run.add_argument("--arm", default=None)
    p_run.add_argument("--seeds", default=None, help="0:5 for a range, or 0,3,7")
    p_run.add_argument("--n-steps", dest="n_steps", default=None, type=int)
    p_run.add_argument("--out", default=None)
    p_run.add_argument("--cache-dir", default=".cache/pcal")
    p_run.set_defaults(func=cmd_run)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
