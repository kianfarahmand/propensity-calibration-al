#!/usr/bin/env python3
"""POST-HOC: what a longer window or a lower ESS floor would have done to G3.

    python scripts/sensitivity_window.py results/processed/evaluation-g3_steps.csv.gz \
        --windows 30 60 100 --floors 10 5 3 0 \
        --out-csv results/processed/sensitivity_window.csv \
        --out-dir results/figures

Decided after seeing the results and recorded as entry 11 of
preregistration/deviations.md. Nothing here changes a gate verdict: G3 is
decided at the frozen setting (window 30, ESS floor 10, weights clipped at the
95th percentile) and stays decided. This script exists because the frozen
setting suppressed the corrected reading on 40-61 per cent of window steps, and
a reader is entitled to ask whether the failure is a property of the estimator
or of that one sample-size choice.

How it works, and what it cannot do. The estimator is rebuilt from the
committed per-step log rather than re-run: each logged step carries its
residual |y - mu| / sd, its propensity, its pool size and the controller's
lambda, which is everything the self-normalized inverse-propensity estimator
consumes. Re-scoring at a longer window is therefore exact in the weights and
in the ESS, and approximate in one respect: under the primary specification the
leave-one-out refit recomputes each window point's mu and sd under the CURRENT
surrogate at every read, and the log holds only the value each point had at its
own step. A point that a 100-step window keeps for seventy more steps is scored
here with its original residual. The reproduction check below bounds how much
that matters at the frozen window, where the two coincide.

Reproduction check, run before any sensitivity number is written: at window 30
with the frozen floor and clipping, the rebuilt reading must match the logged
reading. Four conventions are tried (the window including or excluding the
current step, lambda taken at the current or the previous step) and the one
that reproduces the log is used; if none reproduces it to 1e-6 on 99 per cent
of non-suppressed steps, the script exits non-zero and writes nothing.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g4_primary import bca  # noqa: E402

try:
    from pcal.metrics import z_quantile
except Exception:  # pragma: no cover - the package is installed in CI
    from scipy.stats import norm

    def z_quantile(alpha: float) -> float:
        return float(norm.ppf(0.5 + alpha / 2.0))

NAIVE_ARM, IPW_ARM = 6, 7
CLIP = 95.0
COLS = ["dataset", "arm", "arm_name", "window", "ess_floor", "clip_percentile",
        "n_runs", "suppression_rate", "mean_steps_answered", "median_ess_step100",
        "mean_deviation_own", "n_pairs_common", "mean_difference_common",
        "ci_lo", "ci_hi", "excludes_zero", "favors_ipw", "preregistered",
        "status"]
SIZES = (8, 7, 6)


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": SIZES[0], "axes.titlesize": SIZES[0],
        "axes.labelsize": SIZES[0], "legend.fontsize": SIZES[1],
        "xtick.labelsize": SIZES[2], "ytick.labelsize": SIZES[2],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "lines.linewidth": 1.1,
        "legend.frameon": False, "axes.titlelocation": "left",
    })


def read_log(paths, arms) -> dict:
    """-> {(dataset, arm, seed): dict of per-step arrays, ordered by step}."""
    raw: dict = {}
    for p in paths:
        opener = gzip.open if str(p).endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    arm = int(r["arm"])
                    if arm not in arms:
                        continue
                    key = (r["dataset"], arm, int(r["seed"]))
                    raw.setdefault(key, []).append(r)
                except (TypeError, ValueError, KeyError):
                    continue
    out = {}
    for key, rows in raw.items():
        rows.sort(key=lambda r: int(r["step"]))

        def col(name):
            vals = []
            for r in rows:
                try:
                    vals.append(float(r[name]))
                except (TypeError, ValueError, KeyError):
                    vals.append(float("nan"))
            return np.asarray(vals, dtype=float)

        mu, sd, y = col("mu"), col("sd"), col("y")
        out[key] = {
            "step": np.asarray([int(r["step"]) for r in rows]),
            "resid": np.abs(y - mu) / np.maximum(sd, 1e-300),
            "lam": col("lambda"), "propensity": col("propensity"),
            "pool_size": col("pool_size"), "pool_coverage": col("pool_coverage"),
            "reading": col("reading"), "ess": col("ess"),
            "suppressed": np.asarray([str(r["suppressed"]).lower() in ("true", "1")
                                      for r in rows]),
        }
    return out


def rebuild(run: dict, window: int, kind: str, q: float,
            inclusive: bool, lam_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """-> (estimate, ess) per step, before any floor is applied."""
    n = run["step"].size
    est = np.full(n, np.nan)
    ess = np.zeros(n)
    w_all = 1.0 / (np.maximum(run["pool_size"], 1.0)
                   * np.maximum(run["propensity"], 1e-300))
    for t in range(n):
        hi = t + 1 if inclusive else t
        lo = max(0, hi - window)
        if hi <= lo:
            continue
        lam_i = max(0, t - lam_lag)
        lam = run["lam"][lam_i]
        if not math.isfinite(lam):
            lam = 1.0
        z = (run["resid"][lo:hi] <= q * lam).astype(float)
        if kind == "naive":
            est[t] = float(np.mean(z))
            ess[t] = float(z.size)
            continue
        w = w_all[lo:hi].copy()
        if np.any(~np.isfinite(w)):
            continue
        if CLIP < 100.0 and w.size:
            w = np.minimum(w, float(np.percentile(w, CLIP)))
        total = float(np.sum(w))
        sq = float(np.sum(w ** 2))
        if total <= 0 or sq <= 0:
            continue
        est[t] = float(np.dot(w, z) / total)
        ess[t] = total ** 2 / sq
    return est, ess


def agreement(runs, arm, kind, q, window, floor, inclusive, lam_lag):
    """Fraction of logged, non-suppressed readings the rebuild reproduces."""
    hit = total = 0
    for (dataset, a, seed), run in runs.items():
        if a != arm:
            continue
        est, ess = rebuild(run, window, kind, q, inclusive, lam_lag)
        ok = (~run["suppressed"]) & np.isfinite(run["reading"])
        if not ok.any():
            continue
        live = np.where(ok, est, np.nan)
        diff = np.abs(live[ok] - run["reading"][ok])
        hit += int(np.sum(diff <= 1e-6))
        total += int(diff.size)
    return (hit / total if total else 0.0), total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("steps_csv", nargs="+")
    ap.add_argument("--windows", type=int, nargs="+", default=(30, 60, 100))
    ap.add_argument("--floors", type=float, nargs="+", default=(10.0, 5.0, 3.0, 0.0))
    ap.add_argument("--alpha", type=float, default=0.90)
    ap.add_argument("--steps", type=int, nargs=2, default=(50, 150))
    ap.add_argument("--out-csv", default="results/processed/sensitivity_window.csv")
    ap.add_argument("--out-dir", default="results/figures")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--min-agreement", type=float, default=0.99)
    args = ap.parse_args()

    q = z_quantile(args.alpha)
    runs = read_log(args.steps_csv, {NAIVE_ARM, IPW_ARM})
    datasets = sorted({k[0] for k in runs})
    seeds = sorted({k[2] for k in runs})
    print(f"{len(runs)} runs, arms {NAIVE_ARM} and {IPW_ARM}, "
          f"{len(datasets)} datasets, {len(seeds)} seeds, q(0.90)={q:.6f}")

    best, best_rate = None, -1.0
    for inclusive in (True, False):
        for lam_lag in (0, 1):
            rate, n = agreement(runs, NAIVE_ARM, "naive", q, 30, 10.0,
                                inclusive, lam_lag)
            print(f"  reproduction, naive arm {NAIVE_ARM}: "
                  f"window {'includes' if inclusive else 'excludes'} step t, "
                  f"lambda lag {lam_lag}: {rate:.4f} of {n} readings")
            if rate > best_rate:
                best, best_rate = (inclusive, lam_lag), rate
    inclusive, lam_lag = best
    ipw_rate, ipw_n = agreement(runs, IPW_ARM, "ipw", q, 30, 10.0, inclusive, lam_lag)
    print(f"  reproduction, ipw arm {IPW_ARM} at the chosen convention: "
          f"{ipw_rate:.4f} of {ipw_n} readings")
    if best_rate < args.min_agreement or ipw_rate < args.min_agreement:
        raise SystemExit("the rebuilt estimator does not reproduce the committed "
                         "log; no sensitivity number is trustworthy, so none is "
                         "written")

    lo_step, hi_step = args.steps
    rows = []
    for window in args.windows:
        cache = {}
        for key, run in runs.items():
            cache[key] = rebuild(run, window, "naive" if key[1] == NAIVE_ARM
                                 else "ipw", q, inclusive, lam_lag)
        for floor in args.floors:
            for dataset in datasets:
                per_arm = {}
                for arm in (NAIVE_ARM, IPW_ARM):
                    dev_own, supp, answered, ess100, per_seed = [], [], [], [], {}
                    for seed in seeds:
                        key = (dataset, arm, seed)
                        if key not in runs:
                            continue
                        run = runs[key]
                        est, ess = cache[key]
                        inside = (run["step"] >= lo_step) & (run["step"] <= hi_step)
                        live = inside & np.isfinite(est) & (ess >= floor)
                        supp.append(1.0 - live.sum() / max(inside.sum(), 1))
                        answered.append(float(live.sum()))
                        at100 = np.where(run["step"] == 100)[0]
                        if at100.size:
                            ess100.append(float(ess[at100[0]]))
                        if live.any():
                            d = np.abs(est[live] - run["pool_coverage"][live])
                            d = d[np.isfinite(d)]
                            if d.size:
                                per_seed[seed] = float(np.mean(d))
                                dev_own.append(per_seed[seed])
                    per_arm[arm] = {"per_seed": per_seed,
                                    "suppression": float(np.mean(supp)) if supp else float("nan"),
                                    "answered": float(np.mean(answered)) if answered else float("nan"),
                                    "ess100": float(np.median(ess100)) if ess100 else float("nan"),
                                    "dev": float(np.mean(dev_own)) if dev_own else float("nan"),
                                    "n": len(per_seed)}
                common = sorted(set(per_arm[IPW_ARM]["per_seed"])
                                & set(per_arm[NAIVE_ARM]["per_seed"]))
                if common:
                    d = np.array([per_arm[IPW_ARM]["per_seed"][s]
                                  - per_arm[NAIVE_ARM]["per_seed"][s] for s in common])
                    ci_lo, ci_hi, _p = bca(d, args.resamples,
                                           args.seed + window * 10 + int(floor))
                    mean_d = float(d.mean())
                else:
                    d = np.array([])
                    ci_lo = ci_hi = mean_d = float("nan")
                for arm in (NAIVE_ARM, IPW_ARM):
                    a = per_arm[arm]
                    is_ipw = arm == IPW_ARM
                    rows.append({
                        "dataset": dataset, "arm": arm,
                        "arm_name": "boltz_ei_ipw" if is_ipw else "boltz_ei_naive",
                        "window": window, "ess_floor": floor, "clip_percentile": CLIP,
                        "n_runs": a["n"], "suppression_rate": a["suppression"],
                        "mean_steps_answered": a["answered"],
                        "median_ess_step100": a["ess100"],
                        "mean_deviation_own": a["dev"],
                        "n_pairs_common": int(len(common)) if is_ipw else "",
                        "mean_difference_common": mean_d if is_ipw else "",
                        "ci_lo": ci_lo if is_ipw else "",
                        "ci_hi": ci_hi if is_ipw else "",
                        "excludes_zero": (int(math.isfinite(ci_lo) and math.isfinite(ci_hi)
                                              and (ci_lo > 0 or ci_hi < 0))
                                          if is_ipw else ""),
                        "favors_ipw": (int(math.isfinite(mean_d) and mean_d < 0)
                                       if is_ipw else ""),
                        "preregistered": int(window == 30 and floor == 10.0),
                        "status": ("pre-registered setting, re-scored"
                                   if window == 30 and floor == 10.0
                                   else "post-hoc sensitivity")})

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    floats = {"suppression_rate", "mean_steps_answered", "median_ess_step100",
              "mean_deviation_own", "mean_difference_common", "ci_lo", "ci_hi"}
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        for r in rows:
            w.writerow([f"{r[c]:.6f}" if (c in floats and isinstance(r[c], float))
                        else r[c] for c in COLS])

    # How many datasets would G3's rule have found, at each setting?
    print(f"\n{len(rows)} rows -> {out}")
    print("window  floor  datasets where arm 7's deviation is lower with a "
          "95% CI excluding zero (G3 needs 2 of 3; the frozen setting is 30/10)")
    for window in args.windows:
        for floor in args.floors:
            cells = [r for r in rows if r["arm"] == IPW_ARM
                     and r["window"] == window and r["ess_floor"] == floor]
            won = [r["dataset"] for r in cells
                   if r["excludes_zero"] == 1 and r["favors_ipw"] == 1]
            supp = np.mean([r["suppression_rate"] for r in cells])
            print(f"{window:>6}  {floor:>5.0f}  {len(won)} of {len(cells)}"
                  f"  ({', '.join(won) if won else 'none'}); "
                  f"mean suppression {supp:.3f}")

    style()
    fig, axes = plt.subplots(2, len(datasets), figsize=(7.2, 4.0), squeeze=False)
    colours = {10.0: "#1f4e9c", 5.0: "#4c8fd1", 3.0: "#d95f02", 0.0: "#66a61e"}
    for j, dataset in enumerate(datasets):
        ax = axes[0][j]
        for floor in args.floors:
            xs = args.windows
            ys = [next((r["mean_deviation_own"] for r in rows
                        if r["arm"] == IPW_ARM and r["dataset"] == dataset
                        and r["window"] == w and r["ess_floor"] == floor), np.nan)
                  for w in xs]
            ax.plot(xs, ys, marker="o", ms=3, color=colours.get(floor, "#999999"),
                    label=f"ESS floor {floor:.0f}")
        naive = [next((r["mean_deviation_own"] for r in rows
                       if r["arm"] == NAIVE_ARM and r["dataset"] == dataset
                       and r["window"] == w and r["ess_floor"] == args.floors[0]),
                      np.nan) for w in args.windows]
        ax.plot(args.windows, naive, color="#4d4d4d", ls="--", marker="s", ms=3,
                label="naive (arm 6)")
        ax.set_title(dataset, fontsize=SIZES[0])
        if j == 0:
            ax.set_ylabel("mean |reading - pool truth|", fontsize=SIZES[1])
            ax.legend(fontsize=SIZES[2] - 0.5, loc="best")
        ax2 = axes[1][j]
        for floor in args.floors:
            ys = [next((r["suppression_rate"] for r in rows
                        if r["arm"] == IPW_ARM and r["dataset"] == dataset
                        and r["window"] == w and r["ess_floor"] == floor), np.nan)
                  for w in args.windows]
            ax2.plot(args.windows, ys, marker="o", ms=3,
                     color=colours.get(floor, "#999999"))
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xlabel("estimator window (steps)", fontsize=SIZES[1])
        if j == 0:
            ax2.set_ylabel("suppression rate, arm 7", fontsize=SIZES[1])
    fig.suptitle("Post-hoc: the frozen window and floor against the "
                 "alternatives nobody pre-registered",
                 x=0.02, ha="left", fontsize=SIZES[0] + 1)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig_path = Path(args.out_dir) / "sensitivity_window.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"figure -> {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
