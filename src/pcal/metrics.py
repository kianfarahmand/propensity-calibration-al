"""Outcome measures and the evaluator's ground truth.

PREREGISTRATION.md Section 11 names three co-primary optimization outcomes and
one primary measurement outcome. Everything here that uses unrevealed labels is
evaluator-only and is written to the evaluation half of the run record.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def z_quantile(alpha: float) -> float:
    """The two-sided normal quantile q_alpha of PREREGISTRATION.md Section 1."""
    return float(norm.ppf(0.5 * (1.0 + float(alpha))))


def coverage_indicator(y: np.ndarray, mu: np.ndarray, sd: np.ndarray,
                       alpha: float) -> np.ndarray:
    """1{ |y - mu| <= q_alpha * sd }, elementwise."""
    q = z_quantile(alpha)
    sd = np.maximum(np.asarray(sd, dtype=float), 1e-300)
    return (np.abs(np.asarray(y, dtype=float) - np.asarray(mu, dtype=float))
            <= q * sd).astype(float)


def pool_coverage(y_pool: np.ndarray, mu: np.ndarray, sd: np.ndarray,
                  alpha: float) -> float:
    """C_t(alpha): the estimand, over the unlabeled pool. Evaluator only."""
    return float(np.mean(coverage_indicator(y_pool, mu, sd, alpha)))


def simple_regret(pool, best_revealed_standardized: float) -> float:
    """Best achievable minus best found, in the original units of the target."""
    best_possible = float(np.max(pool.y_raw))
    found = float(pool.to_original_units(best_revealed_standardized))
    return best_possible - found


def topk_enrichment(revealed_indices, y_raw: np.ndarray, k: int) -> float:
    """Fraction of the pool's true top-k that the campaign has in hand."""
    y_raw = np.asarray(y_raw, dtype=float)
    k = min(int(k), y_raw.shape[0])
    top = set(np.argsort(-y_raw, kind="stable")[:k].tolist())
    hits = len(top.intersection(int(i) for i in revealed_indices))
    return hits / float(k)


def discovery_auc(best_curve_standardized, pool) -> float:
    """Normalized area under the best-so-far curve over the campaign.

    0 means the campaign never improved on its seed points; 1 means it held the
    pool optimum from the first step. The normalization is per run, against that
    run's own starting point, because initializations are matched across arms
    but differ across seeds.
    """
    curve = np.asarray(best_curve_standardized, dtype=float)
    if curve.size == 0:
        return float("nan")
    raw = pool.to_original_units(curve)
    start = float(raw[0])
    best_possible = float(np.max(pool.y_raw))
    span = best_possible - start
    if span <= 0:
        return 1.0
    return float(np.mean(np.clip((raw - start) / span, 0.0, 1.0)))


def absolute_deviation(estimates, truth) -> float:
    """Primary measurement outcome: mean |C_hat_t - C_t| over usable steps.

    NaN entries are suppressed readings and are excluded, not treated as zero;
    the suppression rate is reported separately so a method cannot buy accuracy
    by refusing to answer.
    """
    e = np.asarray(estimates, dtype=float)
    t = np.asarray(truth, dtype=float)
    usable = np.isfinite(e) & np.isfinite(t)
    if not usable.any():
        return float("nan")
    return float(np.mean(np.abs(e[usable] - t[usable])))


def summarize_run(record: dict, pool) -> dict:
    """Collapse one run record into the pre-registered outcome set."""
    traj = record["trajectory"]
    ev = record["evaluation"]
    revealed = list(record["initial_indices"]) + [s["index"] for s in traj]
    best_std = traj[-1]["best_y"] if traj else record["initial_best_y"]
    curve = [record["initial_best_y"]] + [s["best_y"] for s in traj]
    steps = np.asarray([s["step"] for s in traj], dtype=int)
    window = (steps >= 50) & (steps <= 150)
    estimates = np.asarray([s["reading"] for s in traj], dtype=float)
    truth = np.asarray([e["pool_coverage"] for e in ev], dtype=float)
    suppressed = np.asarray([bool(s["suppressed"]) for s in traj])
    sel = window if window.any() else np.ones_like(window, dtype=bool)
    return {
        "simple_regret": simple_regret(pool, best_std),
        "topk_enrichment_10": topk_enrichment(revealed, pool.y_raw, 10),
        "topk_enrichment_50": topk_enrichment(revealed, pool.y_raw, 50),
        "discovery_auc": discovery_auc(curve, pool),
        "mean_absolute_deviation": absolute_deviation(estimates[sel], truth[sel]),
        "suppression_rate": float(np.mean(suppressed[sel])) if sel.any() else float("nan"),
        "mean_ess": float(np.nanmean([s["ess"] for s in traj])) if traj else float("nan"),
        "final_lambda": traj[-1]["lambda"] if traj else 1.0,
        "n_steps": len(traj),
    }
