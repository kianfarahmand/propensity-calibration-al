"""The campaign loop and the run record.

Order within a step matters and is not arbitrary:

    1. refit the surrogate on everything revealed so far        -> surrogate at t
    2. evaluator records C_t(alpha) over the unlabeled pool     -> ground truth at t
    3. recompute the windowed residuals under that surrogate
       with each point held out (PREREGISTRATION.md 7.1)
    4. read the monitor, update the controller                  -> lambda_t
    5. score the pool with lambda_t * sigma, sample, log the
       realized propensity                                      -> before any label
    6. reveal the label

Steps 2 and 3 share the same surrogate, which is the whole point of the
leave-one-out refit: the estimate and the thing it estimates have to refer to
the same model, or the comparison measures drift instead of selection bias.

The record is split in two. `trajectory` is everything the campaign could
legitimately see. `evaluation` is the evaluator's ground truth, which uses
unrevealed labels and must never flow back. Validity test 2 permutes the labels
of never-selected pool members and asserts the trajectory is bit-identical,
which is what makes that separation checkable rather than merely intended.
"""
from __future__ import annotations

import time

import numpy as np

from .baselines import GPHedge
from .controller import make_controller
from .metrics import pool_coverage
from .monitors import WindowPoint, make_monitor
from .oracle import Oracle
from .policies import acquisition_scores, select, validate_propensities
from .surrogate import GPSurrogate


def initial_indices(n: int, n_init: int, seed: int) -> np.ndarray:
    """The matched initialization: a function of (dataset, seed) and nothing else.

    Every arm at a given (dataset, seed) starts from exactly these points
    (PREREGISTRATION.md Section 9). Validity test 4 checks it.
    """
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(int(n), size=int(n_init), replace=False))


def run_campaign(pool, arm: dict, seed: int, cfg: dict, n_steps: int,
                 n_init: int, alpha: float | None = None,
                 y_override: np.ndarray | None = None,
                 record_evaluation: bool = True,
                 policy_switch: bool = False) -> dict:
    """Run one (arm, dataset, seed) campaign and return its record."""
    resolved = arm.get("resolved", cfg)
    alpha = float(alpha if alpha is not None else resolved["estimand"]["alpha_primary"])
    secondary = [float(a) for a in resolved["estimand"]["alpha_secondary"]]
    randomized = bool(arm["randomized"])
    arm_id = int(arm["id"])

    y_all = pool.y if y_override is None else np.asarray(y_override, dtype=float)
    oracle = Oracle(y_all, name=pool.name)
    init = initial_indices(pool.n, n_init, seed)
    oracle.reveal_many(init)
    initial_best = float(np.max(oracle.revealed_values))

    rng = np.random.default_rng([int(seed), arm_id])
    monitor = make_monitor(arm["monitor"], resolved)
    controller = make_controller(arm["controller"], resolved)
    switcher = make_controller("policy_switch", resolved) if policy_switch else None
    hedge = GPHedge(resolved, n_steps) if arm["policy"] == "hedge" else None

    surrogate = GPSurrogate(resolved, random_state=int(seed))
    lam = 1.0
    best_y = initial_best
    trajectory: list[dict] = []
    evaluation: list[dict] = []

    for step in range(1, int(n_steps) + 1):
        t0 = time.perf_counter()

        # 1. surrogate at t
        revealed = np.asarray(oracle.revealed_indices, dtype=int)
        surrogate.fit(pool.X[revealed], oracle.revealed_values, indices=revealed)
        unlabeled = oracle.unlabeled_indices()
        if unlabeled.size == 0:
            break
        mu_obs_pool, sd_obs_pool = surrogate.predict(pool.X[unlabeled], include_noise=True)

        # 2. evaluator ground truth, under the same surrogate, before anything acts
        if record_evaluation:
            truth = {"step": step, "n_unlabeled": int(unlabeled.size),
                     "pool_coverage": pool_coverage(pool.y[unlabeled], mu_obs_pool,
                                                    sd_obs_pool, alpha)}
            for a in secondary:
                truth[f"pool_coverage_{a:g}"] = pool_coverage(
                    pool.y[unlabeled], mu_obs_pool, sd_obs_pool, a)
            evaluation.append(truth)

        # 3-4. the reading, and what the controller does with it
        reading, ess, suppressed = float("nan"), float("nan"), True
        if monitor is not None and monitor.window:
            _refresh_window(monitor, surrogate, revealed)
            monitor.set_pool(pool.X[unlabeled])
            reading, ess, suppressed = monitor.read(alpha, lam=1.0)
            if controller is not None:
                lam = (controller.update(monitor, alpha, suppressed, step=step)
                       if controller.kind == "global_scaling"
                       else controller.update(monitor, alpha, suppressed))
            if switcher is not None:
                switcher.update(reading, alpha, suppressed)

        # 5. acquisition under lambda_t * sigma, then sampling with known propensity
        mu_lat, sd_lat = surrogate.predict(pool.X[unlabeled], include_noise=False)
        sd_eff = lam * sd_lat
        if hedge is not None:
            position, pi, policy_name = hedge.select(rng, mu_lat, sd_eff, best_y)
        else:
            policy_name = switcher.policy if switcher is not None else arm["policy"]
            scores = acquisition_scores(policy_name, mu_lat, sd_eff, best_y, resolved)
            position, pi = select(rng, scores, randomized, resolved)
        validate_propensities(pi, float(resolved["acquisition"]["epsilon"]), randomized)
        chosen = int(unlabeled[position])
        propensity = float(pi[position])

        # pre-observation prediction at the chosen point, logged before the label
        mu_sel = float(mu_obs_pool[position])
        sd_sel = float(sd_obs_pool[position])

        # 6. pay for the label
        y_obs = float(oracle.reveal(chosen))
        if hedge is not None:
            hedge.update()
        best_y = max(best_y, y_obs)

        if monitor is not None:
            monitor.push(WindowPoint(step=step, index=chosen, y=y_obs,
                                     propensity=propensity, pool_size=int(unlabeled.size),
                                     mu=mu_sel, sd=sd_sel, x=pool.X[chosen]))

        trajectory.append({
            "step": step, "index": chosen, "propensity": propensity,
            "pool_size": int(unlabeled.size), "mu": mu_sel, "sd": sd_sel,
            "y": y_obs, "best_y": float(best_y), "policy": policy_name,
            "reading": reading, "ess": ess, "suppressed": bool(suppressed),
            "lambda": float(lam), "seconds": time.perf_counter() - t0,
        })

    return {
        "dataset": pool.name, "arm": arm_id, "arm_name": arm["name"], "seed": int(seed),
        "alpha": alpha, "n_init": int(n_init), "n_steps": int(n_steps),
        "initial_indices": [int(i) for i in init], "initial_best_y": initial_best,
        "trajectory": trajectory, "evaluation": evaluation,
        "monitor": arm["monitor"], "controller": arm["controller"],
        "randomized": randomized,
        "policy_switches": (switcher.switches if switcher is not None else 0),
        "config_echo": {
            "c": resolved["acquisition"].get("c"),
            "epsilon": resolved["acquisition"]["epsilon"],
            "window": resolved["estimator"]["window"],
            "clip_percentile": resolved["estimator"]["clip_percentile"],
            "ess_floor": resolved["estimator"]["ess_floor"],
            "loo_refit": resolved["estimator"]["loo_refit"],
            "overrides": arm.get("overrides") or {},
        },
        "pool_summary": pool.summary(),
    }


def _refresh_window(monitor, surrogate, revealed: np.ndarray) -> None:
    """Recompute each windowed residual under the current surrogate, point held out.

    Without this, z_s is computed under the surrogate as it stood at step s-1
    while C_t refers to the surrogate at step t, and the windowed estimate
    carries a model-drift bias on top of the selection bias it is meant to
    measure. Arm 12 is this function switched off, so the correction's
    contribution is measured rather than asserted.
    """
    if not monitor.loo_refit:
        return
    row_of = {int(idx): row for row, idx in enumerate(revealed)}
    mu_loo, sd_loo = surrogate.loo_predict()
    for point in monitor.window:
        row = row_of.get(int(point.index))
        if row is None:
            continue
        point.mu_current = float(mu_loo[row])
        point.sd_current = float(sd_loo[row])
