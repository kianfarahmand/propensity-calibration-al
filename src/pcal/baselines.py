"""Baselines that are neither a monitor nor the proposed controller.

Arm 10, GP-Hedge (Hoffman, Brochu and de Freitas 2011): a portfolio of
acquisition functions, each nominating a candidate, with one nominee chosen by
Hedge weights over cumulative gains. It switches behaviour without any
reliability signal at all, which is the comparison it is here to provide.

Arm 11, one-off global uncertainty scaling: fit a single variance inflation at
step 50 from the selected sample and never touch it again. It is the static
counterpart of the controller - recalibration without the loop.
"""
from __future__ import annotations

import numpy as np

from .policies import acquisition_scores


class GPHedge:
    """Hedge over a fixed portfolio of acquisition functions."""

    kind = "gp_hedge"

    def __init__(self, cfg: dict, n_steps: int):
        h = cfg["competitors"]["gp_hedge"]
        self.cfg = cfg
        self.portfolio = list(h["portfolio"])
        k = len(self.portfolio)
        eta = h.get("eta")
        self.eta = float(eta) if eta else float(np.sqrt(8.0 * np.log(k) / max(n_steps, 1)))
        self.gains = np.zeros(k, dtype=float)
        self.nominations: list[int] = []

    def probabilities(self) -> np.ndarray:
        shifted = self.eta * (self.gains - np.max(self.gains))
        w = np.exp(shifted)
        return w / w.sum()

    def select(self, rng, mu, sd, best_y):
        """Return (position, propensity over the pool, chosen acquisition name).

        The propensity vector is the mixture over nominees. It is a valid
        distribution over the pool, but it has no exploration floor, so it is
        never inverted by a monitor - arm 10 runs without one.
        """
        nominees = []
        for name in self.portfolio:
            scores = acquisition_scores(name, mu, sd, best_y, self.cfg)
            nominees.append(int(np.argmax(scores)))
        p = self.probabilities()
        choice = int(rng.choice(len(self.portfolio), p=p))
        pi = np.zeros(np.asarray(mu).shape[0], dtype=float)
        for j, pos in enumerate(nominees):
            pi[pos] += p[j]
        pi = pi / pi.sum()
        self.nominations = nominees
        self._last_mu = np.asarray(mu, dtype=float)
        return nominees[choice], pi, self.portfolio[choice]

    def update(self) -> None:
        """Reward every acquisition by the posterior mean at its own nominee."""
        for j, pos in enumerate(self.nominations):
            self.gains[j] += float(self._last_mu[pos])


class GlobalScalingController:
    """Arm 11: one variance inflation, fitted once at a fixed step."""

    kind = "global_scaling"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.fit_step = int(cfg["competitors"]["global_scaling"]["fit_step"])
        c = cfg["controller"]
        self.log_lo, self.log_hi = (float(v) for v in c["log_lambda_bounds"])
        self.iterations = int(c["bisection_iterations"])
        self.lam = 1.0
        self.fitted = False
        self.history: list[float] = []

    def update(self, monitor, alpha: float, suppressed: bool, step: int = 0) -> float:
        if self.fitted or suppressed or monitor is None or step < self.fit_step:
            self.history.append(self.lam)
            return self.lam
        from .controller import RecalibrationController

        solver = RecalibrationController(self.cfg)
        self.lam = solver.solve(monitor, alpha)
        self.fitted = True
        self.history.append(self.lam)
        return self.lam
