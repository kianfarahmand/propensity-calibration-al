"""Controllers: what the campaign does with a coverage reading.

PREREGISTRATION.md Section 8.

Primary form, recalibration. Fit a scalar variance inflation lambda_t > 0 such
that the corrected coverage estimate matches nominal, by bisection on log lambda
over [-2, 2], and apply lambda_t * sigma_t inside the acquisition function at
step t+1. lambda_t is exponentially smoothed with factor 0.3 to prevent
oscillation; the smoothing is applied to lambda itself, which is the literal
reading of the frozen text.

Secondary form, policy switch. Switch between EI and max-variance when the
debiased reading leaves alpha +/- 0.05 for three consecutive non-suppressed
steps. Reported separately; it does not enter the primary gate.

Suppressed readings are not zeros and not stale numbers: the controller holds
its previous action and the step is recorded as suppressed.
"""
from __future__ import annotations

import numpy as np


class RecalibrationController:
    """Variance inflation by bisection on log lambda."""

    kind = "recalibration"

    def __init__(self, cfg: dict):
        c = cfg["controller"]
        self.cfg = cfg
        self.log_lo, self.log_hi = (float(v) for v in c["log_lambda_bounds"])
        self.iterations = int(c["bisection_iterations"])
        self.smoothing = float(c["smoothing"])
        self.lam = 1.0
        self.history: list[float] = []

    def solve(self, monitor, alpha: float) -> float:
        """Smallest lambda in the bracket whose corrected estimate reaches alpha.

        The estimate is non-decreasing in lambda - inflating sigma can only
        admit more points into the interval - so bisection is well posed. If the
        bracket cannot reach nominal from either side, lambda saturates at that
        end rather than extrapolating.
        """
        def value_at(log_lam: float) -> float:
            value, _ = monitor._estimate(alpha, float(np.exp(log_lam)))
            return float(value)

        lo, hi = self.log_lo, self.log_hi
        v_lo, v_hi = value_at(lo), value_at(hi)
        if not np.isfinite(v_lo) or not np.isfinite(v_hi):
            return self.lam
        if v_lo >= alpha:
            return float(np.exp(lo))
        if v_hi <= alpha:
            return float(np.exp(hi))
        for _ in range(self.iterations):
            mid = 0.5 * (lo + hi)
            if value_at(mid) < alpha:
                lo = mid
            else:
                hi = mid
        return float(np.exp(0.5 * (lo + hi)))

    def update(self, monitor, alpha: float, suppressed: bool) -> float:
        """Return the lambda to use at the next step."""
        if suppressed or monitor is None:
            self.history.append(self.lam)
            return self.lam
        target = self.solve(monitor, alpha)
        self.lam = (1.0 - self.smoothing) * self.lam + self.smoothing * target
        self.history.append(self.lam)
        return self.lam


class PolicySwitchController:
    """Secondary form: switch acquisition when the reading leaves the band."""

    kind = "policy_switch"

    def __init__(self, cfg: dict):
        ps = cfg["controller"]["policy_switch"]
        self.band = float(ps["band"])
        self.required = int(ps["consecutive"])
        self.policies = list(ps["policies"])
        self.active = 0
        self.outside = 0
        self.switches = 0

    @property
    def policy(self) -> str:
        return self.policies[self.active]

    def update(self, reading: float, alpha: float, suppressed: bool) -> str:
        if suppressed or not np.isfinite(reading):
            return self.policy          # a held action, not a reset
        if abs(reading - alpha) > self.band:
            self.outside += 1
        else:
            self.outside = 0
        if self.outside >= self.required:
            self.active = (self.active + 1) % len(self.policies)
            self.outside = 0
            self.switches += 1
        return self.policy


def make_controller(kind: str, cfg: dict):
    if kind in (None, "none"):
        return None
    if kind == "recalibration":
        return RecalibrationController(cfg)
    if kind == "policy_switch":
        return PolicySwitchController(cfg)
    if kind == "global_scaling":
        from .baselines import GlobalScalingController
        return GlobalScalingController(cfg)
    raise ValueError(f"unknown controller {kind!r}")
