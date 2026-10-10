"""Calibration monitors: four ways to read coverage from a biased sample.

All four answer the same question - what is C_t(alpha) over the unlabeled pool? -
from the stream of points the campaign chose, and they differ only in what they
do about the fact that the campaign chose them.

    naive   the selected-sample mean. The thing the project argues is wrong.
    ipw     self-normalized inverse propensity, PREREGISTRATION.md Section 7.
    online  the Deshpande and Kuleshov 2021 form: online updating on the
            realized stream, no propensities.
    locbo   the LOCBO 2025 form: localized online conformal, weighting past
            residuals by proximity in feature space, no propensities.

The last two are adaptations, not reimplementations, and are labelled as such
in Section 9 of the pre-registration. They are load-bearing comparators: if
arm 7 does not beat them, the contribution is a measurement note and will be
reported as one.

The monitor is deliberately a pure estimator. The residuals it consumes are
computed by the loop - either the pre-observation values recorded at step s-1,
or, under the primary specification, recomputed under the current surrogate
with the point held out (Section 7.1). That split keeps every monitor testable
without a GP anywhere in sight.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import z_quantile


@dataclass
class WindowPoint:
    """One accepted step, as the estimator sees it."""

    step: int
    index: int
    y: float
    propensity: float
    pool_size: int
    mu: float = np.nan      # under the surrogate at step s-1
    sd: float = np.nan
    mu_current: float = np.nan   # under the current surrogate, point held out
    sd_current: float = np.nan
    x: np.ndarray | None = None  # pool features, for the localized monitor only


class Monitor:
    """Base class. Subclasses implement _estimate()."""

    kind = "base"
    uses_propensities = False

    def __init__(self, cfg: dict):
        self.cfg = cfg
        e = cfg["estimator"]
        self.window_size = int(e["window"])
        self.clip_percentile = float(e["clip_percentile"])
        self.ess_floor = float(e["ess_floor"])
        self.loo_refit = bool(e["loo_refit"])
        self.window: list[WindowPoint] = []
        self.pool_X: np.ndarray | None = None

    def push(self, point: WindowPoint) -> None:
        self.window.append(point)
        if len(self.window) > self.window_size:
            self.window = self.window[-self.window_size:]

    def set_pool(self, X: np.ndarray) -> None:
        """Current unlabeled pool features. Only the localized monitor uses it."""
        self.pool_X = X

    def residuals(self, alpha: float, lam: float) -> np.ndarray:
        """z_s = 1{ |y_s - mu_s| <= q_alpha * lam * sd_s } over the window."""
        q = z_quantile(alpha)
        out = np.empty(len(self.window), dtype=float)
        for i, p in enumerate(self.window):
            mu = p.mu_current if (self.loo_refit and np.isfinite(p.mu_current)) else p.mu
            sd = p.sd_current if (self.loo_refit and np.isfinite(p.sd_current)) else p.sd
            out[i] = float(abs(p.y - mu) <= q * lam * max(sd, 1e-300))
        return out

    def read(self, alpha: float, lam: float = 1.0):
        """Return (estimate, ess, suppressed).

        A suppressed reading is NaN, never a number. The controller holds its
        previous action; the step is recorded; the suppression rate is reported.
        """
        if not self.window:
            return float("nan"), 0.0, True
        value, ess = self._estimate(alpha, lam)
        suppressed = (not np.isfinite(value)) or ess < self.ess_floor
        return (float("nan") if suppressed else float(value)), float(ess), bool(suppressed)

    def _estimate(self, alpha: float, lam: float):
        raise NotImplementedError


class NaiveMonitor(Monitor):
    """Coverage at the selected points, unweighted. The uncorrected reading."""

    kind = "naive"

    def _estimate(self, alpha: float, lam: float):
        z = self.residuals(alpha, lam)
        return float(np.mean(z)), float(z.size)


class IPWMonitor(Monitor):
    """Self-normalized inverse-propensity estimator, PREREGISTRATION.md Section 7.

        w_s     = 1 / ( |U_s| * pi_s(x_s) )
        C_hat   = sum w_s z_s / sum w_s
        ESS     = (sum w)^2 / sum w^2

    Weights are clipped at the 95th percentile of the weights inside the current
    window, always rather than adaptively, before normalization. Hajek rather
    than Horvitz-Thompson because the estimand is a probability the controller
    consumes, so being bounded in [0, 1] is worth the finite-sample bias.
    """

    kind = "ipw"
    uses_propensities = True

    def weights(self) -> np.ndarray:
        w = np.asarray([1.0 / (p.pool_size * max(p.propensity, 1e-300))
                        for p in self.window], dtype=float)
        if self.clip_percentile < 100.0 and w.size:
            cap = float(np.percentile(w, self.clip_percentile))
            w = np.minimum(w, cap)
        return w

    @staticmethod
    def effective_sample_size(w: np.ndarray) -> float:
        total = float(np.sum(w))
        sq = float(np.sum(w ** 2))
        if sq <= 0:
            return 0.0
        return total ** 2 / sq

    def _estimate(self, alpha: float, lam: float):
        z = self.residuals(alpha, lam)
        w = self.weights()
        total = float(np.sum(w))
        if total <= 0:
            return float("nan"), 0.0
        return float(np.dot(w, z) / total), self.effective_sample_size(w)


class OnlineCalibrationMonitor(Monitor):
    """The Deshpande and Kuleshov 2021 form: online updating, no propensities.

    Coverage on the realized query stream is tracked by the online update

        p <- p + eta * (z_s - p)

    applied in window order, which is the stochastic-approximation step those
    methods use to keep the realized stream calibrated. It differs from the
    naive monitor by weighting recent steps more heavily, and from the IPW
    monitor by making no use of how the points were chosen - which is exactly
    the comparison H3 needs. It targets the calibration of the realized stream,
    not of the pool; the pre-registration's claim is that this distinction is
    the whole point, and arm 8 exists to test that claim rather than assume it.
    """

    kind = "online"

    def _estimate(self, alpha: float, lam: float):
        z = self.residuals(alpha, lam)
        eta = float(self.cfg["competitors"]["online_calibration"]["learning_rate"])
        p = float(z[0])
        for value in z[1:]:
            p = p + eta * (value - p)
        ess = (2.0 - eta) / eta if z.size > 1 else 1.0
        return p, min(ess, float(z.size))


class LocalizedConformalMonitor(Monitor):
    """The LOCBO 2025 form: localized online conformal, no propensities.

    Each pool point is given a kernel-weighted average of the windowed coverage
    indicators, weighting past points by proximity in the PCA feature space,
    and the pool estimate is the mean over pool points. The bandwidth is the
    median pairwise distance inside the window (the median heuristic), so no
    extra constant is introduced.

    This is the strongest no-propensity competitor available: it tries to reach
    the pool marginal by extrapolating locally rather than by reweighting.
    """

    kind = "locbo"

    def _estimate(self, alpha: float, lam: float):
        z = self.residuals(alpha, lam)
        if self.pool_X is None or self.pool_X.shape[0] == 0:
            return float(np.mean(z)), float(z.size)
        Xw = self._window_features()
        if Xw is None:
            return float(np.mean(z)), float(z.size)
        d_ww = np.sqrt(np.maximum(_sqdist(Xw, Xw), 0.0))
        iu = np.triu_indices(d_ww.shape[0], k=1)
        h = float(np.median(d_ww[iu])) if iu[0].size else 1.0
        h = max(h * float(self.cfg["competitors"]["localized_conformal"]["bandwidth_scale"]),
                1e-6)
        d_pw = np.maximum(_sqdist(self.pool_X, Xw), 0.0)
        K = np.exp(-0.5 * d_pw / (h ** 2))
        row = K.sum(axis=1)
        good = row > 0
        if not good.any():
            return float(np.mean(z)), float(z.size)
        local = (K[good] @ z) / row[good]
        return float(np.mean(local)), float(z.size)

    def _window_features(self):
        rows = [p.x for p in self.window if getattr(p, "x", None) is not None]
        if len(rows) != len(self.window) or not rows:
            return None
        return np.asarray(rows, dtype=float)


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return (np.sum(A ** 2, axis=1)[:, None] + np.sum(B ** 2, axis=1)[None, :]
            - 2.0 * A @ B.T)


_REGISTRY = {
    "naive": NaiveMonitor,
    "ipw": IPWMonitor,
    "online": OnlineCalibrationMonitor,
    "locbo": LocalizedConformalMonitor,
}


def make_monitor(kind: str, cfg: dict) -> Monitor | None:
    """None means the arm runs without a monitor, which is a real arm (5, 2-4, 10)."""
    if kind in (None, "none"):
        return None
    if kind not in _REGISTRY:
        raise ValueError(f"unknown monitor {kind!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[kind](cfg)
