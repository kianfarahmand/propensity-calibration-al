"""Acquisition scores and the randomized sampler that makes propensities known.

PREREGISTRATION.md Section 6:

    pi_t(x) = (1 - eps) * softmax( a_t(x) / tau_t ) + eps / |U_t|
    tau_t   = c * MAD_t

The exploration floor eps is what makes the whole method possible: it bounds
every propensity away from zero, so the importance weights in monitors.py are
bounded. tau is set from the median absolute deviation of the score over the
current pool, which makes it scale-free, which is why one c transfers across
datasets with completely different target units.

The realized propensity of the selected point is logged at selection time,
before the label is revealed. Nothing here ever touches an unrevealed label.

Acquisition uses the latent sd, not the observation sd (see surrogate.py).
EI uses xi = 0 and UCB uses kappa = 2.0; neither was pre-registered and both
are recorded in preregistration/deviations.md.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

GREEDY = ("ei", "ucb", "max_variance", "random")


def acquisition_scores(name: str, mu: np.ndarray, sd: np.ndarray,
                       best_y: float, cfg: dict) -> np.ndarray:
    """Score every pool candidate. Larger is always more desirable."""
    a = cfg["acquisition"]
    mu = np.asarray(mu, dtype=float)
    sd = np.maximum(np.asarray(sd, dtype=float), 1e-12)
    if name == "random":
        return np.zeros_like(mu)
    if name == "max_variance":
        return sd
    if name == "ucb":
        return mu + float(a["ucb_kappa"]) * sd
    if name == "ei":
        improvement = mu - float(best_y) - float(a["ei_xi"])
        z = improvement / sd
        return improvement * norm.cdf(z) + sd * norm.pdf(z)
    raise ValueError(f"unknown acquisition {name!r}")


def mad(values: np.ndarray) -> float:
    """Median absolute deviation, the scale that makes c dataset-independent."""
    v = np.asarray(values, dtype=float)
    return float(np.median(np.abs(v - np.median(v))))


def boltzmann_propensities(scores: np.ndarray, c: float, eps: float,
                           mad_floor: float = 1e-12) -> np.ndarray:
    """The pre-registered sampling distribution over the current pool.

    A degenerate score vector (zero MAD, e.g. the random arm) falls out of this
    as the uniform distribution without a special case, because the softmax of a
    constant vector is uniform.
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.shape[0]
    if n == 0:
        raise ValueError("empty pool")
    tau = max(float(c) * mad(scores), float(mad_floor))
    shifted = (scores - np.max(scores)) / tau
    weights = np.exp(shifted)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        soft = np.full(n, 1.0 / n)
    else:
        soft = weights / total
    pi = (1.0 - float(eps)) * soft + float(eps) / n
    pi = pi / pi.sum()
    return pi


def greedy_propensities(scores: np.ndarray) -> np.ndarray:
    """A point mass on the argmax: a valid distribution, with no usable weights.

    Greedy arms are included so H1 can be tested across policies with different
    bias signs. They do not feed a propensity-weighted monitor, and nothing in
    the codebase may invert one of these.
    """
    pi = np.zeros(np.asarray(scores).shape[0], dtype=float)
    pi[int(np.argmax(scores))] = 1.0
    return pi


def select(rng, scores: np.ndarray, randomized: bool, cfg: dict,
           c: float | None = None):
    """Choose one pool position. Returns (position, propensity vector).

    `position` indexes the *current pool array*, not the original dataset row;
    the caller maps it back. The full propensity vector is returned so validity
    test 5 can check it is a distribution over U_t at every step.
    """
    scores = np.asarray(scores, dtype=float)
    if randomized:
        a = cfg["acquisition"]
        if c is None:
            c = a.get("c")
        if c is None:
            raise ValueError(
                "acquisition.c is null in configs/policies.yaml. It is fixed by "
                "the S05 tuning sweep on the tuning pair; no evaluation run may "
                "start before then."
            )
        pi = boltzmann_propensities(scores, c=float(c), eps=float(a["epsilon"]),
                                    mad_floor=float(a["mad_floor"]))
        pos = int(rng.choice(scores.shape[0], p=pi))
    else:
        pi = greedy_propensities(scores)
        pos = int(np.argmax(scores))
    return pos, pi


def validate_propensities(pi: np.ndarray, eps: float, randomized: bool,
                          tolerance: float = 1e-9) -> None:
    """Validity test 5, as an assertion the harness can run at every step."""
    pi = np.asarray(pi, dtype=float)
    if np.any(pi < 0):
        raise ValueError("negative propensity")
    total = float(pi.sum())
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"propensities sum to {total!r}, not 1")
    if randomized:
        floor = float(eps) / pi.shape[0]
        if np.any(pi < floor - tolerance):
            raise ValueError("a propensity fell below the eps/|U_t| exploration floor")
