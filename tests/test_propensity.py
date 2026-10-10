"""Validity test 5 of PREREGISTRATION.md Section 14, and the estimator mechanics.

Test 5: logged propensities form a valid distribution over U_t at every step
(non-negative, summing to 1 within 1e-9) and satisfy pi_t(x) >= eps / |U_t|.

The exploration floor is not decoration. It is the only thing bounding the
importance weights in monitors.py, so a campaign that violated it would produce
an estimator with unbounded variance and no warning.
"""
from __future__ import annotations

import numpy as np
import pytest

from pcal.monitors import IPWMonitor, NaiveMonitor, WindowPoint
from pcal.policies import (
    acquisition_scores,
    boltzmann_propensities,
    greedy_propensities,
    validate_propensities,
)

N_STEPS = 12
N_INIT = 10


# ---------------------------------------------------------------- test 5
@pytest.mark.parametrize("c", [0.25, 0.5, 1.0, 2.0, 4.0])
@pytest.mark.parametrize("n", [5, 200])
def test_boltzmann_is_a_distribution_with_the_floor(c, n):
    rng = np.random.default_rng(int(c * 100) + n)
    scores = rng.normal(scale=10.0, size=n)
    eps = 0.05
    pi = boltzmann_propensities(scores, c=c, eps=eps)
    assert np.all(pi >= 0)
    assert abs(pi.sum() - 1.0) <= 1e-9
    assert np.all(pi >= eps / n - 1e-12)
    validate_propensities(pi, eps, randomized=True)


def test_constant_scores_give_the_uniform_distribution():
    pi = boltzmann_propensities(np.zeros(50), c=1.0, eps=0.05)
    assert np.allclose(pi, 1.0 / 50)


def test_extreme_scores_do_not_overflow():
    scores = np.array([0.0, 1e6, -1e6, 5e5])
    pi = boltzmann_propensities(scores, c=1e-6, eps=0.05)
    assert np.isfinite(pi).all()
    assert abs(pi.sum() - 1.0) <= 1e-9
    assert np.all(pi >= 0.05 / 4 - 1e-12)


def test_greedy_is_a_point_mass_and_is_not_floored():
    scores = np.array([0.1, 0.9, 0.3])
    pi = greedy_propensities(scores)
    assert pi.tolist() == [0.0, 1.0, 0.0]
    validate_propensities(pi, 0.05, randomized=False)
    with pytest.raises(ValueError):
        validate_propensities(pi, 0.05, randomized=True)


def test_logged_propensities_respect_the_floor_at_every_step(pool, arms):
    from pcal.loop import run_campaign

    eps = float(arms[7]["resolved"]["acquisition"]["epsilon"])
    for arm_id in (5, 7):
        arm = arms[arm_id]
        record = run_campaign(pool, arm, 0, arm["resolved"],
                              n_steps=N_STEPS, n_init=N_INIT)
        for step in record["trajectory"]:
            assert step["propensity"] >= eps / step["pool_size"] - 1e-12
            assert step["propensity"] <= 1.0
            assert step["pool_size"] == pool.n - N_INIT - (step["step"] - 1)


def test_scores_are_finite_for_every_acquisition(pool, policies):
    rng = np.random.default_rng(0)
    mu = rng.normal(size=40)
    sd = np.abs(rng.normal(size=40)) + 1e-3
    for name in ("random", "ei", "ucb", "max_variance"):
        a = acquisition_scores(name, mu, sd, best_y=float(mu.max()), cfg=policies)
        assert np.isfinite(a).all()
        assert a.shape == mu.shape


# ------------------------------------------------- estimator mechanics, Section 7
def window(n=20, seed=0, pool_size=500, biased=True):
    """A window of accepted steps with known propensities and known coverage."""
    rng = np.random.default_rng(seed)
    points = []
    for s in range(n):
        # a deliberately uneven sampling distribution over two strata
        covered = s % 4 != 0                       # 75% covered in the selected sample
        pi = (0.5 if covered else 0.05) / pool_size * pool_size
        pi = (0.5 if covered else 0.05)
        points.append(WindowPoint(step=s + 1, index=s, y=0.0, propensity=pi,
                                  pool_size=pool_size,
                                  mu=0.0 if covered else 10.0, sd=1.0,
                                  x=rng.normal(size=3)))
    return points


def fill(monitor, points):
    for p in points:
        monitor.push(p)
    return monitor


def test_self_normalized_estimate_stays_in_the_unit_interval(policies):
    m = fill(IPWMonitor(policies), window())
    value, ess, suppressed = m.read(0.90)
    assert 0.0 <= value <= 1.0
    assert ess > 0
    assert not suppressed


def test_clipping_caps_the_weights_and_raises_ess(policies):
    points = window()
    points[0].propensity = 1e-6          # one enormous weight
    clipped = fill(IPWMonitor(policies), points)
    import copy

    no_clip_cfg = copy.deepcopy(policies)
    no_clip_cfg["estimator"]["clip_percentile"] = 100
    unclipped = fill(IPWMonitor(no_clip_cfg), copy.deepcopy(points))

    w_clipped = clipped.weights()
    w_raw = unclipped.weights()
    assert w_clipped.max() < w_raw.max()
    assert IPWMonitor.effective_sample_size(w_clipped) > \
        IPWMonitor.effective_sample_size(w_raw)


def test_ess_is_bounded_by_the_window_length(policies):
    m = fill(IPWMonitor(policies), window())
    assert 0 < m.read(0.90)[1] <= len(m.window)


def test_readings_below_the_ess_floor_are_suppressed_not_guessed(policies):
    import copy

    cfg = copy.deepcopy(policies)
    cfg["estimator"]["ess_floor"] = 1e9
    m = fill(IPWMonitor(cfg), window())
    value, ess, suppressed = m.read(0.90)
    assert suppressed
    assert np.isnan(value), "a suppressed reading must be NaN, never a number"


def test_ipw_corrects_a_bias_the_naive_reading_reports(policies):
    """The premise of the whole project, on a case where the answer is known.

    Points are drawn from a pool where coverage is 0.5, by a sampler that
    prefers covered points two to one. The naive mean of the selected sample
    reports the sampler's preference; the weighted estimate reports the pool.
    """
    rng = np.random.default_rng(0)
    pool_size, n = 1000, 400
    covered = rng.random(n) < 2.0 / 3.0          # selected sample is 2/3 covered
    points = []
    for s, is_cov in enumerate(covered):
        # propensity is twice as large for covered points, so the weight halves
        pi = (2.0 if is_cov else 1.0) / (1.5 * pool_size)
        points.append(WindowPoint(step=s + 1, index=s, y=0.0, propensity=pi,
                                  pool_size=pool_size,
                                  mu=0.0 if is_cov else 10.0, sd=1.0))
    cfg = {**policies}
    cfg["estimator"] = {**policies["estimator"], "window": n, "clip_percentile": 100}
    naive = fill(NaiveMonitor(cfg), points).read(0.90)[0]
    ipw = fill(IPWMonitor(cfg), points).read(0.90)[0]
    assert naive == pytest.approx(2.0 / 3.0, abs=0.06)
    assert ipw == pytest.approx(0.5, abs=0.06)
    assert abs(ipw - 0.5) < abs(naive - 0.5)
