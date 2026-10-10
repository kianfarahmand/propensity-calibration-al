"""Validity test 6 of PREREGISTRATION.md Section 14.

    The leave-one-out refit excludes the held-out point from the training set.

The closed form in surrogate.py is exact at fixed hyperparameters, so "exclude"
can be checked literally: compare it against a GP physically refitted on the
other n-1 rows with the same kernel, and separately assert that the held-out
row is absent from that refit's training matrix.

This is the test that protects Section 7.1. If the downdate quietly left the
point in, the windowed residuals would be in-sample residuals, the drift
correction would be a no-op, and arm 7 and arm 12 would differ by nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

from pcal.loop import _refresh_window
from pcal.monitors import IPWMonitor, WindowPoint
from pcal.surrogate import GPSurrogate


@pytest.fixture
def fitted(pool, policies):
    train = np.arange(0, 30)
    return (GPSurrogate(policies, random_state=0)
            .fit(pool.X[train], pool.y[train], indices=train), train)


# ---------------------------------------------------------------- test 6
@pytest.mark.parametrize("i", [0, 1, 7, 17, 29])
def test_closed_form_matches_an_explicit_refit(fitted, i):
    surrogate, _ = fitted
    mu_loo, sd_loo = surrogate.loo_predict()
    mu_ref, sd_ref, _ = surrogate.refit_without(i)
    assert mu_loo[i] == pytest.approx(mu_ref, rel=1e-6, abs=1e-8)
    assert sd_loo[i] == pytest.approx(sd_ref, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("i", [0, 5, 29])
def test_the_held_out_row_is_absent_from_the_refit(fitted, i):
    surrogate, _ = fitted
    _, _, X_kept = surrogate.refit_without(i)
    held = surrogate.X_train[i]
    assert X_kept.shape[0] == surrogate.X_train.shape[0] - 1
    assert not np.any(np.all(np.isclose(X_kept, held), axis=1)), \
        "the held-out point is still in the training matrix"


def test_loo_is_strictly_wider_than_the_in_sample_prediction(fitted):
    """Removing a point cannot make the model more certain about it."""
    surrogate, _ = fitted
    _, sd_loo = surrogate.loo_predict()
    _, sd_in = surrogate.predict(surrogate.X_train, include_noise=True)
    assert np.all(sd_loo >= sd_in - 1e-9)
    assert np.mean(sd_loo) > np.mean(sd_in)


def test_loo_residuals_are_not_the_in_sample_residuals(fitted):
    surrogate, _ = fitted
    mu_loo, _ = surrogate.loo_predict()
    mu_in, _ = surrogate.predict(surrogate.X_train)
    in_sample = np.abs(surrogate.y_train - mu_in)
    loo = np.abs(surrogate.y_train - mu_loo)
    assert np.mean(loo) > np.mean(in_sample)


def test_refresh_window_uses_the_current_surrogate(pool, policies, fitted):
    """The wiring: a windowed point that is in the training set gets refreshed."""
    surrogate, train = fitted
    monitor = IPWMonitor(policies)
    for step, idx in enumerate(train[:5], start=1):
        monitor.push(WindowPoint(step=step, index=int(idx), y=float(pool.y[idx]),
                                 propensity=0.01, pool_size=pool.n,
                                 mu=0.0, sd=1.0, x=pool.X[idx]))
    assert all(not np.isfinite(p.mu_current) for p in monitor.window)
    _refresh_window(monitor, surrogate, train)
    mu_loo, sd_loo = surrogate.loo_predict()
    for row, p in enumerate(monitor.window):
        assert p.mu_current == pytest.approx(mu_loo[row])
        assert p.sd_current == pytest.approx(sd_loo[row])


def test_refresh_is_a_no_op_when_the_ablation_switches_it_off(pool, policies, fitted):
    """Arm 12 is exactly this switch, so it has to actually switch something."""
    import copy

    surrogate, train = fitted
    cfg = copy.deepcopy(policies)
    cfg["estimator"]["loo_refit"] = False
    monitor = IPWMonitor(cfg)
    monitor.push(WindowPoint(step=1, index=int(train[0]), y=float(pool.y[train[0]]),
                             propensity=0.01, pool_size=pool.n, mu=0.0, sd=1.0))
    _refresh_window(monitor, surrogate, train)
    assert not np.isfinite(monitor.window[0].mu_current)


def test_arms_7_and_12_differ_on_the_same_seed(pool, arms):
    """The ablation must change the campaign, or it measures nothing."""
    from pcal.loop import run_campaign

    a7 = run_campaign(pool, arms[7], 0, arms[7]["resolved"], n_steps=14, n_init=10)
    a12 = run_campaign(pool, arms[12], 0, arms[12]["resolved"], n_steps=14, n_init=10)
    assert a7["config_echo"]["loo_refit"] is True
    assert a12["config_echo"]["loo_refit"] is False
    r7 = [s["reading"] for s in a7["trajectory"]]
    r12 = [s["reading"] for s in a12["trajectory"]]
    assert r7 != r12, "the leave-one-out ablation changed no reading at all"
