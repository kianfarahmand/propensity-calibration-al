"""Shared fixtures.

The fast test job must not touch the network, so every test here runs on a
synthetic pool built to the same shape as a real one. The real-data checks that
genuinely need matminer are marked slow and excluded from the default job.
"""
from __future__ import annotations

import numpy as np
import pytest

from pcal.config import apply_overrides, arm_config, load_policies
from pcal.data import Pool


def make_pool(n: int = 120, d: int = 4, seed: int = 0, name: str = "synthetic") -> Pool:
    """A pool with the same contract as a real one: standardized X, standardized y."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    signal = np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 1] ** 2 - 0.3 * X[:, 2]
    y_raw = signal + rng.normal(scale=0.15, size=n)
    y_mean, y_std = float(np.mean(y_raw)), float(np.std(y_raw))
    return Pool(
        name=name, X=np.ascontiguousarray(X), y=(y_raw - y_mean) / y_std, y_raw=y_raw,
        y_mean=y_mean, y_std=y_std, formulas=[f"X{i}" for i in range(n)],
        direction="max", units="arb", kind="synthetic",
        n_raw_rows=n, n_after_dedup=n, n_components=d, explained_variance=1.0,
    )


@pytest.fixture
def pool() -> Pool:
    return make_pool()


@pytest.fixture
def policies() -> dict:
    """The real configs, with the tuning constant filled in.

    acquisition.c is null until the S05 sweep fixes it, and the sampler refuses
    to run without it on purpose. Tests supply a value so the machinery is
    exercised; no test result depends on which value.
    """
    return apply_overrides(load_policies(), {"acquisition.c": 1.0,
                                             "estimator.ess_floor": 3})


@pytest.fixture
def arms(policies):
    return {i: arm_config(policies, i) for i in (2, 5, 6, 7, 12, 13)}
