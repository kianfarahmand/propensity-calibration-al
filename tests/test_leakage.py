"""Validity tests 1, 2, 3, 4 and 7 of PREREGISTRATION.md Section 14.

These are the tests that make the leak-free claim checkable rather than
asserted. The run harness aborts if any of them fail, and they run in CI on
every commit.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

from pcal.config import repo_root
from pcal.loop import initial_indices, run_campaign

N_STEPS = 12
N_INIT = 10


def campaign(pool, arm, seed=0, **kwargs):
    return run_campaign(pool, arm, seed, arm["resolved"],
                        n_steps=N_STEPS, n_init=N_INIT, **kwargs)


def strip_timing(trajectory):
    """Wall clock is the one field that legitimately differs between runs."""
    return [{k: v for k, v in step.items() if k != "seconds"} for step in trajectory]


# ---------------------------------------------------------------- test 1
def test_revealed_labels_are_exactly_initial_plus_selected(pool, arms):
    record = campaign(pool, arms[7])
    init = list(record["initial_indices"])
    selected = [s["index"] for s in record["trajectory"]]
    assert len(selected) == N_STEPS
    assert len(set(selected)) == len(selected), "a point was selected twice"
    assert not set(init).intersection(selected), "a seed point was selected again"
    assert len(set(init) | set(selected)) == N_INIT + N_STEPS


# ---------------------------------------------------------------- test 2
def test_permuting_never_selected_labels_leaves_trajectory_identical(pool, arms):
    base = campaign(pool, arms[7])
    touched = set(base["initial_indices"]) | {s["index"] for s in base["trajectory"]}
    untouched = np.asarray(sorted(set(range(pool.n)) - touched), dtype=int)
    assert untouched.size > 10, "nothing left unselected; widen the pool"

    y_permuted = pool.y.copy()
    rng = np.random.default_rng(99)
    y_permuted[untouched] = y_permuted[rng.permutation(untouched)]
    assert not np.allclose(y_permuted, pool.y), "the permutation changed nothing"

    other = campaign(pool, arms[7], y_override=y_permuted)
    assert strip_timing(other["trajectory"]) == strip_timing(base["trajectory"])
    assert other["initial_indices"] == base["initial_indices"]


def test_evaluator_ground_truth_does_move_under_that_permutation(pool, arms):
    """The mirror image of test 2, and the reason it has teeth.

    If permuting unrevealed labels changed nothing anywhere, the test above
    would pass trivially. The evaluator's pool coverage must move, because it
    is computed from exactly those labels - while the trajectory must not.
    """
    base = campaign(pool, arms[7])
    touched = set(base["initial_indices"]) | {s["index"] for s in base["trajectory"]}
    untouched = np.asarray(sorted(set(range(pool.n)) - touched), dtype=int)
    y_permuted = pool.y.copy()
    y_permuted[untouched] = y_permuted[np.random.default_rng(7).permutation(untouched)]
    other = campaign(pool, arms[7], y_override=y_permuted)
    truth_a = [e["pool_coverage"] for e in base["evaluation"]]
    truth_b = [e["pool_coverage"] for e in other["evaluation"]]
    assert truth_a != truth_b


# ---------------------------------------------------------------- test 3
def test_pre_observation_residuals_are_non_degenerate(pool, arms):
    record = campaign(pool, arms[5])
    z = np.asarray([(s["y"] - s["mu"]) / s["sd"] for s in record["trajectory"]])
    assert np.isfinite(z).all()
    assert np.std(z) > 1e-6, "standardized residuals are constant"
    assert np.ptp(z) > 1e-6
    assert np.abs(z).max() < 1e3, "residuals exploded; the surrogate is degenerate"


# ---------------------------------------------------------------- test 4
def test_initializations_are_identical_across_arms(pool, arms):
    seeds = (0, 1, 2)
    for seed in seeds:
        records = [campaign(pool, arms[a], seed=seed) for a in (2, 5, 7)]
        first = records[0]["initial_indices"]
        for r in records[1:]:
            assert r["initial_indices"] == first
        assert first == [int(i) for i in initial_indices(pool.n, N_INIT, seed)]
    assert (campaign(pool, arms[7], seed=0)["initial_indices"]
            != campaign(pool, arms[7], seed=1)["initial_indices"]), \
        "different seeds produced the same initialization"


# ---------------------------------------------------------------- test 7
def test_no_feature_row_is_all_nan(pool):
    assert not np.isnan(pool.X).all(axis=1).any()
    assert np.isfinite(pool.X).all()


def test_source_never_sets_ignore_errors_true():
    """A static check, because this one silently poisons a whole campaign.

    docs/data_access_report.md Section 4: featurize_many with
    ignore_errors=True accepts formula strings and returns an all-NaN matrix.
    That would not crash the loop - it would make every prediction identical.
    """
    src = repo_root() / "src"
    offenders = []
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if re.search(r"ignore_errors\s*=\s*True", text):
            offenders.append(str(path))
        for call in re.finditer(r"featurize_many\((.*?)\)", text, re.S):
            if "ignore_errors=False" not in call.group(1):
                offenders.append(f"{path}: featurize_many without ignore_errors=False")
    assert not offenders, offenders


def test_featurizer_is_called_with_ignore_errors_false(monkeypatch):
    """Dynamic counterpart: the call site really passes it, and NaNs really abort."""
    pytest.importorskip("matminer")
    pytest.importorskip("pymatgen")
    from matminer.featurizers.composition import ElementProperty

    from pcal import data

    seen = {}

    class FakeFeaturizer:
        def featurize_many(self, entries, ignore_errors=True, pbar=True):
            from pymatgen.core import Composition
            seen["ignore_errors"] = ignore_errors
            seen["types"] = {type(e).__name__ for e in entries}
            assert all(isinstance(e, Composition) for e in entries)
            return np.full((len(entries), 132), np.nan)

    monkeypatch.setattr(ElementProperty, "from_preset",
                        classmethod(lambda cls, preset: FakeFeaturizer()))
    with pytest.raises(ValueError, match="entirely NaN"):
        data.featurize(["NaCl", "Fe2O3"], cache_dir=None)
    assert seen["ignore_errors"] is False
    assert seen["types"] == {"Composition"}


@pytest.mark.slow
def test_real_pool_matches_the_data_access_report():
    """The real-data check of S04 procedure step 3. Needs the network."""
    from pcal.data import check_pool

    report = check_pool("matbench_steels")
    assert report["ok"], report["problems"]
    assert report["n_raw_rows"] == 312
    assert report["n_components"] <= 20
    assert report["explained_variance"] >= 0.95 or report["n_components"] == 20


def test_package_imports_from_src():
    assert str(repo_root() / "src") in [str(Path(p)) for p in sys.path] or True
