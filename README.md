# propensity-calibration-al

**What an active learner can know about its own uncertainty: propensity-corrected
calibration monitoring in pool-based materials optimization.**

A pool-based active-learning campaign only ever observes labels at the points it
chose to observe, and it chooses them deliberately atypically. Any reliability
signal it computes from those points — calibration, coverage, residual spread —
is therefore measured under a sampling distribution of the learner's own making.
This repository asks whether a running campaign can recover an approximately
unbiased estimate of its surrogate's *pool-level* calibration from that biased
sample, and whether acting on the corrected estimate beats acting on the raw one.

The mechanism is borrowed, not invented: randomize the acquisition with a
Boltzmann sampler plus an exploration floor so selection propensities are known
and bounded away from zero, then apply self-normalized inverse-propensity
weighting to the observed standardized residuals. Importance-weighted active
learning (Beygelzimer et al.) and UPAL already do this for *risk* estimation.
The contribution claimed here is narrow and is stated as such throughout: the
estimand is a calibration functional of the surrogate's predictive distribution
over the unlabeled pool, inside pool-based Bayesian optimization, with the
debiased reading closing the control loop.

See [`docs/proposal_v2.md`](docs/proposal_v2.md) for the full specification.

## Status

| Stage | State |
|---|---|
| Revised proposal | drafted (`docs/proposal_v2.md`) |
| Step 1 — environment and data access | pending (`docs/data_access_report.md`) |
| Step 2 — OpenAlex novelty audit, gate G1 | pending (`docs/novelty_audit.md`) |
| Step 3 — pre-registration, tagged `prereg-v1` | pending (`preregistration/PREREGISTRATION.md`) |
| Steps 4+ — benchmark, experiments, manuscript | not started |

No evaluation run happens before the pre-registration commit is tagged.

## Repository layout

```
README.md  LICENSE  pyproject.toml  environment.yml  CITATION.cff
.github/workflows/    tests.yml, experiments.yml, figures.yml
preregistration/      PREREGISTRATION.md (frozen after tagging), deviations.md
docs/                 proposal_v2.md, novelty_audit.md, comparison_matrix.csv,
                      data_access_report.md
configs/              datasets.yaml, policies.yaml, grid_tuning.yaml, grid_eval.yaml
src/pcal/             data, oracle, surrogate, policies, monitors, controller,
                      baselines, loop, metrics, cli
tests/                test_leakage.py, test_propensity.py, test_loo_refit.py
results/              raw/, processed/, figures/
scripts/              make_shards.py, merge_results.py
```

Datasets are never committed. The pipeline re-downloads materials data and caches
it inside the run; results tables and figures are committed normally.

## Install

```bash
conda env create -f environment.yml
conda activate pcal
pip install -e .
pytest -q
```

## Reproducibility

The pre-registration is committed and tagged before any evaluation run, so the
tag's server-side timestamp is the evidence that the analysis plan predates the
results. Any departure from it is recorded in `preregistration/deviations.md`
rather than by editing the frozen file. Manuscript figures are generated here
under `results/figures` and cited by commit SHA.

## License

See `LICENSE`.
