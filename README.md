# propensity-calibration-al

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23287242.svg)](https://doi.org/10.5281/zenodo.23287242)

**What an active learner can know about its own uncertainty: propensity-corrected
calibration monitoring in pool-based materials optimization.**

Code, pre-registration, result tables and figures for a pre-registered study of
one question: a pool-based active-learning campaign only ever sees labels at the
points it chose, so what is the reliability signal it computes from those points
actually worth, and can the campaign do better from inside its own loop?

The study answers the first half positively and the second half negatively, and
reports both at full length. **Every gate verdict below was fixed by a decision
rule written before any evaluation run existed**
([`preregistration/PREREGISTRATION.md`](preregistration/PREREGISTRATION.md),
frozen at tag [`prereg-v1`](https://github.com/kianfarahmand/propensity-calibration-al/releases/tag/prereg-v1),
archived at [10.5281/zenodo.23287243](https://doi.org/10.5281/zenodo.23287243)).

## What was found

| Gate | Question it decides | Verdict |
|---|---|---|
| **G1** | Does the framing survive a systematic novelty audit? | **passed, narrowed**: four claims surrendered as prior art |
| **G2** | Is the selection-bias gap real on materials pools? | **passed**: up to 12 coverage points, and it flips sign |
| **G3** | Is the propensity-corrected reading a better control input than the naive one? | **failed**: better on 1 dataset of 3, significantly worse on 1 |
| **G4** | Does a controller consuming it beat randomization, the naive monitor, online calibration and localized conformal? | **failed**: 1 of 48 Holm-corrected cells |

Three results, stated at the width the evidence supports:

1. **The premise holds.** Interval coverage measured at the points an acquisition
   policy selected differs from coverage over the whole unlabeled pool by up to
   $0.118$ at a nominal $0.90$ level, and the sign of that difference depends on
   both the policy and the dataset: greedy expected improvement reads $-0.047$
   on `elastic_tensor_2015` and $+0.050$ on `heusler_magnetic`, while on
   `elastic_tensor_2015` expected improvement and max-variance sit on opposite
   sides of zero. Uniform random selection covers zero everywhere, as it must.
   No fixed adjustment repairs a diagnostic that behaves this way.
2. **The standard remedy has a price, and here is the number.** Randomizing the
   acquisition so selection propensities are loggable costs $-0.090$
   $[-0.115, -0.069]$ of top-50 enrichment and $-0.0118$ $[-0.0277, -0.0020]$
   of discovery-curve AUC against the greedy campaign it replaces. These are the
   largest effects anywhere in the study, and they are charged by the enabling
   step rather than by the method.
3. **The control loop does not pay at campaign scale.** The propensity-corrected
   reading beats the naive one on one dataset of three and loses on another; the
   controller built on it beats all four reference arms on no co-primary
   outcome on any dataset (1 of 48 Holm-corrected cells separates from zero in
   its favour, and it only buys back what randomization cost). A post-hoc sweep over the estimator window and the
   effective-sample-size floor (540 extra runs) changes nothing. The mechanism
   is visible in the data: randomizing the acquisition is itself a debiasing
   step, so by the time propensities exist the bias they would correct has
   largely gone, and what remains is estimated at an effective sample size of
   3–8 against a floor of 10.

**Read [`docs/results_summary.md`](docs/results_summary.md) for everything the
project measured**, in the order the pre-registration lists it, with the table
and figure behind every number. A claim absent from that file is a claim the
manuscript does not make.

## Figures

| | |
|---|---|
| [`results/figures/g2_gap_by_policy.png`](results/figures/g2_gap_by_policy.png) | the selection-bias gap and its sign flip (G2) |
| [`results/figures/g3_estimator_vs_truth.png`](results/figures/g3_estimator_vs_truth.png) | each monitor's reading against ground truth |
| [`results/figures/g3_ess.png`](results/figures/g3_ess.png) | the effective-sample-size floor that gates how often the corrected reading exists |
| [`results/figures/g4_discovery_curves.png`](results/figures/g4_discovery_curves.png) | what the campaigns actually found |
| [`results/figures/g4_paired_effects.png`](results/figures/g4_paired_effects.png) | the paired differences behind G4 |
| [`results/figures/f4_randomization_chain.png`](results/figures/f4_randomization_chain.png) | what randomization costs and what the monitor buys back |
| [`results/figures/sensitivity_window.png`](results/figures/sensitivity_window.png) | post-hoc: G3 at four estimator settings |

## Reproducing it

Everything in `results/processed/` and `results/figures/` regenerates from the
committed per-step tables with one command, and CI asserts that it does on every
push:

```bash
conda env create -f environment.yml
conda activate pcal
pip install -e .
pytest -q                      # seven validity tests from PREREGISTRATION.md §14
python scripts/make_figures.py # every gate table and every figure, from committed data
```

To re-run the experiments rather than the analysis, the campaigns themselves are
dispatched by [`.github/workflows/experiments.yml`](.github/workflows/experiments.yml)
on GitHub Actions and sharded by dataset and seed block; a single run is

```bash
pcal run --dataset elastic_tensor_2015 --arm 7 --seed 0 --steps 150
```

Datasets are never committed: every run re-downloads from the sources verified
in [`docs/data_access_report.md`](docs/data_access_report.md) and caches inside
the run.

The analysis set is **1170 runs** = 13 arms × 3 datasets × 30 seeds × 150 steps
(175 500 logged steps, 4.66 runner-hours), at a single acquisition temperature
fixed on a disjoint tuning pair before any evaluation run
([`results/processed/run_inventory.csv`](results/processed/run_inventory.csv)).

## What is where

```
preregistration/   PREREGISTRATION.md  frozen at tag prereg-v1, never edited
                   deviations.md       all 11 departures, each flagged as decided
                                       before or after the affected result was seen
docs/              results_summary.md  every number the project measured
                   novelty_audit.md    the G1 audit: 2859 records, 25 strata, 0.80
                                       measured precision on the decisive band
                   comparison_matrix.csv, novelty_corpus.csv  the audit's evidence
                   data_access_report.md, proposal_v2.md
src/pcal/          data, oracle, surrogate, policies, monitors, controller,
                   baselines, loop, metrics, cli
tests/             the seven validity tests, run in CI on every commit
configs/           datasets, policies, tuning and evaluation grids
scripts/           analysis and figures; make_figures.py is the single entry point
results/processed/ merged per-run and per-step tables (the analysis inputs)
results/figures/   the figure set the manuscript uses
.github/workflows/ tests.yml, experiments.yml, figures.yml
```

## How the claims are kept honest

- **The plan predates the results.** `prereg-v1` is tagged at commit
  `79d819a` with a server-side timestamp, and the same tree is archived under a
  Zenodo version DOI. The frozen file is never edited.
- **Departures are logged, not absorbed.** All eleven are in
  [`preregistration/deviations.md`](preregistration/deviations.md), each stating
  whether it was decided before or after the affected result was visible. Two
  were decided after, and everything downstream of them is reported as
  exploratory.
- **The falsifiers were written in advance** (PREREGISTRATION.md §16) and two of
  them fired. They are reported as findings rather than reinterpreted.
- **The analysis is a merge over committed tables, never a re-run.** Each gate
  script reads `results/processed/*_steps.csv.gz` and is runnable from the
  committed data alone.
- **The figure set regenerates byte-identically** from a clean checkout; the
  `figures` workflow fails if it does not.

## Citing

Cite the pre-registration by its **version** DOI
[10.5281/zenodo.23287243](https://doi.org/10.5281/zenodo.23287243), which pins
the frozen plan permanently; the badge above is the concept DOI, which always
resolves to the latest archived version. [`CITATION.cff`](CITATION.cff) carries
the machine-readable form. The manuscript cites this repository by commit SHA,
so any figure can be traced to the exact revision that produced it.

## License

BSD-3-Clause. See [`LICENSE`](LICENSE).
