# Pre-registration

**Project.** What an active learner can know about its own uncertainty:
propensity-corrected calibration monitoring in pool-based materials optimization.

**Status.** Frozen at tag `prereg-v1`. This file is not edited after the tag.
Every departure is logged in `preregistration/deviations.md`, with its date, its
reason, and whether it was decided before or after seeing the affected result.

**Preconditions already discharged.** Step 1 (`docs/data_access_report.md`)
confirmed credential-free access to the datasets named below. Step 2
(`docs/novelty_audit.md`) is Gate G1 and it passed, with four claims surrendered;
Section 2 below states the claim at the width that survived. No evaluation run
has been executed at the time of this tag.

---

## 1. Estimand

Let the surrogate at step `t` have predictive mean `mu_t(x)` and predictive
standard deviation `sigma_t(x)`. Let `U_t` be the unlabeled pool at step `t`.
The target quantity is pool-marginal coverage at nominal level `alpha`:

```
C_t(alpha) = (1 / |U_t|) * sum_{x in U_t} 1{ |y(x) - mu_t(x)| <= q_alpha * sigma_t(x) }
```

with `q_alpha` the standard normal quantile. This is an instance of what Pion
and Vazquez (arXiv:2512.05611) call design-marginal `mu`-coverage, with `mu` the
uniform measure on the unlabeled pool. It is unobservable to a real campaign and
exactly computable by the evaluator on a retrospective benchmark.

`alpha = 0.90` is primary. `alpha in {0.50, 0.80, 0.95}` are reported as
secondary and are not used for any gate decision.

## 2. The claim under test

Four things are explicitly **not** claimed, each because the Step 2 audit found
prior art: inverse-propensity correction of active-selection bias; recognition
that a design algorithm induces its own covariate shift and that weighting fixes
it; closing a control loop on a shift-corrected uncertainty object; and the
design-marginal calibration functional itself together with the standardized
leave-one-out residual construction.

What is claimed, and what this study tests: that the design-marginal calibration
of a surrogate can be estimated **during** an adaptive campaign, from the sample
the campaign itself selected, using propensities that are known because the
acquisition was deliberately randomized; that this estimate is materially less
biased than the naive selected-sample reading; and that acting on it improves
the campaign.

## 3. Hypotheses

- **H1 (premise).** On real materials pools, coverage measured at the selected
  points differs from pool-marginal coverage, and the sign of that difference
  differs between at least two acquisition policies.
- **H2 (measurement).** The self-normalized inverse-propensity estimator of
  `C_t(0.90)` has smaller absolute bias against the evaluator's ground truth than
  the naive selected-sample estimator, at effective sample sizes the campaign
  actually reaches.
- **H3 (control).** A controller consuming the corrected estimate outperforms
  controllers consuming the naive estimate, outperforms randomized acquisition
  with no monitor, and outperforms online calibration that does not use
  propensities.
- **H4 (cost).** The randomization needed to make propensities usable costs
  optimization performance, and that cost is quantified rather than assumed
  negligible.

H1 is Gate G2, H2 is Gate G3, H3 is Gate G4. H4 has no gate; it is reported
whatever it shows.

## 4. Data

All sets are fetched at run time from the sources verified in
`docs/data_access_report.md`. No dataset is committed to the repository.

**Tuning pair** — every free parameter below is fixed on these two and never
re-touched:

| dataset | n | target | type |
|---|---|---|---|
| `matbench_steels` | 312 | yield strength | experimental |
| `dielectric_constant` | 1056 | `poly_electronic` | DFT |

**Evaluation set** — untouched until every parameter is frozen:

| dataset | n | target | type |
|---|---|---|---|
| `matbench_expt_gap` | 4604 | experimental band gap (eV) | experimental |
| `elastic_tensor_2015` | 1181 | `K_VRH` | DFT |
| `heusler_magnetic` | 1153 | `e_form` | DFT |

Both an experimental (aleatoric-bearing) and a DFT (deterministic-target) set
appear in each half, so the two noise regimes are never confounded with the
tuning/evaluation split.

### 4.1 Pool construction, fixed before any run

1. Parse each formula with `pymatgen.core.Composition`. Passing formula strings
   to a matminer featurizer is prohibited; `ignore_errors=False` is mandatory at
   every featurization call site (`docs/data_access_report.md`, Section 4).
2. Featurize with `matminer` `ElementProperty.from_preset("magpie")`, 132
   dimensions. Reject the run if any feature row is all-NaN.
3. Collapse duplicate reduced formulas, keeping the median target value.
4. Drop rows with missing or non-finite targets.
5. If the pool still exceeds 3000 entries, subsample to exactly 3000 without
   replacement using `numpy.random.default_rng(20261010)`. The same subsample is
   used by every arm and every seed.
6. Standardize features (mean 0, variance 1) on the full pool, then PCA to the
   smallest number of components retaining 95% of variance, capped at 20. Both
   are unsupervised and use no labels, so neither leaks.

Targets are standardized to zero mean and unit variance on the full pool for
modelling; all reported regret and enrichment figures are back-transformed to
the original units.

## 5. Surrogate

Exact Gaussian process regression (`sklearn.gaussian_process`), kernel
`ConstantKernel * Matern(nu=2.5, isotropic) + WhiteKernel`. The isotropic kernel
is chosen over per-dimension length scales deliberately and in advance: anisotropic
optimization over 132 Magpie dimensions is the known compute bottleneck for this
project, and the PCA step above already supplies the dimensionality reduction.
Hyperparameters are refit by marginal-likelihood maximization at every step with
`n_restarts_optimizer=2`. Objective is maximization; minimization targets are
negated at load.

## 6. Acquisition

Boltzmann sampling over the acquisition score with a uniform exploration floor:

```
pi_t(x) = (1 - eps) * softmax( a_t(x) / tau_t ) + eps / |U_t|,   x in U_t
```

- `eps = 0.05`, fixed, not tuned.
- `tau_t = c * MAD_t`, where `MAD_t` is the median absolute deviation of `a_t`
  over the current pool. Scale-free by construction, so one `c` transfers across
  datasets. `c` is selected on the tuning pair only, from the grid
  `{0.25, 0.5, 1.0, 2.0, 4.0}`, by the criterion in Section 10.
- Realized propensity `pi_t(x_t)` is logged at selection time, before the label
  is revealed.

## 7. Estimator

Over a trailing window of `W` accepted steps:

```
w_s      = 1 / ( |U_s| * pi_s(x_s) )
C_hat_t  = sum_s w_s * z_s  /  sum_s w_s
ESS_t    = ( sum_s w_s )^2 / sum_s w_s^2
```

with `z_s = 1{ |y(x_s) - mu(x_s)| <= q_alpha * sigma(x_s) }`.

- `W = 30`, fixed.
- Weights are clipped at the 95th percentile of the weights inside the current
  window before normalization. Clipping is applied always, not adaptively.
- Readings with `ESS_t < 10` are suppressed: the controller holds its previous
  action and the step is recorded as suppressed. The suppression rate is a
  reported secondary outcome.
- Self-normalized (Hajek) rather than Horvitz-Thompson, because the estimand is
  a probability and the controller consumes it; the bound in `[0, 1]` is worth
  the finite-sample bias.

### 7.1 Model drift

`z_s` is computed under the surrogate as it stood at step `s-1`, while `C_t`
refers to the surrogate at step `t`. The primary specification removes this by a
**leave-one-out refit**: each windowed residual is recomputed under the current
surrogate with that point held out of the training set, as an exact rank-one GP
downdate. The no-refit variant is retained as a pre-registered ablation so the
correction's contribution is measured rather than asserted.

## 8. Controller

Primary form, **recalibration**: fit a scalar variance-inflation factor
`lambda_t > 0` such that the corrected coverage estimate matches nominal, i.e.
solve `C_hat_t(alpha; lambda_t * sigma_t) = alpha` by bisection on
`log lambda_t` over `[-2, 2]`, and apply `lambda_t * sigma_t` inside the
acquisition function at step `t+1`. `lambda_t` is updated by exponential
smoothing with factor 0.3 to prevent oscillation.

Secondary form, **policy switch**: switch between EI and max-variance when the
debiased reading leaves the band `alpha +/- 0.05` for three consecutive
non-suppressed steps. Reported separately; it does not enter the primary gate.

## 9. Arms

All arms share matched initializations: the same 10 seed points per
(dataset, seed) pair, drawn by `numpy.random.default_rng(seed)`.

| # | arm | role |
|---|---|---|
| 1 | Random | floor |
| 2 | Expected improvement, greedy | standard practice |
| 3 | Upper confidence bound, greedy | standard practice |
| 4 | Max variance, greedy | pure exploration control |
| 5 | **Boltzmann-EI, no monitor** | isolates randomization from debiasing |
| 6 | Boltzmann-EI + naive selected-sample monitor | the uncorrected controller |
| 7 | **Boltzmann-EI + propensity-corrected monitor** | the proposed method |
| 8 | Boltzmann-EI + online calibration (Deshpande and Kuleshov 2021 form) | no-propensity competitor |
| 9 | Boltzmann-EI + localized online conformal (LOCBO 2025 form) | no-propensity competitor |
| 10 | GP-Hedge portfolio | switching without a reliability signal |
| 11 | One-off global uncertainty scaling, fit at step 50 | static recalibration control |
| 12 | Arm 7 with no leave-one-out refit | ablation: drift correction |
| 13 | Arm 7 with no weight clipping | ablation: variance control |

Arms 5, 8 and 9 are load-bearing. Arm 5 is the only comparison separating the
benefit of debiasing from the effect of randomizing. Arms 8 and 9 are the
methods that solve the same problem without propensities; if arm 7 does not beat
them, the contribution is a measurement note and will be reported as one.

CA-SMART is **not** an arm. The Step 2 audit could not locate it in the
literature under that name, and an unverifiable baseline cannot be implemented.

## 10. Protocol

- 150 selection steps per run, one point per step, after 10 seed points.
- 30 seeds per (arm, dataset) on the evaluation set; 10 seeds on the tuning pair.
- Tuning criterion for `c` (Section 6): on the tuning pair only, the value
  minimizing mean absolute error of `C_hat_t(0.90)` against the evaluator's
  ground truth over steps 50-150, averaged across the two tuning datasets and
  their 10 seeds, among values whose median ESS at step 100 is at least 10.
  Ties broken toward the smaller `c`.
- Once `c` is fixed, the tuning datasets are not used again for any purpose.

## 11. Outcomes

**Co-primary** (optimization), each computed per run and compared pairwise:

1. Simple regret at step 150, in original target units.
2. Top-k enrichment at k = 10 and k = 50 (fraction of the pool's true top-k
   recovered), reported at step 150.
3. Normalized discovery-curve AUC over the 150 steps.

**Primary measurement outcome** (for H2 and G3): mean absolute deviation of the
estimator from the evaluator's `C_t(0.90)`, over non-suppressed steps in 50-150.

**Secondary**: ESS trajectory, suppression rate, `lambda_t` trajectory,
policy-switch count, wall-clock per step, and coverage at
`alpha in {0.50, 0.80, 0.95}`.

## 12. Inference

- All comparisons are **paired on (dataset, seed)** against the named comparator
  arm.
- Uncertainty is a 95% bias-corrected accelerated bootstrap with 10000
  resamples, **resampled over runs and never over iterations**, because
  iterations within a run are dependent.
- Multiplicity: the three co-primary optimization outcomes are corrected by
  Holm's method within each (comparison, dataset). Secondary outcomes are not
  corrected and are reported as descriptive.
- No interim analysis, no optional stopping, no post-hoc addition of datasets,
  arms or metrics. If a run fails to complete it is re-run with the same seed;
  seeds are never dropped for their results.

## 13. Decision rules

| Gate | Rule | Status |
|---|---|---|
| **G1** | Does the framing survive the novelty audit? | **Passed, narrowed** (`docs/novelty_audit.md`) |
| **G2** | On at least 2 of 3 evaluation datasets, the mean selected-minus-pool coverage gap over steps 50-150 has a 95% CI excluding 0 for at least one greedy policy, **and** the point estimates of that gap have opposite signs for at least two of the four policies in arms 1-4 | pending |
| **G3** | Arm 7's mean absolute deviation from ground-truth `C_t(0.90)` is lower than arm 6's, paired 95% CI on the difference excluding 0, on at least 2 of 3 evaluation datasets | pending |
| **G4** | Arm 7 beats arms 5, 6, 8 and 9 on at least 2 of the 3 co-primary outcomes after Holm correction, on at least 2 of 3 evaluation datasets | pending |

Failing G4 while passing G2 and G3 is an anticipated, acceptable, and publishable
outcome: a measurement result plus an honest negative result about the control
loop built on it. The manuscript is written to that structure from the start, so
there is no incentive to reinterpret a null.

Failing G2 means the premise does not hold on real materials data; the result is
then a negative replication of the synthetic pilot and is reported as such.

## 14. Validity tests

The run harness aborts if any of these fail. They run in CI on every commit.

1. Revealed labels equal the initial set plus the selected set, exactly.
2. Permuting the labels of never-selected pool members leaves the trajectory
   bit-identical.
3. Pre-observation standardized residuals are non-degenerate.
4. Initializations are identical across arms for a given (dataset, seed).
5. Logged propensities form a valid distribution over `U_t` at every step
   (non-negative, sum to 1 within 1e-9) and satisfy `pi_t(x) >= eps / |U_t|`.
6. The leave-one-out refit excludes the held-out point from the training set.
7. No feature row is all-NaN, and no featurizer is called with `ignore_errors=True`.

## 15. Compute

Primary runner is GitHub Actions on this public repository, sharded by dataset
and seed block, within the 6-hour per-job cap. Kaggle CPU notebooks are the
backup. The code runs locally at reduced seed count so reproduction never depends
on a free tier remaining free. Raw per-run records land in `results/raw/`,
merged tables in `results/processed/`, figures in `results/figures/`. The
manuscript cites code by commit SHA.

## 16. What would falsify the contribution

Stated plainly so it cannot be renegotiated later:

- If the selected-minus-pool coverage gap on real materials pools is small and
  single-signed, the premise fails and the controller has nothing to correct.
- If the inverse-propensity estimator's variance at reachable effective sample
  sizes exceeds the naive estimator's bias, the correction is not worth its cost.
- If arms 8 and 9 match or beat arm 7, propensity logging buys nothing over
  online calibration and the contribution reduces to the measurement in H1 and H2.
- If the randomization cost in H4 is large enough that arm 5 loses badly to arm 2,
  the whole approach is impractical regardless of how well the estimator works.
