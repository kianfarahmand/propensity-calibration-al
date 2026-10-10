# What an active learner can know about its own uncertainty: propensity-corrected calibration monitoring in pool-based materials optimization

**Revision v2.** Supersedes *Knowledge-Adaptive Active Learning for Autonomous
Materials Discovery* (v1). v1 is retained only as provenance; where the two
disagree, this document governs. Section 11 lists the specific corrections.

---

## 1. Summary

A pool-based active-learning campaign chooses where to look, then learns only
from what it chose. Every reliability diagnostic it can compute online — interval
coverage, standardized-residual spread, calibration error — is computed on a
sample drawn by its own acquisition rule, which deliberately selects atypical
candidates (high predicted mean, high predicted variance, or both). The diagnostic
therefore estimates calibration *under the acquisition distribution*, not
calibration over the pool the campaign is actually trying to search.

This matters in particular for any method that *reacts* to such a signal. If the
controller switches acquisition policy when the reliability reading degrades,
and the bias of that reading depends on which policy is active, then the
controller perturbs the bias of its own input. Our synthetic pilot shows this
bias is large enough to flip sign across standard policies (Section 4), so the
concern is not hypothetical.

The proposal is to make the selection probabilities *known* rather than merely
non-uniform — by replacing greedy argmax acquisition with a Boltzmann sampler
carrying an exploration floor — and then to estimate the pool-level calibration
functional with a self-normalized inverse-propensity estimator over a trailing
window of observations, with the debiased reading closing the control loop.

## 2. The defect this attacks

Let the surrogate at step `t` have predictive mean `mu_t(x)` and predictive
standard deviation `sigma_t(x)` on the unlabeled pool `U_t`. The quantity a
campaign would like to know is the pool-marginal coverage at nominal level
`alpha`,

```
C_t(alpha) = (1 / |U_t|) * sum_{x in U_t} 1{ |y(x) - mu_t(x)| <= q_alpha * sigma_t(x) }
```

where `q_alpha` is the standard-normal quantile. `C_t(alpha)` is **unobservable
during a real campaign** — it depends on labels the campaign has not bought — but
it is **exactly computable by the evaluator on a retrospective benchmark**, where
all labels exist and are merely withheld from the learner. That asymmetry is what
makes the question empirically answerable at zero marginal cost.

The naive online monitor replaces the pool average with an average over the
points the campaign happened to select. Those points are not a random sample of
the pool; they are the output of the acquisition rule. The resulting estimator is
biased by construction, and nothing about collecting more of them removes the
bias.

## 3. Central question

Narrowed from v1 to exactly two testable sub-questions:

- **Q1 (measurement).** Can a campaign obtain an approximately unbiased estimate
  of its surrogate's pool-level calibration from the biased sample it chose?
- **Q2 (control).** Does acting on that corrected estimate beat acting on the
  biased one?

Q1 is a measurement question with a ground truth available on retrospective
benchmarks. Q2 is a downstream-utility question. The design is deliberately
arranged so that a positive answer to Q1 with a null answer to Q2 remains a
complete and publishable contribution — a measurement result plus an honest
negative result about the control loop built on it.

## 4. Evidence so far

A prior session built a leak-free pool-based loop and ran it on a synthetic 5-D
Styblinski-Tang problem: 1500-point pool, Matern-5/2 GP surrogate, 10 initial
points, 50 selections, 12 seeds, initializations matched across arms. Four
leakage tests pass (revealed labels equal initial plus selected; permuting
never-selected labels leaves the trajectory identical; pre-observation residuals
are non-degenerate; initializations are identical across arms).

Two findings:

1. **The surrogate is overconfident on the pool.** Its nominal 90% predictive
   interval covers about 0.75 of the pool.
2. **The selection-bias gap depends on the policy, including its sign.**
   Coverage measured at selected points minus pool coverage:

   | policy | coverage gap (selected − pool) |
   |---|---|
   | Expected improvement | −0.125 |
   | Upper confidence bound | −0.070 |
   | Max variance | +0.244 |
   | Random | −0.037 |

   Sanity checks behave as expected: EI and UCB beat random; max-variance
   sampling is worse than random.

The sign flip between exploitative policies and the pure-exploration policy is
the specific defect this project attacks. These numbers are synthetic and carry
no materials claim; reproducing the gap on real materials data is gate G2.

## 5. Method

### 5.1 Randomized acquisition with known propensities

Replace greedy selection with a Boltzmann sampler over the acquisition score
`a_t(x)`, mixed with a uniform exploration floor:

```
pi_t(x) = (1 - eps) * softmax( a_t(x) / tau ) + eps / |U_t|,   x in U_t
```

The floor `eps > 0` guarantees `pi_t(x) >= eps / |U_t|` everywhere, so importance
weights are bounded and the estimator below has finite variance. The temperature
`tau` controls the exploration/estimation trade-off: `tau -> 0` recovers greedy
argmax and degenerates the estimator; large `tau` buys effective sample size at
the cost of optimization performance. Characterizing that frontier is an
experiment in its own right, not a tuning afterthought.

### 5.2 Self-normalized inverse-propensity estimator

Over a trailing window of accepted steps `s`, with the realized selection `x_s`
and its logged propensity `pi_s(x_s)`, weight each observation by

```
w_s = 1 / ( |U_s| * pi_s(x_s) )
```

and form the self-normalized (Hajek) estimate of coverage,

```
C_hat_t(alpha) = sum_s w_s * z_s(alpha)  /  sum_s w_s
```

where `z_s(alpha) = 1{ |y(x_s) - mu(x_s)| <= q_alpha * sigma(x_s) }`. Weights are
clipped at a pre-registered threshold. The effective sample size

```
ESS = ( sum_s w_s )^2 / sum_s w_s^2
```

is reported at every step, and readings below a pre-registered ESS floor are
suppressed rather than acted on. Self-normalization is chosen over the plain
Horvitz-Thompson form deliberately: it is biased but bounded in `[0, 1]`, which
matters because the estimand is a probability and the controller consumes it.

### 5.3 Model drift, and the leave-one-out refit

A complication that must be handled explicitly rather than assumed away: the
windowed residual `z_s` is computed under the surrogate as it stood at step
`s−1`, while the estimand `C_t` refers to the surrogate at step `t`. The window
therefore carries a **model-drift bias** distinct from selection bias, and the
two can either cancel or compound.

Primary fix: a leave-one-out refit that recomputes each windowed residual under
the *current* surrogate with that point held out of the training set. For an
exact GP this is a rank-one downdate rather than a full refit. The cost is a
per-step multiplier equal to the window length, which is the dominant term in the
compute budget (Section 9). The no-refit variant is retained as an ablation so
the contribution of this correction is measurable rather than asserted.

### 5.4 Controller

The controller consumes `C_hat_t` (and its ESS) and acts in one of two
pre-registered forms:

- **Recalibration form.** Fit a scalar (or low-dimensional) variance-inflation
  factor so that the debiased coverage estimate matches nominal, and apply it to
  `sigma_t` inside the acquisition function.
- **Policy-switch form.** Switch among acquisition policies when the debiased
  reading crosses a pre-registered band.

Both forms are specified completely before any evaluation run. Controller
parameters are fixed on a tuning pair of datasets and are never touched on the
evaluation set.

## 6. Relation to prior work — what is *not* claimed

Propensity correction of an active learner's self-inflicted sampling bias is
**established prior art**. Importance-weighted active learning (Beygelzimer et
al.) and UPAL for the pool-based case already use inverse-propensity weighting to
obtain unbiased risk estimates under adaptive sampling. This project presents
IPW as a borrowed device, never as a proposed novel technique. Any draft sentence
implying otherwise is a defect.

Adjacent families that the Step 2 audit must position against explicitly:
CA-SMART (switching on whether an observation falls outside a 95% prediction
interval — structurally the closest competitor), adaptive acquisition portfolios
(GP-Hedge, No-PASt-BO, LMABO), and online calibration inside BO (Deshpande et
al., LOCBO, and Pion & Vazquez on calibration under adaptive designs).

The claim that remains, stated at its actual width: the *estimand* is a
calibration functional of the surrogate's predictive distribution over the pool,
inside pool-based Bayesian optimization, and the debiased reading closes the
control loop. Gate G1 (Section 8) decides whether even that survives contact with
the literature.

## 7. Experimental design

### 7.1 Arms

Roughly eleven arms, all with matched initializations:

1. Random
2. Expected improvement (greedy)
3. Upper confidence bound (greedy)
4. Max variance (greedy)
5. **Boltzmann-EI, no monitor** — *the single most important baseline*
6. Boltzmann-EI + naive (uncorrected) monitor controller
7. Boltzmann-EI + propensity-corrected monitor controller — the proposed method
8. CA-SMART
9. GP-Hedge
10. No-PASt-BO
11. One-off global uncertainty scaling
12. Online conformal recalibration

Arm 5 is load-bearing. It is the only comparison that separates the benefit of
debiasing from the mere effect of randomizing the acquisition. A result that
beats greedy EI but not Boltzmann-EI is a result about randomization, not about
calibration monitoring, and must be reported as such.

### 7.2 Datasets

Zero-cost public sets via matminer/Matbench: experimental band gap, steel
fatigue, dielectric, and JARVIS DFT properties. Composition descriptors (Magpie)
are fixed before any comparative run. The tuning/evaluation dataset split is
pre-registered so controller parameters are never tuned on evaluation data.

**At least one experimental dataset with real measurement noise is mandatory.**
DFT targets are deterministic, so "uncertainty" on a DFT benchmark is purely
model misspecification with no aleatoric component; a reliability story told only
on DFT data would be telling half the story and must say so explicitly.

### 7.3 Metrics

Co-primary, all pre-specified:

- Simple regret (retained, but known to saturate on pools with many
  near-optimal candidates — which is precisely why it is not the sole primary)
- Top-k enrichment at k in {10, 50}
- Normalized discovery-curve AUC

Secondary: policy-switch count, ESS trajectory, runtime.

### 7.4 Statistics

Paired differences against the matched baseline, with bootstrap confidence
intervals resampled **over runs, never over iterations** — iterations within a
run are dependent and bootstrapping over them would understate uncertainty by
construction.

### 7.5 Leakage and bookkeeping tests

The prototype's four passing tests port forward, plus two new ones:

5. Logged propensities form a valid distribution over the unlabeled pool at every
   step and satisfy the `eps` floor.
6. The leave-one-out refit genuinely excludes the held-out point from training.

## 8. Decision gates

| Gate | Test | If it fails |
|---|---|---|
| **G1** | Does the framing survive a systematic OpenAlex novelty audit? | Re-frame or stop before writing benchmark code |
| **G2** | Does the selection-bias gap appear on real materials data? | The premise is synthetic-only; report that |
| **G3** | Does the propensity-corrected estimator beat the naive monitor at usable ESS? | Q1 answered negatively; measurement paper reports why |
| **G4** | Does the controller beat both the naive-monitor arm and the randomize-only arm? | Q2 answered negatively; still a complete paper |

Passing G3 and failing G4 is an anticipated and acceptable outcome: a
measurement-plus-negative-result contribution.

## 9. Compute and cost

The full grid — roughly 11 arms x 5 datasets x 30 seeds x 150 steps, with the
leave-one-out refit multiplying per-step cost by the window length — exceeds the
author's local machine (4 cores, 8 GB). Primary runner is **GitHub Actions on
this public repository**: standard runners are free and unlimited there, at
4 vCPU / 16 GB (versus 2 vCPU / 8 GB on private repos), with a 6-hour per-job
cap, sharded by dataset and seed block. Kaggle CPU notebooks (12-hour sessions,
20 GB disk) are the backup. The code must also run locally at reduced seed count,
so reproduction never depends on a hosted free tier remaining free.

One known bottleneck, to be settled *before* any comparative run: GP fitting on
132-dimensional Magpie descriptors is dominated by anisotropic length-scale
optimization, not by pool size. A dimensionality reduction (e.g. PCA) or an
isotropic kernel must be chosen and pre-registered rather than selected after
seeing timings.

The entire project — compute, data, and publication — costs nothing. Venue:
arXiv preprint plus either TMLR (peer reviewed, open access, no APC) if the work
lands as methods-and-measurement, or Computational Materials Science on the
subscription route (no APC unless open access is elected) if the materials
framing proves stronger. The venue is chosen after the evidence review, never
before.

## 10. Reproducibility and provenance

The pre-registration is committed and tagged (`prereg-v1`) before any evaluation
run; the tag's server-side timestamp is the evidence that the analysis plan
predates the results, which is the objection this project is most exposed to.
Departures are logged in `preregistration/deviations.md` rather than by editing
the frozen file. Figures are generated in this repository and cited by commit SHA
in the manuscript. Datasets are never committed.

## 11. Corrections to v1

1. **Title.** v1's "Knowledge-Adaptive ... Autonomous Materials Discovery"
   oversells: nothing in the method concerns knowledge representation or
   autonomous operation. It is reliability-aware acquisition on a retrospective
   benchmark. Retitled accordingly.
2. **Broken LaTeX** (`e*t`, `x\_{t+1}`) and `utm_source=chatgpt.com` tracking
   tags on every link are removed.
3. **An uncited reference.** v1 cited a "2025 Computational Materials Science
   study on data and model uncertainty in active learning" that did not surface
   in two independent searches. Step 2 either resolves it to a real record or
   deletes it. It is not cited again until then.
4. **Scope.** The central question is narrowed to Q1 and Q2 in Section 3.
5. **Dropped.** The eight-week timeline and the fixed Computational Materials
   Science target are no longer part of the plan.

## 12. Immediate next steps

1. **Step 1.** Build the environment; verify which materials datasets actually
   download; freeze `environment.yml` from the solved environment. Deliverable:
   `docs/data_access_report.md`.
2. **Step 2.** OpenAlex novelty audit across the six query families, with full
   texts for the decisive papers. Gate G1. Deliverables:
   `docs/novelty_audit.md`, `docs/comparison_matrix.csv`.
3. **Step 3.** Write and freeze `preregistration/PREREGISTRATION.md`; tag the
   commit `prereg-v1`. No evaluation run before this tag exists.
