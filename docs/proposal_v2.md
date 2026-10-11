# What an active learner can know about its own uncertainty: propensity-corrected calibration monitoring in pool-based materials optimization

**Revision v2.** Supersedes *Knowledge-Adaptive Active Learning for Autonomous
Materials Discovery* (v1). v1 is retained only as provenance; Section 11 lists
the specific corrections.

> **Status, 2026-10-11.** This is the design document the project was planned
> from, corrected in session M01 so that nothing in it contradicts the novelty
> audit or the frozen plan. It is **not** the governing document. The plan that
> governs is [`../preregistration/PREREGISTRATION.md`](../preregistration/PREREGISTRATION.md),
> frozen at tag `prereg-v1`, with every departure logged in
> [`../preregistration/deviations.md`](../preregistration/deviations.md); what
> was actually measured is in
> [`results_summary.md`](results_summary.md). Where this file and those
> disagree, they win. Sections 6 to 9 and 11 below were brought in line with
> them; Sections 1 to 5 are the original argument and are unchanged.

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

The Step 2 audit ([`novelty_audit.md`](novelty_audit.md): 25 OpenAlex strata,
2859 records, 2357 after deduplication, two-pass screen at 0.80 measured
precision on the decisive band) found the framing survives, but only after
**four claims are surrendered outright**. They are prior art and are never
presented otherwise:

1. **Inverse-propensity correction of an active learner's self-inflicted
   sampling bias.** Beygelzimer et al. (2009), Ganti and Gray (2011), and
   Farquhar et al. (2021) own it. Their corrected functional is a *risk*.
2. **Recognising that a design algorithm induces its own covariate shift, and
   correcting it by weighting.** Fannjiang et al. (PNAS 2022) named this
   feedback covariate shift; their estimand is coverage *at the designed
   points*, in a single design round.
3. **Closing a control loop with a shift-corrected uncertainty object.**
   Stanton et al. (2022) feed covariate-shift-corrected conformal prediction
   sets into the acquisition.
4. **The design-marginal calibration functional and the standardized
   leave-one-out residual construction of Section 5.3.** Pion and Vazquez
   (arXiv:2512.05611) formalise the estimand this project calls pool-marginal
   coverage, and build their estimators on the same residual device. Their
   design measure is given and their residuals come from cross-validation on an
   existing design.

**CA-SMART is struck.** v1 and the first draft of this document called it
"structurally the closest competitor". The audit could not locate any record
under that name in OpenAlex, and targeted searches for the described behaviour
returned nothing. An unverifiable baseline cannot be implemented or cited, so
it is not an arm and is not referenced again. If a citation is ever produced it
would enter as a new arm with a logged deviation.

**Two works contest the contribution seriously enough to be mandatory
comparators rather than related work:** Deshpande et al. (2021), *Online
Calibrated and Conformal Prediction Improves Bayesian Optimization*, and Kim et
al. (IEEE TSP 2025), *Robust Bayesian Optimization via Localized Online
Conformal Prediction*. Both attack the same action-dependent-data problem with
online learning instead of propensities, and their guarantees hold on the
realised query stream. They enter the design as arms 8 and 9, and Gate G4 is
hardened accordingly (Section 8). If the pool-marginal reading cannot be shown
to beat an online-calibration reading, the contribution reduces to the
measurement — which is what happened.

**Acquisition portfolios** (GP-Hedge: Brochu, Hoffman and de Freitas 2010;
No-PASt-BO: Vasconcelos et al. 2019) switch acquisition functions on observed
reward with no reliability signal; they are policy-switching controls in the arm
list, not prior art for the estimand.

**Materials-side work this project builds on**, all of which measures
calibration on the selected sample without raising the selection-bias question —
that is, all of which uses the arm-6 reading: Thomas-Mitchell, Hawe and Popelier
(MLST 2023), *Calibration of uncertainty in the active learning of machine
learning force fields*; Jiang et al. (J. Mater. Inf. 2026), *Uncertainty
estimation affects predictor selection and its calibration improves materials
optimization*; and Li et al., *A study of the role of data and model uncertainty
in active learning*, Comput. Mater. Sci. 247:113512 (doi
10.1016/j.commatsci.2024.113512).

The claim that remains, stated at its actual width: **estimating the
design-marginal calibration of a surrogate from the adaptively selected sample,
using propensities known because the acquisition was deliberately randomized,
during the campaign, and letting that estimate close the control loop** —
together with the measurement that the naive reading's bias flips sign with the
acquisition policy.

## 7. Experimental design

### 7.1 Arms

**Thirteen arms**, as pre-registered (PREREGISTRATION.md §9), all with matched
initializations — the same ten seed points per (dataset, seed) pair:

| # | arm | role |
|---|---|---|
| 1 | Random | floor; unbiased monitor by construction |
| 2 | Expected improvement, greedy | standard practice |
| 3 | Upper confidence bound, greedy | standard practice |
| 4 | Max variance, greedy | pure exploration control |
| 5 | **Boltzmann-EI, no monitor** | isolates randomization from debiasing |
| 6 | Boltzmann-EI + naive selected-sample monitor | the uncorrected controller |
| 7 | **Boltzmann-EI + propensity-corrected monitor** | the proposed method |
| 8 | Boltzmann-EI + online calibration (Deshpande et al. form) | no-propensity competitor |
| 9 | Boltzmann-EI + localized online conformal (LOCBO form) | no-propensity competitor |
| 10 | GP-Hedge portfolio | switching without a reliability signal |
| 11 | One-off global uncertainty scaling at step 50 | static recalibration control |
| 12 | Arm 7 with no leave-one-out refit | ablation: drift correction |
| 13 | Arm 7 with no weight clipping | ablation: variance control |

Arms 5, 8 and 9 are load-bearing. Arm 5 is the only comparison that separates
the benefit of debiasing from the mere effect of randomizing the acquisition: a
result that beats greedy EI but not Boltzmann-EI is a result about
randomization, not about calibration monitoring, and must be reported as such.
Arms 8 and 9 are the methods that solve the same problem without propensities;
they are implemented as adaptations of their sources rather than
reimplementations (deviations entry 6).

CA-SMART and No-PASt-BO are not arms — the first because it could not be
located in the literature, the second because GP-Hedge already supplies the
portfolio control.

### 7.2 Datasets

Zero-cost public sets via matminer/Matbench, with the tuning/evaluation split
pre-registered so no controller parameter is ever tuned on evaluation data.

- **Tuning pair** (every free parameter fixed here, then never touched again):
  `matbench_steels` (312, experimental yield strength) and
  `dielectric_constant` (1056, DFT `poly_electronic`).
- **Evaluation set**: `matbench_expt_gap` (4604, experimental band gap),
  `elastic_tensor_2015` (1181, `K_VRH`) and `heusler_magnetic` (1153,
  `e_form`); after pool construction, 2990, 1077 and 868 candidates.

Composition descriptors (Magpie, 132-d) are fixed before any comparative run.

**At least one experimental dataset with real measurement noise is mandatory**,
and both halves of the split carry one. DFT targets are deterministic, so
"uncertainty" on a DFT benchmark is purely model misspecification with no
aleatoric component; a reliability story told only on DFT data would be telling
half the story and must say so explicitly.

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

The binding statement of these rules is PREREGISTRATION.md §13; this table is a
summary, with the verdicts added after the fact. Full numbers in
[`results_summary.md`](results_summary.md).

| Gate | Test | Verdict |
|---|---|---|
| **G1** | Does the framing survive a systematic OpenAlex novelty audit? | **passed, narrowed** — four claims surrendered (Section 6) |
| **G2** | Does the selection-bias gap appear on real materials data, with opposite signs for at least two policies? | **passed** — up to 0.118, sign flips by policy and by dataset |
| **G3** | Does the propensity-corrected estimator beat the naive monitor at reachable ESS, on ≥2 of 3 datasets? | **failed** — 1 of 3, and significantly worse on 1 |
| **G4** | Does the controller beat the naive-monitor arm, the randomize-only arm **and** the two online-calibration arms, on ≥2 co-primary outcomes on ≥2 of 3 datasets? | **failed** — 1 of 48 Holm-corrected cells |

G4 is harder than the gate v1 proposed: the audit made arms 8 and 9 mandatory,
so beating the naive monitor and the randomize-only arm is no longer sufficient.

Passing G2 and failing G3 and G4 is an anticipated and acceptable outcome, and
is the one that occurred: a measurement result plus an honest negative result
about the control loop built on it. The manuscript was written to that shape.

## 9. Compute and cost

The full grid — as executed, 13 arms x 3 evaluation datasets x 30 seeds x 150
steps, 1170 runs and 175 500 logged steps, plus the tuning sweep and a post-hoc
sensitivity re-run of 540 further runs — exceeds the author's local machine
(4 cores, 8 GB). Primary runner is **GitHub Actions on
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
3. **An uncited reference, now resolved.** v1 cited a "2025 Computational
   Materials Science study on data and model uncertainty in active learning"
   that did not surface in two independent searches. The Step 2 audit resolved
   it: Li, Jiang, Ni, Li, Huang, Zhao, Liu and Ye, *A study of the role of data
   and model uncertainty in active learning*, Computational Materials Science
   247:113512, doi 10.1016/j.commatsci.2024.113512 (online 2024, in the January
   2025 issue). The year in v1 was wrong, the paper is real, and it is cited in
   Section 6.
4. **Scope.** The central question is narrowed to Q1 and Q2 in Section 3.
5. **Dropped.** The eight-week timeline and the fixed Computational Materials
   Science target are no longer part of the plan.
6. **CA-SMART struck** (session M01). The audit could not find it; an
   unverifiable baseline cannot be implemented or cited. See Section 6.
7. **Arm list corrected** from eleven to the pre-registered thirteen, and Gate
   G4 restated in the hardened form the audit forced (Sections 7.1 and 8).


## 12. Where this went

Every step below the line is complete; this section is kept so the plan can be
read against what happened.

1. **Environment and data access** — `data_access_report.md`. All five Matbench
   sets and seven raw matminer sets download without credentials; OQMD and the
   Materials Project API were dropped (403 and key-gated respectively).
2. **Novelty audit, Gate G1** — `novelty_audit.md`, Section 6 above.
3. **Pre-registration** — frozen at tag `prereg-v1`, commit `79d819a`, archived
   at doi 10.5281/zenodo.23287243.
4. **Implementation** — `src/pcal/`, seven validity tests, three CI workflows.
5. **Gates G2, G3, G4** — 1170 runs on GitHub Actions; verdicts in Section 8.
6. **Results and figures** — `results_summary.md`, `results/figures/`,
   regenerated end to end by `python scripts/make_figures.py`.
7. **Manuscript** — drafted; submission is the remaining step.

The repository README carries the short version of the outcome.
