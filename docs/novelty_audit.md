# Novelty audit (Gate G1)

Run 2026-10-10 against OpenAlex. Everything below is measured; the queries, the
screening rule, the grades and the full screened corpus are committed alongside
this file so the audit can be re-run or disputed: `docs/comparison_matrix.csv`
(the sixteen works the verdict is argued against), `docs/novelty_corpus.csv`
(the 1078 records that survived the first screen, with their P/C/L grades) and
`docs/novelty_corpus_full.csv.gz` (all 2357 deduplicated records including the
ones the first screen rejected).

## 0. What is being audited

The contribution under test, written as three separable components:

- **P** - correcting selection bias with importance / inverse-propensity weights,
  using selection probabilities that are known because the acquisition was
  deliberately randomized.
- **C** - the quantity being estimated is the **calibration** of the surrogate's
  predictive distribution (coverage of its predictive intervals), not its
  accuracy, risk, or acquisition value.
- **L** - the setting is a sequential pool-based campaign, the estimand is
  **pool-marginal** (over the entire unlabeled pool, not over the queried
  points), and the estimate is produced *during* the campaign and fed back into
  the acquisition policy.

G1 asks a single question: does any existing work do P and C and L together?

## 1. Corpus construction

Queries were run as `title_and_abstract.search` filters rather than OpenAlex's
default `search`, which also matches full text and inflated every count by one
to two orders of magnitude (e.g. `"active learning" AND "propensity"`: 1032
under `search`, 58 restricted to title and abstract). Phrase semantics were
verified first: `"importance weighted active learning"` returns 6 hits forward
and 0 reversed, so quoting is doing phrase matching and not loose AND.

| stratum | query | records |
|---|---|---|
| `B11_sampling_bias_al` | `"active learning" AND "sampling bias"` | 68 |
| `B12_al_matdisc_unc` | `"active learning" AND "materials discovery" AND "uncertainty"` | 64 |
| `B13_stoch_acq` | `"stochastic acquisition function" OR "randomized acquisition"` | 145 |
| `B14_boltz_acq` | `("boltzmann" OR "softmax") AND "acquisition function"` | 7 |
| `B15_ess_iw_al` | `"importance weights" AND "active learning"` | 8 |
| `B16_cal_covshift` | `("calibration" OR "calibrated") AND "covariate shift" AND "importance weight"` | 3 |
| `B17_unbiased_estimator_al` | `"unbiased estimator" AND "active learning" AND "pool"` | 1 |
| `B18_cal_drift_bo` | `"model drift" AND ("bayesian optimization" OR "active learning")` | 10 |
| `B1_iwal` | `"importance weighted active learning"` | 6 |
| `B2_al_propensity` | `"active learning" AND "propensity"` | 58 |
| `B3_al_ipw` | `"active learning" AND "inverse propensity"` | 2 |
| `B4_bo_selbias` | `("bayesian optimization" OR "bayesian optimisation") AND "selection bias"` | 5 |
| `B5_al_selbias` | `"active learning" AND "selection bias"` | 37 |
| `B6_al_calibration` | `"active learning" AND ("calibration" OR "calibrated")` | 693 |
| `B7_bo_calibration` | `("bayesian optimization" OR "bayesian optimisation") AND ("calibration" OR "recalibration")` | 791 |
| `B8_conf_covshift` | `"conformal prediction" AND "covariate shift"` | 99 |
| `B9_conf_al` | `"conformal prediction" AND "active learning"` | 47 |
| `C1_al_unbiased` | `"active learning" AND ("unbiased" OR "bias correction" OR "debiasing" OR "statistical bias")` | 165 |
| `C2_al_impsamp` | `"active learning" AND "importance sampling"` | 150 |
| `C3_bo_impsamp` | `("bayesian optimization" OR "bayesian optimisation") AND "importance sampling"` | 30 |
| `C4_poolbased` | `"pool-based active learning" OR "pool based active learning"` | 211 |
| `C5_cov_predunc` | `"coverage" AND "predictive uncertainty" AND ("active learning" OR "bayesian optimization")` | 20 |
| `C6_adaptive_bias` | `"adaptive sampling" AND "bias" AND "uncertainty"` | 46 |
| `C7_acq_eps` | `"acquisition function" AND ("epsilon-greedy" OR "randomization" OR "exploration bonus")` | 321 |
| `C8_al_riskest` | `"active learning" AND ("risk estimate" OR "generalization error estimate" OR "test error estimate")` | 8 |

Union: **2859** records, **2357** after `dedup_scholarly`
(research document types only, duplicate titles collapsed, published version
preferred) - a 17.6% duplicate rate,
mostly arXiv/journal and figshare/journal pairs of the same paper.

### Recall validation

Eleven landmarks named in advance were resolved by title and checked for
membership. Five were present after the first (B) strata. The miss that
mattered was Farquhar, Gal and Rainforth's *On Statistical Bias In Active
Learning*, which is squarely on axis P; adding the C strata
(`C1_al_unbiased`, `C2_al_impsamp`, `C8_al_riskest`) recovered it. The six
still absent are landmarks of adjacent fields the inclusion rule does not cover
(calibrated regression without active learning, deep Bayesian active learning
without calibration, doubly-robust off-policy evaluation in RL, materials UQ
benchmarking); they were added to the corpus by hand so the screen would judge
them rather than silently omit them.

One landmark could not be confirmed to exist at all: a title of the form
*"Calibrated Uncertainty Estimation Improves Bayesian Optimization"* returns no
OpenAlex record. It is not cited anywhere in this project. The nearest real work
is Deshpande and Kuleshov's 2021 *Online Calibrated and Conformal Prediction
Improves Bayesian Optimization*.

## 2. Screening

Two passes, both against written rules applied identically to every stratum.

**Pass 1, recall-oriented binary screen** of all 2859 records against a
five-clause inclusion rule, with explicit exclusions for the two dominant
contaminants: "active learning" in the classroom-pedagogy sense, and
"calibration" meaning instrument, sensor or simulator-parameter calibration.
Ambiguous records were included by instruction. Result: **1313 INCLUDE**,
no unparsed batches.

**Pass 2, graded proximity** of the 1313 included records on the P/C/L
decomposition, scored 0-3 with the letters present, using the reasoning-class
model. After deduplication:

| grade | meaning | records |
|---|---|---|
| 3 | the work itself combines P, C and L | 5 |
| 2 | exactly two of P, C, L | 197 |
| 1 | one of P, C, L | 455 |
| 0 | none | 421 |
| - | excluded at pass 1, not graded | 1279 |

### Measured precision

A random sample of **60 of the 202** records at grade >= 2 was read by
hand. **48 were genuinely on at least two of the three axes; 12 were not** -
measured precision **0.80 (48/60)**. The twelve errors, which are what a reader
should expect to find if they re-run this:

- Spatiotemporal Lattice-Constrained Event Linking ... Accident Reports (NLP event linking, no calibration)
- Improving Robustness and Efficiency in Active Learning with Contrastive Loss (AL method, no calibration claim)
- KODEX - KSEEK: ACTIVE (unparseable record)
- Active Learning from Skewed Data ... Weighted Extreme Learning Machine (class-imbalance weighting, not selection-bias correction)
- SpeedCP: Fast Kernel-based Conditional Conformal Prediction (localisation, not shift correction)
- Distributed active learning with application to battery health management (no weighting)
- Ask-n-Learn: Active Learning via Reliable Gradient Representations (no calibration)
- DASH: Decoupled Adaptive Surrogate-Acquisition Harness for Automated BO (no calibration claim)
- Machine Learning-Assisted Bio-Interfacial Engineering ... Nanocomposites (application paper)
- Adapting Nonstationary Multi-output Gaussian Processes to Bayesian Optimization (modelling, no calibration)
- Active Learning Guided Rule and Transformer Fusion for Quality Assurance in Code Security (application paper)
- Inductive Conformal Predictor for Sparse Coding Classifiers (conformal, but no active loop)

Every error is a false positive of the same kind: the grader accepted a loose
reading of "calibration" (confidence-based selection, robustness, localisation)
or of "importance weighting" (class-imbalance weights, kernel weights). None of
the twelve is a near-miss on all three axes, so the G1 verdict below does not
turn on them. The grade >= 2 band is also visibly dominated by one coherent
literature the grader classifies as P+L - adaptive-Kriging reliability analysis,
where importance sampling and active learning are combined to estimate small
failure probabilities. That work is genuinely P+L; its estimand is a failure
probability, not a calibration functional, so it sits outside the claim and is
not a threat.

## 3. The verdict

**The framing survives, but it must be narrowed and three of its claims
surrendered outright.**

Nothing in the corpus does P, C and L together as defined above. But the
distance is smaller than the original proposal assumed, and two works contest
parts of the claim seriously enough that they become mandatory comparators
rather than related work.

### Surrendered

1. **Propensity correction of an active learner's self-inflicted bias is not
   novel and must never be presented as such.** Beygelzimer et al. (2009) and
   Ganti and Gray (2011) own it; Farquhar et al. (2021) refined when it helps.
   This was already the project's position and the audit confirms it.
2. **Recognising that a design algorithm induces its own covariate shift, and
   correcting it by weighting, is not novel.** Fannjiang et al. (PNAS 2022)
   named this "feedback covariate shift" and solved it for prediction-set
   coverage in biomolecular design.
3. **Closing a loop with a shift-corrected uncertainty object is not novel.**
   Stanton et al. (2022) feed conformal prediction sets corrected for covariate
   shift directly into the acquisition.

### What is left, stated precisely

The estimand. Every work above either estimates a quantity *at the queried or
designed points* (Fannjiang: coverage at the designs; Stanton: prediction sets
at candidate queries; Deshpande and Kuleshov: calibration along the realised
query stream) or estimates a *risk* rather than a calibration functional
(IWAL, UPAL, Farquhar). No work estimates **the calibration of the surrogate
over the whole unlabeled pool, from the biased sample the campaign chose, while
the campaign is running** - and then acts on that number.

That is the paper: a pool-marginal calibration functional, a self-normalized
IPW estimator for it with logged propensities and a reported effective sample
size, the leave-one-out refit that separates model drift from selection bias,
and the demonstration that the naive monitor's bias **flips sign with the
acquisition policy** so that a controller which switches policy corrupts its own
input signal.

### The comparator that can sink this

**Deshpande and Kuleshov (2021), and Robust BO via Localized Online Conformal
Prediction (2025).** They attack the same problem - uncertainty estimates
degraded by action-dependent data - with online learning instead of
propensities, and they provably maintain calibration on non-i.i.d. data. The
difference is the target: their guarantee is on the sequence of points the
campaign actually queried; this project's estimand is the pool. If the
pool-marginal reading cannot be shown to differ from, and to beat, an online-
calibration reading in downstream campaign performance, then the existing method
already suffices and the contribution collapses to a measurement note.

**This changes the experimental design**, and the change is binding on the
pre-registration: the arm list must gain an online-calibration arm and a
weighted-conformal arm alongside the Boltzmann-EI-no-monitor baseline. G4 is now
"beats the naive monitor, the randomize-only arm, **and** online calibration",
which is a harder gate than the one originally written.

## 4. Corrections owed to the proposal

The proposal's citation to a "2025 Computational Materials Science study on data
and model uncertainty in active learning" resolves to a real paper with the wrong
year: Li, Jiang, Ni, Li, Huang and Zhao, *A study of the role of data and model
uncertainty in active learning*, Computational Materials Science, **2024**,
doi:10.1016/j.commatsci.2024.113512. Correct the year; do not delete the
citation.

Two materials-side works found here were absent from the proposal and belong in
it, because both already show that calibration quality changes active-learning
outcomes in this domain - which is the premise the project builds on:
*Calibration of uncertainty in the active learning of machine learning force
fields* (MLST 2023) and *Uncertainty estimation affects predictor selection and
its calibration improves materials optimization* (2026, three alloy datasets).
Both measure calibration on the selected sample without raising the selection-
bias question, which makes them the naive-monitor arm this project benchmarks
against rather than competitors.

## 5. Limitations

OpenAlex only. No Semantic Scholar or Google Scholar cross-check, no citation
snowballing beyond the landmark list, and abstracts were missing for
561 of 2859 records
(title-only screening for those). Screening is model-applied at 0.80 measured
precision on the decisive band, not hand-applied to all 2357 records.
A work that frames the same contribution in unusual vocabulary and sits outside
all 25 strata would be missed; the landmark recall check is the only
evidence against that, and it failed once before the C strata were added.
