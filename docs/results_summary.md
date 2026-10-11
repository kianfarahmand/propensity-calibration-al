# Results summary

Everything the project measured, in the order
[`preregistration/PREREGISTRATION.md`](../preregistration/PREREGISTRATION.md)
lists it. Every number here comes from a committed table, every table from a
committed script, and `python scripts/make_figures.py` regenerates the whole
set from a clean checkout. This file is what the manuscript is written from: a
claim absent here is a claim the manuscript does not make.

Generated in session S08, 2026-10-11.

---

## 1. The analysis set

`results/processed/run_inventory.csv`

1170 runs = 13 arms x 3 datasets x 30 seeds, 150 steps each, 175 500 logged
steps, 4.66 runner-hours on GitHub Actions. One value of the tuning constant,
`acquisition.c = 1.0`, fixed by S05 on the tuning pair before any evaluation run
and never revisited. Every arm is seeded from `(dataset, seed)` alone, so the
arms measured in S05 (1-4), S06 (6, 7, 12, 13) and S07 (5, 8, 9, 10, 11) are one
grid read from three dispatches rather than three separate experiments.

| dataset | pool after construction | target |
|---|---|---|
| `elastic_tensor_2015` | 1077 | K_VRH |
| `heusler_magnetic` | 868 | e_form |
| `matbench_expt_gap` | 2990 | experimental band gap |

A further 540 runs (arms 6 and 7 only) were dispatched for the post-hoc
sensitivity analysis in section 6; they are not part of the pre-registered
analysis set and no gate reads them.

---

## 2. Co-primary outcomes (optimization)

Section 11, in order. All comparisons paired on `(dataset, seed)`, 95% BCa
bootstrap over runs with 10000 resamples, Holm correction across the three
co-primary outcomes within each (comparison, dataset).

**Tables.** `results/processed/g4_primary.csv` (arm 7 against each of arms 5, 6,
8, 9), `results/processed/g4_randomization_cost.csv` (H4: arm 5 against arm 2),
`results/processed/f4_randomization_chain.csv` (the arm 2 -> 5 -> 7 chain),
`results/processed/descriptive_arms.csv` (every arm, descriptive).

**Figures.** `results/figures/g4_discovery_curves.png`,
`results/figures/g4_paired_effects.png`,
`results/figures/f4_randomization_chain.png`.

### 2.1 Simple regret at step 150

Degenerate on two of three datasets: every expected-improvement arm finds the
pool maximum inside 150 steps on `elastic_tensor_2015` and `heusler_magnetic`,
so simple regret is identically zero and the outcome separates no arms there. A
tie at zero scores p = 1 and is never read as a win (deviations entry 10). Only
`matbench_expt_gap` carries information, and nothing separates on it.

### 2.2 Top-k enrichment, k = 10 and k = 50

The outcome that moves. Enters Holm once, at its less favourable k, and counts
for the focal arm only if both k agree in sign (deviations entry 9).

The largest effects in the whole study are not the method's; they are the price
of making it possible. Arm 5 (Boltzmann EI, no monitor) minus arm 2 (greedy EI),
top-50 enrichment:

| dataset | difference | 95% BCa | Holm p |
|---|---|---|---|
| `elastic_tensor_2015` | **-0.090** | [-0.115, -0.069] | 0.005 |
| `heusler_magnetic` | **-0.021** | [-0.033, -0.012] | 1.000 |
| `matbench_expt_gap` | -0.025 | [-0.093, +0.037] | 1.000 |

### 2.3 Normalized discovery-curve AUC

Arm 5 minus arm 2: -0.0118 [-0.0277, -0.0020] on `elastic_tensor_2015`
(Holm p 0.026) and -0.0071 [-0.0145, -0.0025] on `heusler_magnetic`
(Holm p 0.001). Nothing separates on `matbench_expt_gap`.

### 2.4 The chain, which is the paper's argument (Figure 4)

Three paired differences on the same runs: what randomization costs (5 - 2),
what the corrected monitor buys back (7 - 5), and the net against the campaign a
practitioner would actually run (7 - 2). Asterisk = 95% BCa interval excluding
zero.

| dataset | outcome | randomize (5-2) | monitor (7-5) | net (7-2) |
|---|---|---|---|---|
| `elastic_tensor_2015` | top-50 | -0.0900* | +0.0160 | **-0.0740*** |
| `elastic_tensor_2015` | top-10 | -0.0233* | +0.0200 | -0.0033 |
| `elastic_tensor_2015` | AUC | -0.0118* | -0.0004 | **-0.0122*** |
| `heusler_magnetic` | top-50 | -0.0213* | -0.0040 | **-0.0253*** |
| `heusler_magnetic` | top-10 | -0.0033 | -0.0100 | -0.0133 |
| `heusler_magnetic` | AUC | -0.0071* | **+0.0075*** | +0.0003 |
| `matbench_expt_gap` | top-50 | -0.0247 | -0.0160 | -0.0407 |
| `matbench_expt_gap` | top-10 | -0.0967 | -0.0800 | **-0.1767*** |
| `matbench_expt_gap` | AUC | -0.0045 | +0.0205 | +0.0160 |

Read it as one sentence: the only link that separates from zero in the method's
favour anywhere is discovery AUC on `heusler_magnetic`, +0.0075 [+0.0018,
+0.0151], and on that dataset and metric it buys back exactly what randomization
cost - the net against greedy EI is +0.0003 [-0.0047, +0.0064], a wash. Every
other separating link is a cost.

---

## 3. Primary measurement outcome (H2, G3)

Mean absolute deviation of the monitor's reading from the evaluator's
`C_t(0.90)` over non-suppressed steps in 50-150.
Tables `results/processed/g3_estimator_error.csv` and
`results/processed/g3_suppression.csv`; figures
`results/figures/g3_estimator_vs_truth.png` and `results/figures/g3_ess.png`.

Arm 7 (propensity-corrected) minus arm 6 (naive), own support, paired over 30
seeds. Negative favours the correction:

| dataset | arm 7 | arm 6 | difference | 95% BCa |
|---|---|---|---|---|
| `elastic_tensor_2015` | 0.0668 | 0.0584 | **+0.0084** | [+0.0010, +0.0170] |
| `heusler_magnetic` | 0.0528 | 0.0517 | +0.0011 | [-0.0058, +0.0088] |
| `matbench_expt_gap` | 0.0445 | 0.0723 | **-0.0278** | [-0.0418, -0.0082] |

One dataset in the correction's favour, one tie, one in which the correction is
significantly **worse**. Re-scoring on the common support - the steps where both
arms produced an eligible reading - changes nothing.

**Availability, which is half the story.** The naive monitor answers at every
step by construction. The corrected one does not:

| dataset | suppression rate, arm 7 | median ESS at step 100 (floor 10) |
|---|---|---|
| `elastic_tensor_2015` | 0.400 | 6.95 |
| `heusler_magnetic` | 0.572 | 3.00 |
| `matbench_expt_gap` | 0.614 | 7.65 |

Three of thirty runs on `matbench_expt_gap` never produce an eligible reading at
all and are dropped from the pairing rather than imputed (deviations entry 8).

**Ablations** (arms 12 and 13, same table): the leave-one-out refit and the
weight clipping each help only on `matbench_expt_gap`. On `elastic_tensor_2015`
the full method is worse than both of its own ablations. Clipping buys
availability, not accuracy: unclipped weights fall below the floor on 88 per
cent of steps on `matbench_expt_gap` and are no worse where they do answer
(common-support difference +0.0047 [-0.0179, +0.0281], 10 surviving pairs).

---

## 4. Secondary outcomes

ESS trajectories and suppression rates are in
`results/processed/g3_suppression.csv` and `results/figures/g3_ess.png`; `lambda`
trajectories, per-step wall-clock and the policy-switch count are columns of the
committed per-step tables. Coverage at `alpha in {0.50, 0.80, 0.95}` is recorded
per step by the evaluator. Not corrected for multiplicity, reported as
descriptive, and not read as evidence for or against any gate.

---

## 5. The gates

| Gate | Rule (Section 13) | Verdict |
|---|---|---|
| **G1** | Does the framing survive the novelty audit? | **Passed, narrowed** |
| **G2** | On >=2 of 3 datasets the mean selected-minus-pool coverage gap over steps 50-150 has a 95% CI excluding 0 for at least one greedy policy, **and** the point estimates have opposite signs for >=2 of the four policies | **Passed** |
| **G3** | Arm 7's mean absolute deviation from ground-truth `C_t(0.90)` is lower than arm 6's, 95% CI on the paired difference excluding 0, on >=2 of 3 datasets | **Failed** (1 of 3) |
| **G4** | Arm 7 beats arms 5, 6, 8 **and** 9 on >=2 of the 3 co-primary outcomes after Holm, on >=2 of 3 datasets | **Failed** (0 of 3) |

### G2 - passed (`results/processed/g2_coverage_gap.csv`, Figure 1)

Selected-minus-pool coverage at the nominal 0.90 level, steps 50-150, BCa over
30 seeds:

| dataset | uniform random | greedy EI | greedy UCB | greedy max-variance |
|---|---|---|---|---|
| `elastic_tensor_2015` | +0.009 | **-0.047*** | **-0.031*** | **+0.012*** |
| `heusler_magnetic` | -0.003 | **+0.050*** | +0.002 | +0.009 |
| `matbench_expt_gap` | -0.005 | **-0.080*** | **-0.118*** | -0.010 |

Both halves of the rule are met and then some. The gap is large where it is
present, uniform random covers zero on all three datasets as it must, and the
sign flips twice over: between policies on the same dataset
(`elastic_tensor_2015`: EI -0.047 against max-variance +0.012) and for the same
policy between datasets (greedy EI: -0.047 on `elastic_tensor_2015`, +0.050 on
`heusler_magnetic`). **This is the project's positive result.** A diagnostic
whose bias changes sign with the policy and the dataset cannot be corrected by
any fixed adjustment, which is what makes an uncorrected controller unsound
rather than merely imprecise.

### G3 - failed

The rule needs 2 of 3 datasets and gets 1. Stated plainly: **the
propensity-corrected reading is not a uniformly better control input than the
naive one.** On `elastic_tensor_2015` it is significantly worse. This is a
pre-registered falsifier firing (Section 16, second bullet: "if the
inverse-propensity estimator's variance at reachable effective sample sizes
exceeds the naive estimator's bias, the correction is not worth its cost").

What survives is the dataset-dependent claim: on `matbench_expt_gap`, where the
naive reading is worst (0.0723 absolute deviation against 0.0584 and 0.0517
elsewhere), the correction repays its variance and cuts the deviation by 38 per
cent.

**Why it failed, as far as the data can say.** Randomizing the acquisition is
itself a debiasing step and competes with the propensity correction. Under the
Boltzmann sampler with the 0.05 exploration floor the *naive* reading is already
within 5-7 coverage points of pool truth on all three evaluation datasets,
against the 12-point gap G2 measured for greedy UCB. The correction is
estimating a residual bias that randomization has already mostly removed, at the
cost of a self-normalized inverse-propensity estimator over 30 steps whose ESS
sits at 3-8 against a floor of 10. Any future framing must compare against the
naive reading **under randomization**, never against the greedy gap.

### G4 - failed

The rule needs arm 7 to beat all four references on >=2 of 3 co-primary outcomes
on >=2 of 3 datasets. It beats all four references on **no** outcome on **any**
dataset. Of the 48 Holm-corrected cells in `g4_primary.csv`, exactly one
separates from zero in arm 7's favour: discovery AUC against arm 5 on
`heusler_magnetic`, +0.0075 [+0.0017, +0.0152], Holm p 0.024. Of the 40
non-degenerate cells, 23 favour arm 7 - a coin flip.

Section 16's third bullet names this outcome: "if arms 8 and 9 match or beat arm
7, propensity logging buys nothing over online calibration and the contribution
reduces to the measurement in H1 and H2." That is where the project is.

---

## 6. Post-hoc sensitivity: would a longer window have rescued G3?

**Labelled post-hoc throughout. Decided after seeing the results
(`preregistration/deviations.md` entry 11). It changes no gate verdict.**

`results/processed/sensitivity_window.csv`, `results/figures/sensitivity_window.png`.

The corrected monitor is suppressed on 40-61 per cent of window steps at the
frozen setting, so the fair question is whether G3 failed because the correction
does not repay its variance, or because a 30-step window never gave it a chance.
Arms 6 and 7 were re-run over the same datasets and seeds at window 60, window
100, and ESS floor 3, and analysed by the same `scripts/g3_error.py` that
decided G3.

Arm 7 minus arm 6, own support (negative favours the correction):

| setting | `elastic_tensor_2015` | `heusler_magnetic` | `matbench_expt_gap` | mean suppression | datasets won |
|---|---|---|---|---|---|
| **frozen** (w30, floor 10) | +0.0084* | +0.0011 | -0.0278* | 0.529 | 1 of 3 |
| window 60 | -0.0002 | +0.0030 | -0.0540* | 0.547 | 1 of 3 |
| window 100 | -0.0082 | +0.0206* | -0.0504* | 0.485 | 1 of 3 |
| floor 3 | +0.0151* | +0.0320* | -0.0396* | 0.136 | 1 of 3 |

**No setting rescues G3.** Three things are worth carrying into the manuscript:

1. A longer window helps the correction a great deal where it already helped
   (`matbench_expt_gap`: -0.0278 -> -0.0540) and hurts it where it did not
   (`heusler_magnetic`: +0.0011 -> +0.0206). The dataset dependence is not a
   sample-size artefact.
2. The naive reading improves with a longer window at least as fast as the
   corrected one does - on `heusler_magnetic` arm 6's deviation falls 0.0517 ->
   0.0410 -> 0.0360 from window 30 to 60 to 100 while arm 7's rises at 100. More
   window means older residuals, and the leave-one-out refit cannot fully undo
   the model drift the pre-registration flagged in Section 7.1.
3. Dropping the ESS floor to 3 does what it is supposed to do to availability -
   suppression falls from 0.53 to 0.14 - and makes the estimate worse on two of
   three datasets. Availability and accuracy trade against each other here, and
   the frozen floor was not the binding mistake.

An earlier version of this analysis tried to re-score the committed per-step log
instead of re-running, failed its own reproduction check at 11-17 per cent
agreement, and wrote nothing. The reason is recorded in deviations entry 11: the
windowed residuals are recomputed under the current surrogate with the point held
out at every read, so the log cannot support a longer window.

---

## 7. Arms no gate tests (descriptive)

`results/processed/descriptive_arms.csv` - every arm, every dataset, every
co-primary metric, with the paired difference against arm 7. No multiplicity
correction, every row labelled `descriptive`.

The striking one is top-50 enrichment. The arms that simply exploit do far
better at recovering the pool's true top 50 than anything in the comparison set:

| dataset | greedy UCB (3) | GP-Hedge (10) | greedy EI (2) | arm 7 | the rest |
|---|---|---|---|---|---|
| `elastic_tensor_2015` | 0.989 | 0.986 | 0.690 | 0.616 | 0.157-0.603 |
| `heusler_magnetic` | 0.995 | 0.993 | 0.871 | 0.845 | 0.197-0.852 |
| `matbench_expt_gap` | 0.539 | 0.501 | 0.399 | 0.358 | 0.065-0.374 |

**Correction to a note made in S07.** S07 recorded GP-Hedge as recovering more
of each pool's true top 50 than any single-acquisition arm. It does not: greedy
UCB is higher on all three datasets. The S07 note compared GP-Hedge only against
the arms in the G4 reference set and missed arm 3. The observation that survives
is narrower and still worth reporting - GP-Hedge and greedy UCB both recover
0.99 of the top 50 on two datasets while every arm in the gate comparison sits
near 0.6 and 0.85, and neither is better on simple regret.

---

## 8. Deviations from the pre-registration

Eleven entries, each stating whether it was decided before or after seeing the
affected result. Full text in `preregistration/deviations.md`.

| # | Entry | Session | Decided |
|---|---|---|---|
| 1 | Non-finite Magpie columns are dropped before PCA | S04 | before |
| 2 | Coverage uses the observation sd; acquisition uses the latent sd | S04 | before |
| 3 | EI uses xi = 0 and UCB uses kappa = 2.0 | S04 | before |
| 4 | `tau` has a numerical floor | S04 | before |
| 5 | Bisection runs for a fixed 60 iterations | S04 | before |
| 6 | Arms 8 and 9 are forms, not reimplementations | S04 | before |
| 7 | The ESS eligibility bar is read over the tuning pair pooled | S04 | before |
| 8 | Runs with no eligible reading are dropped from the pairing | S06 | **after** |
| 9 | Top-k enrichment enters Holm once, at its less favourable k | S07 | before |
| 10 | The BCa interval and the p-value come from one bootstrap | S07 | before |
| 11 | A sensitivity analysis over the estimator window and the ESS floor | S08 | **after** |

---

## 9. What the project can claim, and what it cannot

### Can claim

1. **The premise, measured on real materials pools.** A surrogate's interval
   coverage measured at the points an acquisition policy selected differs from
   its coverage over the pool, by up to 12 points, and the sign of that
   difference depends on both the policy and the dataset. G2, 360 runs, four
   policies, three datasets.
2. **The price of the standard remedy, quantified.** Randomizing the acquisition
   so propensities are loggable costs up to 0.090 of top-50 enrichment and 0.012
   of discovery AUC against the greedy campaign it replaces. H4, pre-registered,
   reported whatever it showed.
3. **An honest negative result about the control loop.** At the pre-registered
   setting the propensity-corrected reading is not a uniformly better control
   input than the naive one (G3: 1 of 3 datasets, significantly worse on one),
   and a controller consuming it does not beat randomization alone, the naive
   monitor, online calibration or localized conformal prediction on any
   co-primary outcome on any dataset (G4: 1 of 48 Holm-corrected cells). A
   post-hoc sweep over the estimator window and the ESS floor does not change
   this.

### Cannot claim

The four things the novelty audit established as prior art, which were never
ours to claim:

1. Inverse-propensity correction of an active learner's self-inflicted sampling
   bias (Beygelzimer et al. 2009; Ganti and Gray 2011; Farquhar et al. 2021).
2. Recognising that a design algorithm induces its own covariate shift and
   correcting it by weighting (Fannjiang et al., PNAS 2022).
3. Closing a control loop on a shift-corrected uncertainty object (Stanton et
   al. 2022).
4. The design-marginal calibration functional and the standardized
   leave-one-out residual construction (Pion and Vazquez, arXiv:2512.05611).

And four more that this session's results rule out:

5. That the corrected reading is a better control input in general. It is better
   on one dataset of three.
6. That propensity logging buys anything over online calibration or localized
   conformal prediction. G4 says it does not.
7. That the randomization the method requires is cheap. It is the largest effect
   in the study.
8. That the failure is an artefact of the frozen window or floor. The post-hoc
   sweep says it is not.

---

## 10. Figures, in manuscript order

| Figure | File | What it shows |
|---|---|---|
| F1 | `results/figures/g2_gap_by_policy.png` | The selection-bias gap and its sign flip. G2. |
| F2a | `results/figures/g3_estimator_vs_truth.png` | Each reading against the evaluator's `C_t(0.90)`. |
| F2b | `results/figures/g3_ess.png` | The ESS floor that gates how often the corrected reading exists. |
| F3a | `results/figures/g4_discovery_curves.png` | What the campaigns actually found. |
| F3b | `results/figures/g4_paired_effects.png` | The paired differences behind G4, with the zero line. |
| F4 | `results/figures/f4_randomization_chain.png` | arm 2 -> 5 -> 7 as one chain of paired differences. The argument. |
| S1 | `results/figures/sensitivity_window.png` | Post-hoc: G3 at four estimator settings. |

Regenerate all of them, and every table above, with:

```
python scripts/make_figures.py
```
