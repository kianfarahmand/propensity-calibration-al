# Deviations from the pre-registration

`preregistration/PREREGISTRATION.md` is frozen at tag `prereg-v1` and is never
edited. Everything that departs from it is recorded here instead, appended in
chronological order, one entry per departure.

Each entry states:

- **Date** and the commit SHA at which the departure took effect.
- **What** changed, quoting the pre-registered text it replaces.
- **Why**, concretely.
- **When it was decided** — specifically, whether the decision was made *before*
  or *after* seeing any result the change could affect. This is the field a
  reviewer will read first, and an honest "after" is worth more than a
  defensible-sounding "before".
- **What it affects** — which hypotheses, gates, arms or figures.

A departure decided after seeing an affected result does not invalidate the
work, but any outcome downstream of it is reported as exploratory rather than
confirmatory, and the manuscript says so in the same sentence as the result.

---

## S04, 2026-10-10 — implementation choices the frozen plan left open

Commit: `afa8c25d` (and the S04 commits preceding it).

All six entries below were decided **before any evaluation run existed**. No
arm has been executed on an evaluation dataset at the time of writing; the only
campaign that has run is the smoke configuration, on a tuning dataset, at 20
steps, which produces no scientific result and is reported as none. Every one of
these is an ambiguity in the frozen text rather than a change to it: the S04
procedure requires that such ambiguities be logged rather than resolved
silently.

### 1. Non-finite Magpie columns are dropped before PCA

**What.** Section 4.1 step 2 says "Reject the run if any feature row is
all-NaN". It is silent on scattered NaNs. Magpie leaves individual elemental
properties undefined for some elements, so a descriptor *column* can be NaN for
a few compositions while no row is entirely NaN. Feature columns that are not
finite for every pool member are now dropped before standardization, and the
count is recorded in the pool summary of every run record.

**Why.** Standardizing a matrix with scattered NaNs propagates them, and the
pre-registered PCA then fails outright — `SVD did not converge` on
`matbench_steels`. Columns are dropped rather than rows because dropping rows
would silently change the pre-registered pool, and imputing values would invent
descriptors. The operation is unsupervised and label-free, like the rest of
step 6, so it cannot leak.

**Affects.** All datasets, all arms, equally. It cannot favour an arm, because
the pool is constructed once and shared.

### 2. Coverage uses the observation sd; acquisition uses the latent sd

**What.** Section 1 writes `sigma_t(x)` without saying whether it includes the
WhiteKernel noise term. Coverage of `y(x)` is evaluated against
`sqrt(var_f + noise)`; acquisition functions use `sqrt(var_f)`.

**Why.** The coverage event in Section 1 is about an observation, which on an
experimental dataset carries measurement noise; excluding the noise term would
guarantee under-coverage on exactly the datasets chosen for having real noise.
Acquisition is about where the latent function may be high, which is the
conventional form. The evaluator's ground truth and every monitor use the same
convention, so the comparison in H2 is like for like.

### 3. EI uses xi = 0 and UCB uses kappa = 2.0

**What.** Section 6 names the acquisition functions but fixes no exploration
constants for the greedy forms.

**Why.** xi = 0 is classic expected improvement with no free parameter, and
kappa = 2.0 is the most common default. Neither is tuned, and neither is used by
arm 7: arms 2 and 3 exist to exhibit opposite-signed selection bias for H1, not
to be competitive baselines.

### 4. tau has a numerical floor

**What.** Section 6 sets `tau_t = c * MAD_t`. A degenerate score vector gives
`MAD_t = 0` and an undefined temperature. `tau_t = max(c * MAD_t, 1e-12)`.

**Why.** The random arm has a constant acquisition score by construction, and a
converged surrogate can produce one too. The floor makes the softmax of a
constant vector uniform, which is the intended behaviour, rather than a division
by zero.

### 5. Bisection runs for a fixed 60 iterations

**What.** Section 8 specifies bisection on `log lambda` over `[-2, 2]` without a
stopping rule. Sixty halvings of that bracket.

**Why.** It is deterministic, costs nothing next to a GP refit, and converges
far below any precision that could change a decision.

### 6. Arms 8 and 9 are forms, not reimplementations

**What.** Section 9 names arm 8 the "Deshpande and Kuleshov 2021 form" and arm 9
the "LOCBO 2025 form". Arm 8 is implemented as an online stochastic-approximation
update of coverage on the realized query stream; arm 9 as a kernel-weighted
average of windowed coverage indicators over the pool, with the median-heuristic
bandwidth. The learning rate 0.1 and the bandwidth rule were not pre-registered.

**Why.** Both published methods target the calibration of the realized stream
rather than a pool-marginal functional, so neither has an off-the-shelf estimator
of `C_t(alpha)`. The harness needs all monitors to answer the same question for
H2 and H3 to mean anything. The manuscript must describe these as adaptations
and must not claim to have reproduced either paper's method.
