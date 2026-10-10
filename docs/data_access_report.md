# Data access report

Step 1 of the execution plan: establish which materials datasets are actually
reachable, at what cost, from a zero-credential environment — before any
benchmark code is written. Everything below was measured on 2026-10-10, not
quoted from documentation.

Environment: `environment.yml` (conda-forge, exact pins); exact Python
distribution set in `environment.lock.txt`. matminer 0.10.1,
pymatgen 2026.9.24, scikit-learn 1.9.1, numpy 2.4.6,
Python 3.11.17.

## 1. Hosts

Downloads do not come from where the dataset registry says they do. matminer's
registry points at `ndownloader.figshare.com`, which 302-redirects to a
presigned object on `s3-eu-west-1.amazonaws.com`; allowlisting only the figshare
host leaves every figshare-backed dataset failing with a proxy error. The five
Matbench datasets do not touch figshare at all — they are served from
`ml.materialsproject.org`.

| host | role | result |
|---|---|---|
| `ml.materialsproject.org` | serves all 5 Matbench v0.1 `.json.gz` files | 200, all five fetched |
| `ndownloader.figshare.com` | matminer's 30 figshare-hosted datasets | 302 redirect, resolved |
| `s3-eu-west-1.amazonaws.com` | presigned target of that redirect — **the actual bytes** | 200 after allowlisting |
| `figshare.com` | 2 datasets registered under the bare host | 202 |
| `api.figshare.com` | figshare REST metadata | allowlisted; `load_dataset` never calls it |
| `matbench.materialsproject.org` | documentation site | 200, HTML only — not a data endpoint |
| `api.materialsproject.org` | Materials Project REST API | `/heartbeat` 200 (v0.87.3.dev20); data routes need a free registered API key |
| `oqmd.org` | OQMD REST API | 403 on `/` and `/oqmdapi/...` — server-side refusal, not the sandbox proxy |

Consequence for CI: a workflow that fetches data needs egress to
`ml.materialsproject.org`, `ndownloader.figshare.com` **and**
`s3-eu-west-1.amazonaws.com`. GitHub-hosted runners have unrestricted egress, so
this is a constraint on local reproduction only.

## 2. Datasets that download

All twelve candidates fetched successfully with no credentials. Sizes are the
compressed `.json.gz` on disk; fetch times are from this machine and are
network-bound.

| dataset | host | rows | cols | file MB | fetch s | first columns |
|---|---|---|---|---|---|---|
| `matbench_expt_gap` | ml.materialsproject.org | 4604 | 2 | 0.037 | 0.9 | composition, gap expt |
| `matbench_steels` | ml.materialsproject.org | 312 | 2 | 0.009 | 0.9 | composition, yield strength |
| `matbench_dielectric` | ml.materialsproject.org | 4764 | 2 | 3.608 | 20.4 | structure, n |
| `matbench_jdft2d` | ml.materialsproject.org | 636 | 2 | 0.267 | 1.6 | structure, exfoliation_en |
| `matbench_expt_is_metal` | ml.materialsproject.org | 4921 | 2 | 0.035 | 1.4 | composition, is_metal |
| `expt_gap` | ndownloader.figshare.com | 6354 | 2 | 0.051 | 1.8 | formula, gap expt |
| `steel_strength` | ndownloader.figshare.com | 312 | 17 | 0.015 | 1.1 | formula, c, mn, si |
| `dielectric_constant` | ndownloader.figshare.com | 1056 | 16 | 0.885 | 3.4 | material_id, formula, nsites, space_group |
| `jarvis_dft_2d` | ndownloader.figshare.com | 636 | 15 | 0.49 | 2.9 | epsilon_x opt, epsilon_y opt, epsilon_z opt, exfoliation_en |
| `elastic_tensor_2015` | ndownloader.figshare.com | 1181 | 20 | 1.117 | 4.0 | material_id, formula, nsites, space_group |
| `heusler_magnetic` | ndownloader.figshare.com | 1153 | 10 | 0.026 | 1.2 | formula, heusler type, num_electron, struct type |
| `piezoelectric_tensor` | ndownloader.figshare.com | 941 | 13 | 0.751 | 2.6 | material_id, formula, nsites, point_group |

`matbench_expt_gap` (4604 rows, composition and target only) and the raw
`expt_gap` (6354 rows) derive from the same Zhuo et al. source; `matbench_steels`
and `steel_strength` both hold 312 rows, the Matbench copy reduced to composition
and target. The exact row-level relationship between each Matbench set and its
matminer ancestor has not been checked here and is not relied on. The Matbench
variants carry a published train/test protocol and a leaderboard, which is the
reason to prefer them even though the raw matminer sets are larger.

Composition-only pools, usable with Magpie descriptors directly:
`matbench_expt_gap`, `matbench_steels`, `matbench_expt_is_metal`, `expt_gap`,
`steel_strength`, `heusler_magnetic`. Structure-valued pools requiring either
composition extraction or structural descriptors: `matbench_dielectric`,
`matbench_jdft2d`, `dielectric_constant`, `jarvis_dft_2d`,
`elastic_tensor_2015`, `piezoelectric_tensor`.

Measurement noise, which the design requires on at least one evaluation
dataset: `matbench_expt_gap` / `expt_gap` (experimental band gaps) and
`matbench_steels` / `steel_strength` (measured yield strength) are
experimental. Everything else is DFT, where the target is deterministic given
the functional and predictive uncertainty is purely model misspecification.

## 3. Datasets that do not download

**OQMD.** `https://oqmd.org/` and `https://oqmd.org/oqmdapi/formationenergy`
both return HTTP 403 with an HTML body. This is the origin refusing the
request, not the sandbox allowlist (the allowlist failure mode is a tunnel
error at connect time, which is what `api.figshare.com` and
`s3-eu-west-1.amazonaws.com` produced before they were granted). OQMD is
therefore excluded from the dataset plan.

**Materials Project API.** `/heartbeat` answers 200, so the host is reachable,
but every data route requires an API key tied to a registered account. A key is
free, but it is a credential a reviewer cannot be assumed to hold and it would
have to be injected into CI as a secret. Since the MP-derived properties this
project would want are already available credential-free through the matminer
and Matbench copies above, the MP API is not used.

## 4. Featurization cost, and a trap worth recording

Magpie `ElementProperty` descriptors, 132 dimensions, measured here:

| dataset | n | parse s | featurize s | s / composition | matrix MB | NaN rows |
|---|---|---|---|---|---|---|
| `matbench_steels` | 312 | 0.3 | 110.9 | 0.355 | 0.33 | 0 |
| `matbench_expt_gap` | 4604 | 0.3 | 1514.1 | 0.329 | 4.86 | 0 |

The per-composition figure is roughly two orders of magnitude worse than this
featurizer's usual throughput; the sandbox this was measured in reported four
cores fully contended and ~0.9 GB free throughout, and the process accumulated
only 3.5 s of CPU across 1626 s of wall clock, so it was starved rather than
computing. Treat the absolute numbers as a floor on the authoring machine, not
as a property of matminer. The operational conclusion is unchanged and was
already the plan: featurize once, cache the matrix, and run the campaign grid
on CI rather than locally. The feature matrix itself is tiny
(4.86 MB for 4604x132 float64), so the cost
is entirely in building it, and it only ever has to be paid once per dataset.

**The trap.** `ElementProperty.featurize_many` accepts a list of formula
*strings* without complaint when `ignore_errors=True`, and returns a matrix
that is entirely NaN — 4604 of 4604 rows — after spending 1763 s raising and
swallowing one exception per row. Composition objects are mandatory
(`StrToComposition`, or `pymatgen.core.Composition` directly). With proper
Composition inputs both datasets featurize with zero NaN rows. Two guards
follow from this and are owed to the test suite in a later step: assert
`ignore_errors=False` at every featurization call site, and assert the
resulting matrix has no all-NaN row before any pool is built from it. A silent
all-NaN pool would not crash the active-learning loop — it would make every
surrogate prediction identical and quietly invalidate an entire run.

## 5. What this fixes in the plan

- Dataset sources are confirmed zero-cost and credential-free; the no-cost
  constraint survives Step 1 with OQMD and the MP API dropped.
- At least one experimental dataset is available, which the design requires.
- `environment.yml` is now pinned from a solved environment rather than
  carrying floor constraints.
- Nothing here commits to a final dataset selection or to the tuning/evaluation
  split. That belongs in the pre-registration, after the novelty audit.
