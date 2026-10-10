# Current session kickoff prompt

**This file supersedes every earlier kickoff prompt. Ignore all previous
versions.** Paste the block below into a fresh Claude Science session in the
`propensity-calibration-al` project.

Last updated: 2026-10-10, after commit `e48221803d843382c6e5a74fa9bedf70b9e7de5b`.

---

## BEFORE YOU START THE SESSION

Open the host-access panel in the **Claude Science left sidebar** (Customize →
host access) and confirm it lists **zero folder grants**. Two sessions have now
been destroyed by stale grants (`C:\Users\Azad\.claude-science.old`,
`C:\Users\Azad\Desktop\cs-recovery`): arming them fails after ~150s with
`launcher exit 3`, which disables every code cell, `edit_file` and
`save_artifacts` for the whole session. Grants are fixed at session start and
cannot be cleared from inside a running session. This project needs no host
folders at all.

---

## PROMPT

```
Continue the reliability-aware active learning project. Read project memory
first -- the direction, constraints and repo decisions are all recorded there.

Reference artifacts in this project:
  plan              0d873129-ed0d-4ebf-a1bc-ba0baf4b0b53
  original proposal dc285bc1-e6b4-479c-99d8-9ad0760cd58d
  prior evaluation  62718083-3469-41e8-b647-98a67f3926dd
  synthetic pilot   0fc82f76-9a2d-470e-b2e0-6b927a7c1f87

Do not request or accept any host folder access.

Repos (GitHub credential is stored; token verified with Contents: write on both):
  https://github.com/kianfarahmand/propensity-calibration-al        public, code + CI
  https://github.com/kianfarahmand/propensity-calibration-al-paper  private, manuscript

Operating rule: GitHub is the only copy that counts. Create and push every file
in the same action -- never batch commits to the end of a step. Recovery from a
dead session must be git clone and nothing else.

ALREADY DONE (public repo, main @ e48221803d843382c6e5a74fa9bedf70b9e7de5b):
  README.md, .gitignore, pyproject.toml, environment.yml (floor pins, to be
  frozen in Step 1), results/raw/.gitkeep, docs/proposal_v2.md (the revised
  proposal, retitled and with all four v1 corrections applied).

START HERE -- Step 1: request network access for ndownloader.figshare.com,
figshare.com, matbench.materialsproject.org, api.materialsproject.org and
oqmd.org; build the conda env (scikit-learn, matminer, pymatgen, statsmodels,
pytest); confirm which materials datasets actually download; replace
environment.yml with a full export of the solved environment; commit
docs/data_access_report.md.

Then Step 2, the OpenAlex novelty audit, before any benchmark code. Gate G1
decides whether the framing survives. Deliverables: docs/novelty_audit.md,
docs/comparison_matrix.csv, and the corpus timeline figure.

Then save the Step 3 pre-registration as preregistration/PREREGISTRATION.md,
commit it, and tag that commit prereg-v1 before any evaluation run.

Everything free of charge, including publication.
```

---

## Session notes worth carrying forward

- The **GitHub REST API is reachable from the control-plane kernel** even when
  the analysis sandbox is dead. If folder grants break a session again, files can
  still be authored and pushed through the Git Data API (blobs → tree → commit →
  ref) using the stored GitHub token. This is how the skeleton above was
  committed.
- **api.openalex.org is NOT reachable from the control-plane kernel** (proxy 403,
  not on that allowlist). The novelty audit requires a working analysis sandbox.
- Dataset hosts (figshare, matbench, materialsproject, oqmd) are not allowlisted
  by default and need an explicit network-access request at the start of Step 1.
