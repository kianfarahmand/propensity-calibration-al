# Session kickoff

**This file is superseded as an instruction set.** It used to carry the full
kickoff prompt; it no longer does. The project is now driven from an ordered
set of session files in the private companion repository
`kianfarahmand/propensity-calibration-al-paper`, under `sessions/`.

Ignore every earlier kickoff prompt, in chat or elsewhere. There is one current
entry point and it is the private repository's `sessions/STATE.md`.

## How to continue the project

Open a session in the project and say **`run Sxx`** - the id that
`sessions/STATE.md` lists as runnable. Nothing else. The agent reads
`sessions/state.json`, checks the session's dependencies, reads its file, marks
it `in_progress` and pushes that before doing any work, executes it pushing as
it goes, writes the outcome back, expands the next session, regenerates the
state and reports. The contract is `sessions/PROTOCOL.md`.

## Standing constraints, which no session overrides

- **GitHub is the only copy that counts.** Every file is created and pushed in
  the same action. No batching, no commit at the end of a step. Recovery from a
  dead session is `git clone` and nothing else.
- **Grant no host folders.** This project needs none; a stale grant has twice
  broken a session at startup.
- **Nothing costs money.** No article-processing charges, no paid datasets, no
  paid APIs, no paid compute. Free hosted CI on this public repository is the
  experiment runner.
- **The pre-registration is frozen** at tag `prereg-v1` and is never edited.
  Departures are appended to `preregistration/deviations.md`, each stating
  whether it was decided before or after seeing the affected result.
- **Datasets are never committed.** They are re-fetched and cached inside the
  CI run.

## What is in this repository

Code, configs, CI workflows, the frozen pre-registration, the novelty audit,
the data access report, result tables and figures. The manuscript, the
submission material and the session control plane are in the private
repository.
