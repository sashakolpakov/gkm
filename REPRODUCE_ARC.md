# Reproducing the ARC-AGI-3 promoted artifacts

This guide explains how to replay the manuscript's promoted `wa30` and `ls20` artifacts.
It does not rerun the full proposer/discovery process. It replays already
promoted solver artifacts through the ARC-AGI-3 interface.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

An ARC-AGI-3 API key is required for the remote replay. Provide it via the repo
`.env` file (`ARC_API_KEY=...`, see [`.env.example`](.env.example)) or export it
in the environment. The replay script reads `.env` automatically.

## Replay promoted artifacts

`--mode online` is a remote dry run against the same API, with no competition
constraints — run this first, because a desync here costs nothing:

```bash
python arc/crack_lab/replay_scorecard.py --mode online
```

By default this replays both promoted games. To replay a single game:

```bash
python arc/crack_lab/replay_scorecard.py --mode online --games wa30
```

## Generate competition-mode scorecard

`--mode competition` produces the single real scorecard: each environment may be
made once, and the closed scorecard is what the community leaderboard links as
`scorecard_url`.

```bash
python arc/crack_lab/replay_scorecard.py --mode competition
```

## Artifact locations

* `arc/crack_lab/agent_solutions/wa30_legs/`
* `arc/crack_lab/agent_solutions/ls20_legs/`

The same canonical layout also holds the bounded campaign's partial promoted
artifacts: `ft09_legs/`, `sp80_legs/`, `g50t_legs/`, `tr87_legs/`, and
`r11l_legs/`, each through L4. Their `checkpoint.json` files contain the independently
replayed paths; they are not included in the manuscript's complete-history table
below.

Each folder holds the promoted `checkpoint.json` (the replay-validated flat
action path, `reached` level count, and `total_marginal_C`), the shared leg
library, per-level players, and WIP snapshots.

The manuscript's compact clean-history index is under
`arc/manuscript/artifact_history/`. It contains complete published ledgers and
hashed clean core files; the full dirty restart context remains in each artifact's
`wip_context/` tree. Verify the index without changing an artifact:

```bash
python arc/manuscript/build_artifact_history.py --check
```

Dirty WIP is intentionally not solver evidence. In particular,
`ft09_legs/wip_context/level_01/interrupted_a9a30e6e4da1/files/proposer_last.log`
records two separate Sonnet commands accessing `env._game`. These are retained as
operational cheating evidence, not promoted or used to support a result. Promotion
requires a separately clean
workspace to pass the current taint check and fresh replay.

Markdown inline-code quotations of rejected commands in a debrief transcript are
not execution evidence. The scanner still checks executable workspace files and
actual command records; the rejection ledger records the commands that were blocked.

Each tagged scratch workspace admits only one orchestrator process at a time via
an OS-level lock. Checkpoint recording is an upsert by level; loading or saving a
legacy checkpoint with repeated level numbers keeps the last entry and removes the
superseded charge from `total_marginal_C`.

## What should be checked

A reviewer should verify:

* the promoted replay completes the claimed number of levels;
* action counts match the artifact documentation;
* the solver does not require hidden harness labels;
* the replay starts from a fresh environment/session;
* the marginal-complexity ledger matches the promoted artifact trail.

## Reviewer checklist

Useful external checks include:

- replay `wa30` from a clean environment;
- replay `ls20` from a clean environment;
- confirm action counts;
- inspect promoted solver code;
- inspect WIP snapshots and literal-to-leg refactors;
- verify that marginal-complexity accounting matches the promoted trail;
- compare against graph-exploration or executable-world-model baselines;
- test whether the replay path is robust to fresh sessions/seeds where applicable.

## Current claimed promoted artifacts

<!-- BEGIN GENERATED: ARC_ARTIFACT_STATUS -->
| Game | Verified levels | Replay actions | Published ledger charge |
|---|---:|---:|---:|
| `wa30` | 9/9 | 596 | 1458 |
| `ls20` | 7/7 | 393 | 362 |

Both published ledgers contain one entry for every replay-validated level. The operational checkpoint may retain only records accumulated after its resume base; the manuscript sidecar supplies the complete audited history. `marginal_C` means positive net retained-description growth per source file. Additions and deletions within the same file are netted before the positive part, so same-size replacement can receive zero.
<!-- END GENERATED: ARC_ARTIFACT_STATUS -->
