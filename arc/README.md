# ARC-AGI-3 Artifact Study

This subject contains the ARC-AGI-3 program-growth harness, promoted solver
artifacts, experiment logs, and manuscript. The scientific object is the retained
sequence of code states and validated replays, not an official benchmark score.

## Verified Endpoints

<!-- BEGIN GENERATED: ARC_ARTIFACT_STATUS -->
| Game | Verified levels | Replay actions | Published ledger charge |
|---|---:|---:|---:|
| `wa30` | 9/9 | 596 | 1458 |
| `ls20` | 7/7 | 393 | 362 |

Both published ledgers contain one entry for every replay-validated level. The operational checkpoint may retain only records accumulated after its resume base; the manuscript sidecar supplies the complete audited history. `marginal_C` means positive net retained-description growth per source file. Additions and deletions within the same file are netted before the positive part, so same-size replacement can receive zero.
<!-- END GENERATED: ARC_ARTIFACT_STATUS -->

The bounded follow-on campaign currently has additional replay-valid partial
artifacts for `ft09` through L4, `sp80` through L4, `g50t` through L1, and `tr87`
through L3. They are kept in the same canonical artifact layout but are not part of
the manuscript's two complete-history table above.

The action totals describe the final replay paths. Exploration used the local
`Arena.clone()` oracle and was not metered, so these values do not measure official
ARC-AGI-3 interaction or sample efficiency. The official wrapper provides `reset()`
and `step()` but no arbitrary state-fork operation.

The historical field `marginal_C` is also narrower than its name suggests. It is the
sum of positive net description-size changes in `legs.py` and `players.py`, including
a surcharge for container-literal elements. Same-size replacement can receive zero.
Use source provenance and replay, not the scalar alone, to assess reuse.

## Provenance

The complete `wa30` run was not mechanic-blind. Earlier source inspection and human
play informed carry/relay priors that entered the main development lineage. A later
neutral-prior run independently validates level 1 only; it does not retroactively make
the nine-level lineage blind. The `ls20` run did not receive an `ls20` mechanic name,
but it still used the stronger clone-enabled local harness.

The complete published `wa30` ledger is
`112, 78, 95, 47, 405, 225, 145, 204, 147` for L1--L9. The current checkpoint is an
operational resume file and therefore retains only its post-base records; the manuscript
sidecar indexes the complete ledger, the clean promotion events, and their file hashes.

The proposer blocks hidden-source and private-runtime inspection before execution.
Rejected tool inputs are preserved verbatim in `blocked_attempts.log` within WIP but do
not taint a promotion because they never ran. The exception is not retroactive: older
WIP that predates the guard remains dirty evidence unless execution-time blocking is
independently recorded. Canonical promoted files are always checked under the current
taint rules, regardless of their creation date.

Debrief prose may quote a rejected command as Markdown inline code. Such a quotation
in `proposer_last.log` is not treated as execution evidence; executable workspace
files and actual command records remain subject to the private-runtime scan.

The separation is motivated by repeated proposer misconduct. During `ft09`, the
Sonnet API proposer emitted two separate commands that accessed `env._game` and
enumerated the private runtime after frame-only probing stalled. We classify these as
cheating attempts in the operational sense: they sought evidence outside the declared
interface. The run is WIP-only, and its exact transcript is preserved under
`crack_lab/agent_solutions/ft09_legs/wip_context/level_01/interrupted_a9a30e6e4da1/`.
The repetition shows that model instructions are not an audit boundary and that
compliance can deteriorate when a proposer stops making progress.

## Entry Points

- [`ARC.md`](ARC.md): detailed domain guide and experiment history.
- [`manuscript/arc_agi3.tex`](manuscript/arc_agi3.tex): scholarly manuscript.
- [`crack_lab/`](crack_lab/): harness and chronological findings.
- [`crack_lab/agent_solutions/`](crack_lab/agent_solutions/): promoted artifacts.
- [`manuscript/artifact_history/`](manuscript/artifact_history/): compact clean-history
  sidecar; the original dirty WIP remains under each artifact.
- [`../REPRODUCE_ARC.md`](../REPRODUCE_ARC.md): replay procedure.

Run focused tests from the repository root:

```bash
python -m pytest arc/crack_lab/test_gkm_legs.py -q
```
