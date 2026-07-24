# ARC-AGI-3 Artifact Study

This subject contains the ARC-AGI-3 program-growth harness, promoted solver
artifacts, experiment logs, and manuscript for the Gödel–Kolmogorov Machine. The
Gödel–Kolmogorov Machine couples verifier-gated self-revision to
description-length selection and retained solver structure. The name
Gödel–Kolmogorov Machine is used in full here before the abbreviation **GKM** is
adopted below. The scientific object is the retained sequence of code states and
validated replays; the official replay score is reported separately.

## Verified Endpoints

<!-- BEGIN GENERATED: ARC_ARTIFACT_STATUS -->
| Game | Verified levels | Replay actions | Published ledger charge |
|---|---:|---:|---:|
| `wa30` | 9/9 | 596 | 1458 |
| `ls20` | 7/7 | 393 | 362 |

Both published ledgers contain one entry for every replay-validated level. The operational checkpoint may retain only records accumulated after its resume base; the manuscript sidecar supplies the complete audited history. `marginal_C` means positive net retained-description growth per source file. Additions and deletions within the same file are netted before the positive part, so same-size replacement can receive zero.
<!-- END GENERATED: ARC_ARTIFACT_STATUS -->

The expanded campaign contains 23 replay-valid endpoints. Beyond the two complete
games, `ft09`, `r11l`, and `tr87` reach L6; `g50t` reaches L5; `ar25`, `re86`,
and `sp80` reach L4; `cd82` and `m0r0` reach L2; and 12 games reach L1. `bp35`
and `tn36` have no promoted level. Only `wa30` and `ls20` have complete
manuscript sidecar histories.

The published [Competition-Mode scorecard](https://arcprize.org/scorecards/9e166671-0953-42f3-89de-a0fd57d7b147)
scores **17.136507936507936%** over all 25 public games. The distinct unweighted
coverage represented by that card is **37/183 = 20.2186%**. The card's stored paths
contained 1448 actions and used 1456 API actions after eight resets. The subsequently
extended local artifacts contain **67/183 = 36.6120%** raw level coverage and 2148
stored replay actions. Those 30 additional clears have independent local replay
certificates but are not silently attributed to the earlier public scorecard.

The latest 49-turn GPT-5.6-sol campaign does not establish that high reasoning is
intrinsically cheaper: its continuation turns were selected after medium failures.
Cold L1 acquisition cost 4 displayed allowance points for 3 medium clears and 18 for
12 high clears; L2+ continuation cost 39 points for 7 medium clears and 14 for one
high clear. Medium produced all four new direct literal-reuse wins. The next-window
policy therefore starts each fresh continuation on medium and admits at most one
bounded high rescue after a clean failure. See
[`crack_lab/CHEAP_CAMPAIGN.md`](crack_lab/CHEAP_CAMPAIGN.md).
Retrospectively, high rescued one of six medium-failed targets at a total charge of
12 displayed points, which supports using it as a fallback rather than a default.

The action totals describe the final replay paths. Exploration used the local
`Arena.clone()` oracle and was not metered, so these values do not measure official
ARC-AGI-3 interaction or sample efficiency. The official wrapper provides `reset()`
and `step()` but no arbitrary state-fork operation.

The historical field `marginal_C` is also narrower than its name suggests. It is the
sum of positive net description-size changes in `legs.py` and `players.py`, including
a surcharge for container-literal elements. Same-size replacement can receive zero.
Use source provenance and replay, not the scalar alone, to assess reuse.

## Solved-Checkpoint Comparator Audit

The cross-system audit uses only retained states that actually cleared a level.
Interim synthesis revisions, repeated same-level commits, and notebook edits are
excluded.

| System | Exact boundary object | Conditional AST marginal | Direct literal reuse |
|---|---|---|---|
| GKM | 63 winning sources across 67 clears; 39 adjacent transitions | 21/39 comparable marginals decrease; 6 fall by at least half | 14 winning players directly call unchanged leg literals; `ar25` L2, `g50t` L4, `ls20` L7, and `m0r0` L2 couple this reuse to sharp drops |
| OPINE-World | 146 pre-solve engines for 153 trace solves; 121 adjacent transitions | 49/115 decrease; 14 fall by at least half | 4 synthesized-planner wins directly call unchanged engine literals; `lp85` L4 and `tu93` L3 couple this reuse to sharp drops |
| baseline1 GPT-5.5 xHigh | 160 retained snapshots for 174 clears; 50 exact winning sources and 18 adjacent transitions | 5/8 comparable marginals decrease; none falls by half | 0: every exact adjacent winning command is a fresh literal action program |
| Retrodict | 170 solved memory checkpoints | No executable marginal is released | 0 executable witnesses; the released object is curated memory |

Under the exact winning-entry-point test, OPINE has hard level-to-level
executable reuse; baseline1 does not, and Retrodict lacks the executable
checkpoint needed to test it. GKM has the strongest literal-leg evidence in
the measured exact set.

The literal-call test changes the earlier cumulative-size interpretation.
baseline1 still has four exact retained authored-source/AST contractions, but
none is demonstrated solver reuse: the winning commands invoke no retained
world-model definition. OPINE supplies two hard counterexamples to the claim
that it solves every level wholly anew. GKM supplies four sharp-drop/reuse
intersections and ten further direct unchanged-leg wins. Its cumulative
executable source nevertheless grows at every exact adjacent transition.

The machine-readable results and analyzers are under [`audit_results/`](audit_results/).
The coupled test is `audit_marginal_literal_reuse.py`; the system-specific boundary
reconstructions remain in the `audit_*_artifacts.py` and
`audit_*_solved_checkpoints.py` scripts.

## Provenance

The complete `wa30` run was not mechanic-blind. Earlier source inspection and human
play informed carry/relay priors that entered the main development lineage. A later
neutral-prior run independently validates level 1 only; it does not retroactively make
the nine-level lineage blind. The `ls20` run did not receive an `ls20` mechanic name,
but it still used the stronger clone-enabled local harness.

The complete published `wa30` ledger is
`112, 78, 95, 47, 405, 225, 145, 204, 147` for L1--L9. The current checkpoint is an
operational resume file and therefore retains only its post-base records, totaling 1243;
the manuscript sidecar indexes the complete 1458-unit ledger, the clean promotion events,
and their file hashes.

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
- [`manuscript/README.md`](manuscript/README.md): reproducible paper/figure build and
  forward-revision sources.
- [`crack_lab/`](crack_lab/): harness and chronological findings.
- [`crack_lab/agent_solutions/`](crack_lab/agent_solutions/): promoted artifacts.
- [`manuscript/artifact_history/`](manuscript/artifact_history/): compact clean-history
  sidecar; the original dirty WIP remains under each artifact.
- [`../REPRODUCE_ARC.md`](../REPRODUCE_ARC.md): replay procedure.

Run focused tests from the repository root:

```bash
python -m pytest arc/crack_lab/test_gkm_legs.py -q
```
