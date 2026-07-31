# ARC-AGI-3 Artifact Study

This subject contains the ARC-AGI-3 program-growth harness, promoted solver
artifacts, experiment logs, and manuscript for the Gödel–Kolmogorov Machine. The
Gödel–Kolmogorov Machine couples verifier-gated self-revision to
description-length selection and retained solver structure. The name
Gödel–Kolmogorov Machine is used in full here before the abbreviation **GKM** is
adopted below. The scientific object is the retained sequence of code states and
validated replays; the official replay score is reported separately.

The producer is universal across the benchmark: it retains the same proposer
contract, Arena interface, blank scaffold, complexity coordinate, and replay gate.
Its promoted solver programs are learned separately for each game and level.

## Verified Endpoints

<!-- BEGIN GENERATED: ARC_ARTIFACT_STATUS -->
| Scope | Verified levels | Stored replay actions | Ledger charge |
|---|---:|---:|---:|
| Frozen v2 release (25 games) | 181/183 | 7001 | — |
| `wa30` uniform history | 9/9 | 597 | 318 |
| `ls20` uniform history | 7/7 | 365 | 760 |

The frozen release contains one checkpoint record for every promoted level. The generated all-game marginal table is `arc/manuscript/generated/marginal_complexity_by_level.md`. The `wa30` and `ls20` rows additionally have one uniform, audited history sidecar per level. `marginal_C` means positive net retained-description growth per source file. Additions and deletions within the same file are netted before the positive part, so same-size replacement can receive zero.
<!-- END GENERATED: ARC_ARTIFACT_STATUS -->

The generated table above is the frozen published-manuscript scope, not the live
campaign frontier. Live coverage, milestones, budget semantics, artifact-uniformity
work, scorecard release gates, and the automated straight-line rerun are defined in
the single [campaign master plan](crack_lab/ARC_AGI3_CAMPAIGN_PLAN.md). Machine-readable
checkpoint and comparison outputs live under [`audit_results/`](audit_results/).
The replaceable runtime scheduler snapshot is
[`crack_lab/ARC_AGI3_CAMPAIGN_QUEUE.json`](crack_lab/ARC_AGI3_CAMPAIGN_QUEUE.json);
it is generated from the master policy and is not a second source of truth.
Mutable counts are not duplicated here.

The contiguous scheduler uses exact-frontier clean retry count as one
game-independent operational complexity coordinate. It drives both the
medium/high/xhigh/max effort ladder and, after the first clean max failure, an
independent observation-only sidecar when capacity is otherwise idle. The
sidecar is a separate agent role—not an `ultra` effort label—and has no direct
WIP or promotion authority. The scheduler allocates capacity but authors no
game-specific sidecar brief: dispatch requires an authenticated same-frontier
native-proposer request or an admitted supervisory handoff. At the same
hard-frontier stage, a separately
isolated supervisory proposer may synthesize authenticated native/sidecar
evidence into a Socratically challenged tactical handoff. It cannot choose the
game, effort, allocation, WIP mode, or promotion. At most one is active per
frontier, and another round requires new admitted evidence or a complete
scheduler-defined reset/continuation pair. Every source observation cited by
the handoff must be reproduced by the receiving native proposer through the
public Arena surface; only ordinary candidate replay tests the proposed tactic.
The deterministic scheduler, native proposer, side expert,
supervisory proposer, and host verifier have distinct receipt-bound roles.
Their policy projections and adversarial tests are reproduced by the commands
in [`../REPRODUCE_ARC.md`](../REPRODUCE_ARC.md).

The frozen v2 [Competition-Mode scorecard](https://arcprize.org/scorecards/cf75e14b-2c25-41cb-bc70-53bd57411edb)
scores **98.11664037825032%** over all 25 public games. Its distinct unweighted
coverage is **181/183 = 98.907103825137%**. The certified paths contain 7001 actions,
and the card used 7069 API actions including resets. The all-game
[ONLINE shakedown](https://arcprize.org/scorecards/e293eeae-c0de-4263-a916-0a40ad282cbc)
preceded this definitive Competition replay.

GKM is universal at the producer level: all games share the fixed proposer contract,
Arena interface, blank scaffold, complexity coordinate, and replay gate. The
executable programs retained by that producer are game- and level-dependent learned
outputs, not a different hand-authored system for each game.

The action totals describe the final replay paths. Exploration used the local
`Arena.clone()` oracle and was not metered, so these values do not measure official
ARC-AGI-3 interaction or sample efficiency. The official wrapper provides `reset()`
and `step()` but no arbitrary state-fork operation.

The historical field `marginal_C` is also narrower than its name suggests. It is the
sum of positive net description-size changes in `legs.py` and `players.py`, including
a surcharge for container-literal elements. Same-size replacement can receive zero.
Use source provenance and replay, not the scalar alone, to assess reuse.

## Solved-Checkpoint Comparator Audit

The cross-system audit excludes interim synthesis revisions, repeated same-level
commits, and notebook edits. Its generated statistics are
[`manuscript/generated/comparator_stats.md`](manuscript/generated/comparator_stats.md);
the exact schema and raw-artifact reproduction protocol are documented in
[`manuscript/opine_world_comparison.md`](manuscript/opine_world_comparison.md).
The machine-readable rows and analyzers are under
[`audit_results/`](audit_results/) and the `audit_*` scripts. This README does not
copy their changing counts.

## Provenance

The canonical `wa30` and `ls20` artifacts are fresh, uniform reacquisitions. Each level
has its own replay-validated promotion manifest, exact winning-source boundary, parent
manifest link, and core-file hashes. Their canonical ledgers are respectively
`43, 20, 32, 50, 39, 23, 28, 34, 49` (318 total) and
`40, 54, 86, 114, 138, 170, 158` (760 total). Earlier `wa30` work was informed by
human play and mechanic-specific priors; it is preserved only as explicitly
superseded provenance and is not spliced into the uniform reacquisition.

Passing the source/environment taint audit does not turn these into official
interaction-efficiency evaluations: both reacquisitions still use the stronger
clone-enabled local harness, and discovery interactions are not included in replay
action totals.

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
- [`crack_lab/ARC_AGI3_CAMPAIGN_PLAN.md`](crack_lab/ARC_AGI3_CAMPAIGN_PLAN.md):
  canonical campaign, audit, scorecard, release, and supervisor plan.
- [`manuscript/arc_agi3.tex`](manuscript/arc_agi3.tex): scholarly manuscript.
- [`manuscript/README.md`](manuscript/README.md): reproducible paper/figure build and
  empirical tables, figures, and review sources.
- [`crack_lab/`](crack_lab/): harness and chronological findings.
- [`crack_lab/agent_solutions/`](crack_lab/agent_solutions/): promoted artifacts.
- [`manuscript/artifact_history/`](manuscript/artifact_history/): compact clean-history
  sidecar; mutable acquisition WIP is retained only until the final schema-v2 freeze
  and release-consumer audit.
- [`../REPRODUCE_ARC.md`](../REPRODUCE_ARC.md): replay procedure.

Run focused tests from the repository root:

```bash
python -m pytest arc/crack_lab/test_gkm_legs.py -q
```
