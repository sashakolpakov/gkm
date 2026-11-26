# ARC-AGI-3: replay-validated program growth

GKM retains executable solver components after verified progress, fresh replay,
taint and provenance checks. A coding-model proposer discovers the game-specific
mechanics and writes the solver; the host supervisor orchestrates and verifies.
Complexity and free energy are recorded diagnostics, not admission or candidate
ranking rules.

The common discovery interface supplies frames, actions, progress and local clone
lookahead. Official Competition scoring replays frozen action paths without a model
or clone calls. Public-set solution replay and discovery are separate measurements.

## Current GPT-6.1 release

The [frozen v3 release](crack_lab/releases/arc_agi3_gkm_v3_182/README.md) contains a
new GPT-6.1-sol zero-seed acquisition across all 25 games: 182/183 levels, 24 complete
games and 6,685 stored actions. Its [closed Competition scorecard](https://arcprize.org/scorecards/fb027490-5351-4a77-85f2-41959e3e6b64)
records 99.27272727%. Only lf52 level 10 is unclaimed under the fixed 600-action
full-game limit.

The release includes original winning source and promotion chains, the actual
producer source, independently certified replay evidence, token accounting and a
content-addressed receipt. Use its [reproduction guide](crack_lab/releases/arc_agi3_gkm_v3_182/REPRODUCE.md)
for offline checks and API replay. The supervisor supplied no game-specific solver
advice in this campaign.

## Frozen manuscript dataset

The [selected manuscript](manuscript/README.md) analyses the earlier GPT-5.6 dataset.
Its results and figures remain tied to the [frozen v2 release](crack_lab/releases/arc_agi3_gkm_v2_181/README.md)
and pinned source-history revision. They are not replaced by the new campaign.

<!-- BEGIN GENERATED: ARC_ARTIFACT_STATUS -->
| Scope | Verified levels | Stored replay actions | Ledger charge |
|---|---:|---:|---:|
| Frozen v2 release (25 games) | 181/183 | 7001 | — |
| `wa30` uniform history | 9/9 | 597 | 318 |
| `ls20` uniform history | 7/7 | 365 | 760 |

The frozen release contains one checkpoint record for every promoted level. The supporting source-charge table is `arc/manuscript/generated/marginal_complexity_by_level.md`. The `wa30` and `ls20` rows additionally have one uniform, audited history sidecar per level. `marginal_C` means positive net retained-description growth per source file. Additions and deletions within the same file are netted before the positive part, so same-size replacement can receive zero. Its baseline is the source in the working workspace before that harness cycle; resumed WIP can already be present, so it is not necessarily growth from the preceding promoted source. The conditional marginal-complexity proxy is published separately in `arc/manuscript/generated/conditional_marginal_profiles.md`; the source-growth counter is not a conditional-complexity measurement.
<!-- END GENERATED: ARC_ARTIFACT_STATUS -->

The source histories, conditional marginal profiles and reuse audits can be
regenerated through [REPRODUCE_ARC.md](../REPRODUCE_ARC.md). Historical acquisition
snapshots needed by those analyses are read from the explicitly pinned Git tree.
The compact promotion evidence and manuscript sidecars remain in the checkout;
failed continuations and exploratory scratch notes are not publication inputs.

## Implementation and operation

- [`crack_lab/gkm_legs.py`](crack_lab/gkm_legs.py): proposal, retained leg library,
  execution, replay and admission.
- [`crack_lab/gkm_arena.py`](crack_lab/gkm_arena.py) and
  [`arc_agi3_adapter.py`](arc_agi3_adapter.py): discovery interface and game adapters.
- [`audit_submission_taint.py`](audit_submission_taint.py) and
  [`proposer_integrity.py`](crack_lab/releases/arc_agi3_gkm_v3_182/producer_source/arc/crack_lab/proposer_integrity.py): shared
  provenance and supervisor-noninterference checks.
- [`producer_source/PROTOCOL.md`](crack_lab/releases/arc_agi3_gkm_v3_182/producer_source/PROTOCOL.md):
  the actual v3 campaign protocol and retry ladder. The immutable producer is
  published inside its release rather than inferred from later framework development.
- [`manuscript/arc_agi3.tex`](manuscript/arc_agi3.tex) and
  [`manuscript/gkm_one_page_summary.md`](manuscript/gkm_one_page_summary.md): paper and
  companion summary.

Generated queues, failed WIP, scratch workspaces, sockets and build intermediates
are disposable host state. Accepted release evidence and sealed accounting captures
are retained separately. The [campaign master plan](crack_lab/ARC_AGI3_CAMPAIGN_PLAN.md)
keeps historical policy and prospective design context without supplying a live queue.
