# The Gödel–Kolmogorov Machine: Verifier-Gated Self-Improving Solver Growth for ARC-AGI-3

*Alexander Kolpakov* — [code and artifacts](https://github.com/sashakolpakov/gkm) · [manuscript](arc_agi3.tex)

The Gödel–Kolmogorov Machine (GKM) retains verified executable solver components and
exposes their source history for audit. Its central question is how much new description
the next verified solution needs, given the retained program. The proposer contract, public Arena interface,
blank scaffold, and admission gate are shared across games; learned programs remain
game- and level-dependent. The intellectual lineage is Schmidhuber’s Gödel machine,
PowerPlay, and artificial curiosity.

The host tries bounded calls to incumbent legs before asking a coding model for new
code. The first candidate reaching the exact next boundary and passing execution,
independent full-path replay, taint/provenance, and containment checks is promoted.
Its full solver tree becomes the next parent. Source cost and
`F_diag(k) = -k + 0.02 * sum(recorded_C_j)` are recorded afterwards and do not admit,
reject, or rank candidates. Free energy ranking and curiosity-based control are
proposed extensions. This admission contract and later endpoint certification do
not establish that older discovery used the later containment controls. The later
no-debrief setting is distinct from inherited paid-debrief histories.

**Frozen public replay result.**

<!-- BEGIN GENERATED: ARC_ARTIFACT_STATUS -->
| Scope | Verified levels | Stored replay actions | Ledger charge |
|---|---:|---:|---:|
| Frozen v2 release (25 games) | 181/183 | 7001 | — |
| `wa30` uniform history | 9/9 | 597 | 318 |
| `ls20` uniform history | 7/7 | 365 | 760 |

The frozen release contains one checkpoint record for every promoted level. The supporting source-charge table is `arc/manuscript/generated/marginal_complexity_by_level.md`. The `wa30` and `ls20` rows additionally have one uniform, audited history sidecar per level. `marginal_C` means positive net retained-description growth per source file. Additions and deletions within the same file are netted before the positive part, so same-size replacement can receive zero. Its baseline is the source in the working workspace before that harness cycle; resumed WIP can already be present, so it is not necessarily growth from the preceding promoted source. The conditional marginal-complexity proxy is published separately in `arc/manuscript/generated/conditional_marginal_profiles.md`; the source-growth counter is not a conditional-complexity measurement.
<!-- END GENERATED: ARC_ARTIFACT_STATUS -->

The Competition-Mode replay scores 98.1166%; raw coverage is 181/183 = 98.9071%.
The stored paths contain 7001 actions; the official scorecard records 7069 API calls.
These figures exclude clone-enabled discovery interaction, failed proposals, and
model inference. They do not measure held-out generalization or full discovery cost.

**Conditional marginal complexity.** The ideal question is K(next program | retained
program), not the difference in their source sizes. The empirical proxy M compresses
novel normalized-AST statements after removing literal matches to the preceding exact
winning program. It is representation-dependent, omits complete decoding metadata,
and does not estimate ideal K with a calibrated error bound.

The audit admits 171 exact winning checkpoints and 141 comparable marginals: 64
decrease, including 21 half-or-more drops. Generalized source accounting finds
retained-definition paths in 115 of 144 adjacent source pairs, including 18 sharp
drops. Sixty-one pairs require following intermediate glue beyond a direct call.
Missing source is not bridged. Historical runtime call events were not collected. The main figure compares three cases; the appendix plots all
25 games, including unknown and unclaimed boundaries.

The `dc22` proxy is 1058, 604, 716, 885, 828, 1379 compressed bytes: it has two dips
despite strictly increasing promoted-source growth. `wa30` has troughs at L2 and L6.
In `ls20`, M falls from 737 to 247 at L2, with a static call site to unchanged
`follow_cardinal_runs`; later route data add novelty while the leg remains unchanged.
At both dc22 dips, the new routine calls unchanged walk_segments and toggle_control;
at both wa30 troughs, it calls unchanged walk and interact. These are indirect
reuse paths omitted by the narrow direct-call count. The case audit also identifies
a compact-action case: cn04 L3 falls from 337 to 245 bytes, with a 23-action player
whose only call is env.step.

**Supporting source counters.** Recorded C compares with the workspace before a
winning cycle; G compares authenticated adjacent promoted states. For `dc22`, paired
L2–L6 totals are C=601 and G=801 as retained size rises from 50 to 851. The full
recorded C remains 650. The earlier manuscript interpretation and incomplete source
reconstruction were corrected without repairing the valid solver, counter or replay.
These counters neither replace conditional marginals nor reconstruct all failed
work over the discovery process.

A matched-model, matched-budget library ablation is needed to test causal search
savings. Runtime call measurements can now be collected with the separate observer. See [marginal profiles](generated/conditional_marginal_profiles.md),
[revision evidence](REVISION_EVIDENCE.md) and [response to comments](RESPONSE_TO_REVIEWS.md).

The shared code-only accountant follows aliases, callbacks, wrappers, dispatch tables
and methods. A separate replay observer records actual runtime dispatch, including
generated glue, without changing the candidate or admission rule. Source hashes bind
the observations; unresolved source bindings are never treated as non-reuse. See
[accounting protocol](../REUSE_ACCOUNTING.md) and
[coverage table](generated/reuse_accounting.md).
