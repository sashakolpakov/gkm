# The Gödel–Kolmogorov Machine: Verifier-Gated Self-Improving Solver Growth for ARC-AGI-3

*Alexander Kolpakov* — [code and artifacts](https://github.com/sashakolpakov/gkm) · [manuscript](arc_agi3.tex)

The Gödel–Kolmogorov Machine (GKM) retains verified executable solver components and
exposes their source history for audit. The proposer contract, public Arena interface,
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

The frozen release contains one checkpoint record for every promoted level. The generated all-game marginal table is `arc/manuscript/generated/marginal_complexity_by_level.md`. The `wa30` and `ls20` rows additionally have one uniform, audited history sidecar per level. `marginal_C` means positive net retained-description growth per source file. Additions and deletions within the same file are netted before the positive part, so same-size replacement can receive zero. Its baseline is the source in the working workspace before that harness cycle; resumed WIP can already be present, so it is not necessarily growth from the preceding promoted source.
<!-- END GENERATED: ARC_ARTIFACT_STATUS -->

The Competition-Mode replay scores 98.1166%; raw coverage is 181/183 = 98.9071%.
The stored paths contain 7001 actions; the official scorecard records 7069 API calls.
These figures exclude clone-enabled discovery interaction, failed proposals, and
model inference. They do not measure held-out generalization or full discovery cost.

**Source accounting and reuse evidence.** The proxy counts nonblank, noncomment lines
plus AST container-literal elements in `legs.py` and `players.py`. The historical
`dc22` recorded total is 650 relative to winning-iteration working baselines. On
authenticated L2–L6 pairs, C sums to 601 and G to 801, taking retained size from 50
to 851; the initial scaffold is excluded from primary G comparisons.
The manuscript had conflated these baselines. The solver, replay, and counters were
valid; the audit now uses its complete source snapshot.

The strict acquisition-source audit admits 171 checkpoints, with 144 adjacent
transitions and 141 comparable marginals. Ten replay-verified boundaries lack exact
acquisition source: `ft09` L2, `tr87` L1–L6, `g50t` L3/L4, and `r11l` L3. Missing
levels are not bridged. The audit finds 54 static call-site cases referencing unchanged
earlier legs, including 13 with half-or-more novelty drops. Runtime call execution
is unmeasured; source calls plus whole-program replay do not prove executed reuse.

The complete `ls20` history has a conditional-AST novelty drop from 737 to 247 bytes
at L2, with a static call site to unchanged `follow_cardinal_runs`. `wa30` has a
complete oscillating historical ledger without an adjacent unchanged-leg case under
that narrow test. Sawtooth shape is observed, not enforced or shown to cause performance.

The primary paired C/G audit has 134 values across 24 games. Eighteen of the 20
histories with enough consecutive values reverse in both series. On aligned domains,
`wa30` has two C/G reversals each and `ls20` one each. `dc22` has two in C but none
in G: its recorded L5 trough is a baseline effect. Missing pairs are not bridged;
phase-fallback candidates are reported separately. See the paired table and figure
in manuscript §6.5 and `generated/source_charge_comparison.md`.

A matched-model, matched-budget library ablation and runtime call instrumentation
are needed to test causal search savings. See [revision evidence](REVISION_EVIDENCE.md)
and [response to comments](RESPONSE_TO_REVIEWS.md) for the corrections and limits.
