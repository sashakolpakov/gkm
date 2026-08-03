# The Gödel–Kolmogorov Machine: Verifier-Gated Self-Improving Solver Growth for ARC-AGI-3

*Alexander Kolpakov, July 2026* — code and artifacts: <https://github.com/sashakolpakov/gkm> · docs: <https://sashakolpakov.github.io/gkm/> · manuscript: [`arc_agi3.tex`](arc_agi3.tex)

*Frozen July 2026 manuscript snapshot; live campaign status is maintained in
[`../crack_lab/ARC_AGI3_CAMPAIGN_PLAN.md`](../crack_lab/ARC_AGI3_CAMPAIGN_PLAN.md).*

**Claim.** The Gödel–Kolmogorov Machine is a verifier-driven program-growth approach for local ARC-AGI-3 games. A coding proposer grows solver structure, the simulator validates promoted behavior by replay, and the admission loop prefers incumbent-leg composition before new code. Retained source states make the resulting acquisition and reuse claims auditable.

The producer is universal across the benchmark: every game uses the same proposer
contract, Arena interface, blank scaffold, complexity coordinate, and replay gate.
The executable programs it learns and promotes are game- and level-dependent.

The local harness exposes `step(action) -> frame`, `levels_completed`, and `clone()` for lookahead. Clone-enabled exploration is stronger than the official reset/step interface and is not included in replay action totals. A candidate program is promoted only if fresh replay validates more completed levels under:

```text
F = R + lambda * C,   R = -levels_completed
```

where historical `C` is positive net retained-size growth in the library and player files. Unchanged legs add no charge, but additions and deletions within a file can cancel. Source inspection and replay are required to attribute a low value to reuse.

**Difference from executable world models.** Executable-world-model agents try to build a predictive simulator, verify it, and plan through it. The Gödel–Kolmogorov Machine treats a world model as only one possible kind of useful structure. The object being optimized is broader: solver-program growth. A promoted structure may be a probe, perception routine, BFS, planner, reusable leg, literal replay path, or world model; what matters is that it improves replay-verified reward and pays its marginal description-length cost. The Gödel–Kolmogorov Machine is therefore closer to a Gödel/PowerPlay-style self-improving program with an MDL ledger than to a pure model-building agent. We use **GKM** below only after this full introduction.

**Current promoted artifacts.**

<!-- BEGIN GENERATED: ARC_ARTIFACT_STATUS -->
| Scope | Verified levels | Stored replay actions | Ledger charge |
|---|---:|---:|---:|
| Frozen v2 release (25 games) | 181/183 | 7001 | — |
| `wa30` uniform history | 9/9 | 597 | 318 |
| `ls20` uniform history | 7/7 | 365 | 760 |

The frozen release contains one checkpoint record for every promoted level. The generated all-game marginal table is `arc/manuscript/generated/marginal_complexity_by_level.md`. The `wa30` and `ls20` rows additionally have one uniform, audited history sidecar per level. `marginal_C` means positive net retained-description growth per source file. Additions and deletions within the same file are netted before the positive part, so same-size replacement can receive zero.
<!-- END GENERATED: ARC_ARTIFACT_STATUS -->

The frozen v2 release reaches 181/183 levels across all 25 games with 7001 stored
replay actions; only `lf52` L9--L10 remain. Its all-game ONLINE shakedown preceded a
definitive Competition-Mode replay scoring 98.11664037825032%; raw level coverage is
the separate quantity 181/183 = 98.907103825137%.

On `wa30`, GKM records a logistics game built around carry, helpers, handoffs,
neutralisation, and delivery. Its fresh canonical reacquisition has nine sequential
promotion manifests, a 597-action replay, and per-level charges
`43, 20, 32, 50, 39, 23, 28, 34, 49`. The superseded exploratory lineage remains
available for historical comparison but is not spliced into this ledger.

On `ls20`, the uniform ledger is `40, 54, 86, 114, 138, 170, 158`. The stricter
winning-checkpoint audit finds a sharp conditional-AST drop at L2, from 737 to 247
compressed novelty bytes, while the winning player directly calls the unchanged
`follow_cardinal_runs` leg. The later exact marginals rise through 299 before dipping
to 292 at L7. The cumulative executable solver need not shrink; attribution comes
from the conditional marginal and literal call together.

**Audit trail.** GKM is meant to leave evidence, not just outputs. Every admitted
endpoint has replay and hash evidence; the winning-source subset additionally supports
source-marginal and reuse analysis. This separates literal path reconstruction from
historical executable growth.

**Comparison.** All 181 claimed boundaries are replay-verified wins. The stricter
source audit admits 174 exact winning-source checkpoints; it excludes `ft09` L2 and
`tr87` L1--L6 deterministic reconstructions. The admitted set contains 57 hard
direct-reuse witnesses, including 14 with half-or-more marginal drops. OPINE has
four direct engine-reuse wins, with sharp coupled drops at `lp85` L4 and `tu93` L3,
so it is not solving every level wholly anew. baseline1 has four exact retained-source
contractions, but the released exact adjacent winning commands provide no coupled
unchanged-world-model witness. Retrodict releases curated memory rather than
executable winning checkpoints.

**Current limitation and request.** This is not yet a private ARC-AGI-3 result or a
compute-matched head-to-head. The natural next tests are an independently reviewed
artifact audit, a private-set evaluation, and a compute-matched comparison against
graph exploration and executable world models.

**Bottom line.** GKM treats local solver development as verifier-gated acquisition of reusable program structure. Replay establishes endpoint behavior; checkpoint JSON and retained source expose when novelty was admitted, when prior legs sufficed, and how the solver was refactored.
