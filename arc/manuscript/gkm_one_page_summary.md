# The Gödel–Kolmogorov Machine: Verifier-Gated Self-Improving Solver Growth for ARC-AGI-3

*Alexander Kolpakov, July 2026* — code and artifacts: <https://github.com/sashakolpakov/gkm> · docs: <https://sashakolpakov.github.io/gkm/> · manuscript: [`arc_agi3.tex`](arc_agi3.tex)

**Claim.** The Gödel–Kolmogorov Machine is a verifier-driven program-growth approach for local ARC-AGI-3 games. A coding proposer grows solver structure, the simulator validates promoted behavior by replay, and the admission loop prefers incumbent-leg composition before new code. Retained source states make the resulting acquisition and reuse claims auditable.

The local harness exposes `step(action) -> frame`, `levels_completed`, and `clone()` for lookahead. Clone-enabled exploration is stronger than the official reset/step interface and is not included in replay action totals. A candidate program is promoted only if fresh replay validates more completed levels under:

```text
F = R + lambda * C,   R = -levels_completed
```

where historical `C` is positive net retained-size growth in the library and player files. Unchanged legs add no charge, but additions and deletions within a file can cancel. Source inspection and replay are required to attribute a low value to reuse.

**Difference from executable world models.** Executable-world-model agents try to build a predictive simulator, verify it, and plan through it. The Gödel–Kolmogorov Machine treats a world model as only one possible kind of useful structure. The object being optimized is broader: solver-program growth. A promoted structure may be a probe, perception routine, BFS, planner, reusable leg, literal replay path, or world model; what matters is that it improves replay-verified reward and pays its marginal description-length cost. The Gödel–Kolmogorov Machine is therefore closer to a Gödel/PowerPlay-style self-improving program with an MDL ledger than to a pure model-building agent. We use **GKM** below only after this full introduction.

**Current promoted artifacts.**

<!-- BEGIN GENERATED: ARC_ARTIFACT_STATUS -->
| Game | Verified levels | Replay actions | Published ledger charge |
|---|---:|---:|---:|
| `wa30` | 9/9 | 596 | 1458 |
| `ls20` | 7/7 | 393 | 362 |

Both published ledgers contain one entry for every replay-validated level. The operational checkpoint may retain only records accumulated after its resume base; the manuscript sidecar supplies the complete audited history. `marginal_C` means positive net retained-description growth per source file. Additions and deletions within the same file are netted before the positive part, so same-size replacement can receive zero.
<!-- END GENERATED: ARC_ARTIFACT_STATUS -->

Five partial artifacts extend the same protocol: `ft09`, `r11l`, and `tr87`
reach L6, `g50t` reaches L5, and `sp80` reaches L4; `tu93` reaches L1. Across
all eight endpoints the current local frontier is 44/183 levels with 1692 stored
replay actions. The last published scorecard predates seven of those clears.

On `wa30`, GKM records a logistics game built around carry, helpers, handoffs, neutralisation, and delivery. The solver discovers and preserves reusable structure: freezing target regions before delivered objects overwrite them, complementing autonomous helpers instead of competing with them, exploiting asymmetric carry collision at wall boundaries, neutralising agents that undo delivery, and refactoring repeated transport into ferry legs. The level-9 WIP trail is especially useful for audit: a recovered verified suffix was decoded into five repeated grab-carry-release operations and refactored into `grab_carry_release` and `ferry_each`, so the promoted player is compact composition rather than an opaque replay.

On `ls20`, the harness-native acquisition ledger is sawtoothed. The stricter winning-checkpoint audit independently finds a sharp conditional-AST drop at L7, from 682 to 222 compressed novelty bytes, while the winning player directly calls the unchanged `execute_path` leg. The cumulative executable solver does not shrink; the attribution comes from the marginal and literal call together.

**Audit trail.** GKM is meant to leave evidence, not just outputs. Each promoted step is backed by fresh replay, preserved WIP snapshots, a marginal-complexity charge, and a distinction between literal action recovery and reusable refactor. This makes the artifact reviewable: an external evaluator can inspect which information entered through interaction, which code was introduced, what was charged as a literal, and what later became a reusable leg.

**Comparison.** The exact winning-checkpoint test couples conditional AST novelty to direct calls of unchanged definitions. GKM has ten direct leg-reuse wins; `g50t` L4 and `ls20` L7 also have half-or-more marginal drops. The new `ft09` L6 winner directly calls unchanged `solve_coupled_key_board`, although its conditional AST decline is not sharp. OPINE has four direct engine-reuse wins, with sharp coupled drops at `lp85` L4 and `tu93` L3, so it is not solving every level wholly anew. baseline1 has four exact retained-source contractions, but all 18 exact adjacent winning commands are fresh literal action programs and none invokes retained world-model code. Retrodict releases curated memory rather than executable winning checkpoints.

**Current limitation and request.** This is not yet a private ARC-AGI-3 leaderboard result and not yet a compute-matched head-to-head. The next step is to turn the current artifact into a collaboration-grade evaluation: reproduce on more public games, run a compute-matched comparison against graph exploration and executable world models, and harden the leakage/audit protocol. We are looking for collaborators who can help with one or more of: ARC-AGI-3 evaluation protocol, compute/model access, independent artifact review, private-set or organizer-facing evaluation, and funding for a systematic benchmark run.

**Bottom line.** GKM treats local solver development as verifier-gated acquisition of reusable program structure. Replay establishes endpoint behavior; checkpoint JSON and retained source expose when novelty was admitted, when prior legs sufficed, and how the solver was refactored.
