# GKM: Auditable Self-Improvement for ARC-AGI-3

*Alexander Kolpakov, July 2026* — code and artifacts: <https://github.com/sashakolpakov/gkm> · docs: <https://sashakolpakov.github.io/gkm/> · manuscript: [`arc_agi3.tex`](arc_agi3.tex)

**Claim.** GKM is a verifier-driven program-growth loop for local ARC-AGI-3 games. A coding proposer grows solver structure, the simulator validates promoted behavior by replay, and retained source states make that growth auditable.

The local harness exposes `step(action) -> frame`, `levels_completed`, and `clone()` for lookahead. Clone-enabled exploration is stronger than the official reset/step interface and is not included in replay action totals. A candidate program is promoted only if fresh replay validates more completed levels under:

```text
F = R + lambda * C,   R = -levels_completed
```

where historical `C` is positive net retained-size growth in the library and player files. Unchanged legs add no charge, but additions and deletions within a file can cancel. Source inspection and replay are required to attribute a low value to reuse.

**Difference from executable world models.** Executable-world-model agents try to build a predictive simulator, verify it, and plan through it. GKM treats a world model as only one possible kind of useful structure. The object being optimized is broader: solver-program growth. A promoted structure may be a probe, perception routine, BFS, planner, reusable leg, literal replay path, or world model; what matters is that it improves replay-verified reward and pays its marginal description-length cost. This makes GKM closer to a Goedel/PowerPlay-style self-improving program with an MDL ledger than to a pure model-building agent.

**Current promoted artifacts.**

<!-- BEGIN GENERATED: ARC_ARTIFACT_STATUS -->
| Game | Verified levels | Replay actions | Published ledger charge |
|---|---:|---:|---:|
| `wa30` | 9/9 | 596 | 1458 |
| `ls20` | 7/7 | 393 | 362 |

Both published ledgers contain one entry for every replay-validated level. The operational checkpoint may retain only records accumulated after its resume base; the manuscript sidecar supplies the complete audited history. `marginal_C` means positive net retained-description growth per source file. Additions and deletions within the same file are netted before the positive part, so same-size replacement can receive zero.
<!-- END GENERATED: ARC_ARTIFACT_STATUS -->

On `wa30`, GKM records a logistics game built around carry, helpers, handoffs, neutralisation, and delivery. The solver discovers and preserves reusable structure: freezing target regions before delivered objects overwrite them, complementing autonomous helpers instead of competing with them, exploiting asymmetric carry collision at wall boundaries, neutralising agents that undo delivery, and refactoring repeated transport into ferry legs. The level-9 WIP trail is especially useful for audit: a recovered verified suffix was decoded into five repeated grab-carry-release operations and refactored into `grab_carry_release` and `ferry_each`, so the promoted player is compact composition rather than an opaque replay.

On `ls20`, the complete net-growth trace is sawtoothed. The two troughs coincide with thin players that call unchanged search routines and pass fresh replay. Larger values retain more source or literal plans. The attribution comes from the code and replay, not from the scalar alone.

**Audit trail.** GKM is meant to leave evidence, not just outputs. Each promoted step is backed by fresh replay, preserved WIP snapshots, a marginal-complexity charge, and a distinction between literal action recovery and reusable refactor. This makes the artifact reviewable: an external evaluator can inspect which information entered through interaction, which code was introduced, what was charged as a literal, and what later became a reusable leg.

**Comparison.** The ARC-AGI-3 benchmark paper reports that public-environment harnesses solved `ls20`, `ft09`, and `vc33`; it does not mention `wa30` in the paper source. A graph-exploration baseline evaluates `ls20` but not `wa30`; at the 4,000-interaction comparison point it solves 2 levels on `ls20` and reports that `ls20` levels 3+ become intractable for exhaustive exploration. The closest comparator is the executable-world-model baseline: it reports `ls20` 7/7 with both GPT-5.4 and GPT-5.5, and `wa30` 7/9 with GPT-5.4 versus 4/9 with GPT-5.5. GKM is therefore not unique on `ls20`; its strongest competitive point is `wa30` 9/9 replay-validated, plus the fact that every promoted step is backed by preserved WIP snapshots, replay validation, marginal complexity accounting, charged literals, and leg-library refactors.

**Current limitation and request.** This is not yet a private ARC-AGI-3 leaderboard result and not yet a compute-matched head-to-head. The next step is to turn the current artifact into a collaboration-grade evaluation: reproduce on more public games, run a compute-matched comparison against graph exploration and executable world models, and harden the leakage/audit protocol. We are looking for collaborators who can help with one or more of: ARC-AGI-3 evaluation protocol, compute/model access, independent artifact review, private-set or organizer-facing evaluation, and funding for a systematic benchmark run.

**Bottom line.** GKM reframes local solver development as an artifact audit. Replay establishes endpoint behavior; checkpoint JSON and retained source establish exactly which growth records are available.
