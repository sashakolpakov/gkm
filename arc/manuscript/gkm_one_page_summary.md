# GKM: Auditable Self-Improvement for ARC-AGI-3

**Claim.** GKM is a verifier-driven self-improvement loop for ARC-AGI-3 agents. A coding proposer grows solver structure from the raw interface, the game simulator verifies improvement by replay, and marginal description length decides what new structure is worth keeping. The result is not just a solved-level count; it is an auditable ledger of how competence was acquired.

The agent receives only `step(action) -> frame`, `levels_completed`, and `clone()` for lookahead. Nothing in the harness names the avatar, mechanics, goals, helpers, walls, switches, or plans. A candidate program is promoted only if fresh replay validates more completed levels under:

```text
F = R + lambda * C,   R = -levels_completed
```

where `C` is marginal description length. Reused legs are free, new reusable legs are charged once, per-level glue is charged, and literal recovered paths are charged as literals. This operationalizes a Kolmogorov/MDL version of Schmidhuber-style self-improvement: the system may rewrite or grow itself, but only admitted structure that improves verified reward survives.

**Difference from executable world models.** Executable-world-model agents try to build a predictive simulator, verify it, and plan through it. GKM treats a world model as only one possible kind of useful structure. The object being optimized is broader: solver-program growth. A promoted structure may be a probe, perception routine, BFS, planner, reusable leg, literal replay path, or world model; what matters is that it improves replay-verified reward and pays its marginal description-length cost. This makes GKM closer to a Goedel/PowerPlay-style self-improving program with an MDL ledger than to a pure model-building agent.

**Current promoted artifacts.**

- `wa30`: replay-validated through level 9/9, 596 actions, total marginal complexity 1243.
- `ls20`: replay-validated through level 7/7, 393 actions, total marginal complexity 362.

On `wa30`, GKM records a logistics game built around carry, helpers, handoffs, neutralisation, and delivery. The solver discovers and preserves reusable structure: freezing target regions before delivered objects overwrite them, complementing autonomous helpers instead of competing with them, exploiting asymmetric carry collision at wall boundaries, neutralising agents that undo delivery, and refactoring repeated transport into ferry legs. The level-9 WIP trail is especially useful for audit: a recovered verified suffix was decoded into five repeated grab-carry-release operations and refactored into `grab_carry_release` and `ferry_each`, so the promoted player is compact composition rather than an opaque replay.

On `ls20`, the marginal-complexity trace shows why this accounting matters. It is sawtoothed, not monotone: `43 -> 2` and `45 -> 3` are leg-reuse drops, while `45`, `72`, and `130` are novelty spikes caused by new mechanics or charged plan artifacts. Level 7 then drops to `67`, consistent with partial reuse of level-6 lock/display structure plus new fog/mapping. This is the central evidence: marginal free energy acts as a novelty detector, recording when transfer is sufficient and when the game forces new structure. A solved-level count cannot show this; the sawtooth profile can.

**Comparison.** The ARC-AGI-3 benchmark paper reports that public-environment harnesses solved `ls20`, `ft09`, and `vc33`; it does not mention `wa30` in the paper source. A graph-exploration baseline evaluates `ls20` but not `wa30`; at the 4,000-interaction comparison point it solves 2 levels on `ls20` and reports that `ls20` levels 3+ become intractable for exhaustive exploration. The closest comparator is the executable-world-model baseline: it reports `ls20` 7/7 with both GPT-5.4 and GPT-5.5, and `wa30` 7/9 with GPT-5.4 versus 4/9 with GPT-5.5. GKM is therefore not unique on `ls20`; its strongest competitive point is `wa30` 9/9 replay-validated, plus the fact that every promoted step is backed by preserved WIP snapshots, replay validation, marginal complexity accounting, charged literals, and leg-library refactors.

**Current limitation and request.** This is not yet a private ARC-AGI-3 leaderboard result and not yet a compute-matched head-to-head. The next step is to turn the current artifact into a collaboration-grade evaluation: reproduce on more public games, run a compute-matched comparison against graph exploration and executable world models, and harden the leakage/audit protocol. We are looking for collaborators who can help with one or more of: ARC-AGI-3 evaluation protocol, compute/model access, independent artifact review, private-set or organizer-facing evaluation, and funding for a systematic benchmark run.
