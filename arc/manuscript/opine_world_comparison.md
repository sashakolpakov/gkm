# OPINE-World Comparison

Source checked: <https://github.com/david-courtis/opine-world> and
<https://arxiv.org/abs/2607.01531>.

OPINE-World is now the direct ARC-AGI-3 comparator. It reports 20/25 public
games and 160/183 levels, with action-efficiency score 78.4 against the human
baseline. The method is an object-centric programmatic world-model loop: one
agent acts, a second agent synthesizes `game_engine.py` with
`extract_objects(frame)`, `transition_function(state, action)`, and
`reward_function(state, action, new_state)`, exact replay admits the model, a
planner searches it, and ontology error steers exploration toward poorly
explained object types. The README states that the reported ARC-3 runs used
raw rendered frames, no game source/object identities/action semantics/hints,
sandboxed agents, Opus 4.8 for both agents, and one online competition-mode run
per public game.

The right positioning is not "GKM beats OPINE-World" on solved count. It does
not: GKM currently has deep replay-validated artifacts on two preview games
(`wa30` 9/9 and `ls20` 7/7), while OPINE-World reports a broad public-set run.
OPINE-World is therefore the performance baseline to beat or reproduce against.

The technical distinction is the optimized object. OPINE-World optimizes a
predictive object-centric world model plus planner. GKM optimizes admitted
solver-program growth under marginal description-length accounting. A promoted
GKM artifact may be a world model, but it may also be a probe, perception
routine, BFS, literal path, transport leg, refactored mechanic, or thin
composition over a leg library. Replay-verified reward decides admission, and
the marginal-C ledger records whether a solved level came from transfer,
novelty, or a charged literal.

This distinction is potentially complementary rather than antagonistic:
OPINE's ontology error is a principled curiosity signal over object-type
adequacy, while GKM's marginal-C sawtooth is an audit signal over reusable
solver structure. The clean hybrid experiment is to make an OPINE-style
synthesized world model one candidate leg inside GKM, charge its code and
planner in marginal C, and test whether later levels reuse it cheaply or switch
to non-model legs when those are cheaper.

Conservative manuscript update:

- Replace any phrasing that implies executable-world-model baselines are weak
  or only preliminary. OPINE-World is a strong public-set result.
- Keep GKM's claim narrow: auditable self-improvement and marginal-complexity
  accounting, not current public-set dominance.
- Use `wa30` 9/9 as a depth/anomaly point only after a game-overlap check
  against OPINE's per-game artifact archive.
- The necessary next experiment is compute-matched: same 25 public games, same
  competition mode, comparable model budget, and artifact-level audit of source
  isolation, replay traces, action counts, and complexity/growth traces.

Artifact-audit plan for OPINE:

1. Download the published run-artifact archive and index, for each game and
   level, the synthesized `game_engine.py`, natural-language world model,
   transition buffer, planner calls, critic calls, model rewrites, action count,
   and final replay.
2. Measure reuse versus rediscovery. A genuinely reusable world model should
   make later levels cheaper: fewer synthesis turns, fewer transition-model
   rewrites, fewer exploratory actions, and stable object/type names or
   transition rules across levels. If each level restarts hypothesis search,
   rewrites object ontology, or solves by bespoke planner patches, then the
   20/25 score is mostly agentic spend rather than learned structure.
3. Charge an explicit cost ledger: LLM calls, wall time, action count,
   synthesized-code churn, and per-level reset/retry count. Compare against
   GKM's marginal-C ledger and sawtooth reuse trace. OPINE's README reports the
   sweep used four Claude Max accounts; that is not disqualifying, but it must
   be surfaced as compute budget, not hidden behind solved count.
4. Check the world-model claim directly. For each admitted model, test whether
   it predicts held-out later-level transitions before seeing them, or whether it
   is only exact on the accumulated replay buffer. Exact replay on seen
   transitions is necessary but not enough to establish transfer.
5. Check planner dependence. Count cases where the planner succeeds from the
   admitted model versus cases where the goal-directed agent supplies custom
   plans because the model/planner is inadequate. These should be reported
   separately.
6. Produce a game-overlap table against GKM (`wa30`, `ls20`, and any shared
   public IDs): solved depth, action count, number of model/synth turns, source
   isolation evidence, and whether later levels reused earlier structure.

Pass/fail criterion for the critique: do not argue from impression. Show, from
their own artifacts, whether OPINE has cross-level compression/reuse or whether
it burns fresh agentic search per level. If the artifacts show stable reusable
models, concede that. If they show repeated rediscovery and high rewrite churn,
position GKM's contribution as the missing audit/MDL accounting that OPINE's
score alone does not expose.
