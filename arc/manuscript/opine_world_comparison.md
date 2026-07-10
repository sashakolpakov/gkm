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

