# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## g50t level 1
- Avatar = the object that TRANSLATES under a move (detected via colour+shape
  match). Movement actions 1/2/3/4; action 5 = USE.
- USE on a "switch" tile reconfigures the maze (opens gates -> avatar reachable
  set grows); elsewhere USE is a reset. Goal = drive avatar into goal region;
  reward is sparse (levels_completed).
- HIDDEN STATE: byte-identical frames can transition differently, so frame-key
  BFS is invalid. Plan as concrete action sequences on clones, dedup only on
  avatar position within a fixed gate config.
- Legs added: avatar_tl, _move_explore, plan_unlock_reach / solve_unlock_reach
  (general "unlock-then-reach"). play_level_1 = solve_unlock_reach.

## Debrief refactor (post-clear)
- players.py holds exactly one player, and it was already a thin one-liner
  (`play_level_1 = solve_unlock_reach`); no cross-player duplication to merge.
  The duplication was INSIDE legs.py instead.
- Extracted `clone_after(env, actions)`: "fresh clone, replay these actions,
  never touch the original". This idiom was copy-pasted at three sites
  (avatar_tl probe step, _move_explore BFS expansion, plan_unlock_reach switch
  replay+USE). Written ONCE now; the callers are one-liners.

## Candidate higher-order leg: plan_and_commit(env, planner, **kw)
- RECURRING COMPOSITION PATTERN. Because the substrate has HIDDEN STATE, every
  skill splits into two phases:
    1. `plan_X(clone)   -> concrete action list | None`  (search on a clone)
    2. `run_path(env, plan)`                              (commit on real env)
  i.e. `solve_X == plan_and_commit(env, plan_X, **kw)`.
- This is a higher-order leg: it takes any planner and turns it into a
  committing solver. `solve_unlock_reach` is now literally
  `plan_and_commit(env, plan_unlock_reach, max_toggles=...)`, and any future
  `plan_*` planner gets its `solve_*` for free by reusing this wrapper.
  Players stay thin: `play_level_K` just names which planner to commit.

## Level 2 (g50t)
Same objective as L1 (avatar = compact color-9 block reaches the 9-outline goal
chamber). But the chamber's region is joined to the avatar's only through gates
that open REMOTELY when the avatar walks over wall segments; a USE resets the
avatar to start while PRESERVING opened gates. Crucially a single USE never
enlarges reachability here, so `solve_unlock_reach` (which prunes USEs that
don't immediately grow reach) fails. Added general legs:
  * `_avatar_pos` / `fast_reach` -- clone-cheap avatar-position BFS within a
    fixed gate configuration.
  * `plan_unlock_macro` / `solve_unlock_macro` -- best-first search over
    "walk-somewhere-then-USE" macros keyed on the frozenset of reachable
    positions, chaining staged gate openings until the goal is movement-reachable.
`play_level_2` just calls `solve_unlock_macro`.

## Debrief refactor (post-clear of L2)
- Both players are already thin (`play_level_1 = solve_unlock_reach`,
  `play_level_2 = solve_unlock_macro`); no player-side duplication. As on L1,
  the real duplication lived INSIDE legs.py, between the L1 and L2 solve paths.
- Deduped the reach-BFS: `_move_explore` (L1) and `fast_reach` (L2) were the
  SAME skill -- "flood the avatar's positions with moves, one clone per edge,
  dedup on position, stop on reward" -- written twice, differing ONLY in the
  avatar-locating function. Extracted `_reach_bfs(start, locate)` written ONCE;
  `_move_explore = _reach_bfs(., avatar_tl)` and
  `fast_reach     = _reach_bfs(., framewise _avatar_pos)` are now one-liners.
- Deduped the successor step: both unlock planners advance the gate config the
  same way -- pick a reachable tile, walk there, press USE. Extracted
  `_use_macros(node, reach, reach_fn)` yielding
  `(macro, child_env, child_reward_path, child_reach)`, cheapest walk first.
  `plan_unlock_reach` and `plan_unlock_macro` now share it and differ ONLY in
  how they schedule/prune those successors.

## Candidate higher-order leg: unlock_search(env, reach_fn, schedule)
- RECURRING COMPOSITION PATTERN across L1 and L2. Every "unlock-then-reach"
  solver is a FRONTIER SEARCH over gate-configurations where:
    1. at each node, `reach_fn` floods movement; if the reward is
       movement-reachable, return the winning path;
    2. else the successors are exactly the `_use_macros` (walk-to-a-reachable-
       tile-then-USE); each opens some gates -> a new gate-config child.
  The ONLY thing that varies is the SCHEDULER/PRUNER over that frontier:
    * `plan_unlock_reach` = bounded-depth DFS, keep only USEs that immediately
      grow the reachable set (`max_toggles` depth cap);
    * `plan_unlock_macro` = best-first over children keyed on the frozenset of
      reachable positions, ordered by -|reach| (no growth requirement, so it
      chains staged openings that only pay off later; bounded by max_expand).
- So the candidate higher-order leg is
  `unlock_search(env, reach_fn, schedule)`: parameterise the shared reach+macro
  machinery by a `schedule` policy (bounded-DFS-grow-prune vs best-first-on-
  reach-set) and BOTH existing planners fall out as instances -- future unlock
  variants would supply only a new scheduling policy, not re-implement reach or
  macro expansion. Pairs with the existing `plan_and_commit` wrapper so each
  `solve_*` stays free and players stay one-line compositions.

## Level 3 (g50t)
Same maze family as L2: the goal chamber only becomes movement-reachable after
chaining several `walk-somewhere-then-USE` openings (a USE resets the avatar to
start but PRESERVES gates opened en route). Ran the arena and observed L3 clears
with NO new leg -- the general `solve_unlock_macro` already covers it. Confirmed
`solve_unlock_reach`'s single-USE-must-grow pruning is still too weak here (same
reason as L2), but the best-first USE-macro search stages the openings and wins.

## Debrief refactor (post-clear of L3)
- Compared `play_level_3` to the earlier players. FINDING: L3 introduced no new
  skill -- `play_level_2` and `play_level_3` are the SAME thin call,
  `solve_unlock_macro(env)`. The only "duplication" was cosmetic: `play_level_2`
  passed `max_expand=400`, which is exactly the leg's default, hiding the fact
  that the two players are identical compositions.
- Also spotted the same redundancy in `play_level_1`, which passed
  `max_toggles=2` (again the default of `solve_unlock_reach`).
- Refactor (behaviour-preserving): dropped the redundant default arguments so
  every player is now a bare general-leg call. L2 and L3 now read identically
  (`solve_unlock_macro(env)`), making the leg-reuse explicit. No code was moved
  INTO legs.py because the shared skill (`solve_unlock_macro`) was already
  written ONCE; the players stay thin one-liners naming which leg to commit.

## Candidate higher-order leg (reaffirmed by L3): unlock_search(env, reach_fn, schedule)
- RECURRING COMPOSITION PATTERN, now with THREE levels of evidence. The stronger
  observation from L3 is the "one general leg, many levels" pattern: a single
  parameter-free `solve_unlock_macro` cleared BOTH L2 and L3, i.e. once a leg is
  general enough, new levels of the same family are solved by REUSE, not by new
  code -- the player for a new level collapses to naming an existing leg.
- This is exactly the payoff the `unlock_search(env, reach_fn, schedule)`
  candidate predicts: every "unlock-then-reach" solver is the shared reach+macro
  frontier search (`_reach_bfs` + `_use_macros`) differing ONLY in the
  scheduling/pruning policy:
    * grow-pruned bounded DFS      -> `plan_unlock_reach`  (L1)
    * best-first on -|reach-set|   -> `plan_unlock_macro`  (L2, L3)
  Unifying them under one `unlock_search(env, reach_fn, schedule)` would make the
  unlock skill written literally ONCE, with each planner supplying only a
  `schedule` policy; paired with `plan_and_commit`, future unlock variants get
  their `solve_*` for free and every player stays a one-line composition. Kept
  as a candidate (not yet cut over) to preserve the current, validated behaviour.

## Level 4 (g50t)
Same maze family as L2/L3: the goal chamber only becomes movement-reachable
after chaining several `walk-somewhere-then-USE` openings (a USE resets the
avatar to start but PRESERVES gates opened en route). Ran the arena and observed
L4 clears with NO new leg -- the general `solve_unlock_macro` already covers it,
just as it did L3. The best-first USE-macro search stages the openings and wins;
the L1 `solve_unlock_reach` (single-USE-must-grow pruning) is still too weak here
for the same reason as L2/L3.

## Debrief refactor (post-clear of L4)
- Compared `play_level_4` to the earlier players. FINDING: L4 introduced no new
  skill -- `play_level_2`, `play_level_3` and `play_level_4` are now the SAME
  thin call, `solve_unlock_macro(env)` (three byte-identical player bodies). The
  shared skill is written ONCE in legs.py; the players are pure, thin
  composition that only NAME which general leg to commit.
- There was NO new duplication to extract into legs.py. The recurring code that
  the earlier debriefs flagged -- `clone_after` (fresh-clone-then-replay),
  `_reach_bfs` (movement flood, one clone per edge, dedup on position),
  `_use_macros` (walk-to-reachable-tile-then-USE successor expansion), and the
  `plan_and_commit` wrapper -- was already each written once. L4 reuses all of
  them unchanged, which is exactly the payoff of having factored them earlier.
- The only change made was cosmetic + behaviour-preserving: gave `play_level_4`
  the same explanatory comment style as its L2/L3 siblings so the "identical
  thin composition, deliberate reuse" intent is explicit in the source. Captured
  each level's committed action list before and after; all four plans are
  byte-identical (L1 len 19, L2 len 31, L3 len 64, L4 len 31), and
  `python gkm_try.py` still reports RESULT levels=4.

## Candidate higher-order leg (reaffirmed by L4): unlock_search(env, reach_fn, schedule)
- RECURRING COMPOSITION PATTERN, now with FOUR levels of evidence. The meta-
  observation strengthens: a single parameter-free `solve_unlock_macro` now
  clears L2, L3 AND L4. Once a leg is general enough, each new level of the same
  family is solved by REUSE, not by new code -- the player for a new level
  collapses to naming an existing leg. This is the "one general leg, whole level
  family" pattern, and it is the strongest signal that the underlying skill has
  been captured at the right level of abstraction.
- The candidate remains the same as flagged at L2/L3: every "unlock-then-reach"
  solver is one shared FRONTIER SEARCH over gate-configurations,
    1. at each node, `reach_fn` floods movement; if the reward is
       movement-reachable, return the winning path (`_reach_bfs`);
    2. else the successors are exactly the `_use_macros`
       (walk-to-a-reachable-tile-then-USE), each opening some gates -> a new
       gate-config child,
  and the ONLY thing that varies is the SCHEDULER/PRUNER over that frontier:
    * grow-pruned bounded DFS      -> `plan_unlock_reach`  (L1)
    * best-first on -|reach-set|   -> `plan_unlock_macro`  (L2, L3, L4)
  So `unlock_search(env, reach_fn, schedule)` would parameterise the shared
  reach+macro machinery by a `schedule` policy and BOTH planners fall out as
  instances; paired with `plan_and_commit`, each `solve_*` stays free and every
  player stays a one-line composition.
- STILL KEPT AS A CANDIDATE, NOT CUT OVER. Rationale unchanged and now
  reinforced by L4: the two schedulers differ not just in a pluggable ordering
  key but in control-flow-visible ways (DFS is grow-pruned, order-sensitive, and
  discovers reward at node-entry with no early per-child rp2 return; best-first
  is seen-set-deduped, PQ-ordered, and DOES early-return a child's reward). A
  faithful `unlock_search` skeleton would have to push all of that policy into
  `schedule`, so the abstraction would be LARGER than the two current planners
  while risking a change to the L1 plan (DFS child-ordering is what selects
  which winning macro is returned first). Since the mandate is to NOT change
  behaviour and all four validated plans are byte-identical today, the honest
  call -- consistent with the L1/L2/L3 debriefs -- is to document
  `unlock_search` as the recognised higher-order leg and defer the cut-over
  until a future level actually needs a THIRD scheduling policy (at which point
  writing the skeleton once pays for itself instead of adding abstraction for
  its own sake).

## Level 5 (g50t)
Level 5 keeps the same movement-flood / walk-then-USE substrate, but some useful
commits expose their hidden effect only at a later special-colored surface. Its
`solve_frontier_unlock` leg therefore restricts candidates to those visible
surfaces, greedily commits the candidate with the largest reach growth, and
allows a bounded number of world-changing stalls to carry the hidden state
forward. `play_level_5` remains a thin one-line composition of that leg.

## Debrief refactor (post-clear of L5)
- Compared with the earlier players, `play_level_5` selects a different planner
  but has the same thin shape: one call to a committing `solve_*` leg. No player
  logic needed moving.
- The L5 planner had repeated one substrate skill already present in the older
  unlock planners: walk to a reachable position, press USE, clone the resulting
  state, and flood movement to observe reward/reachability. Extracted that exact
  transition as `_use_macro(node, pos, path, reach_fn)`, written ONCE. Both the
  earlier `_use_macros` expansion and `plan_frontier_unlock` now call it while
  retaining their original candidate order, pruning, and scoring.

## Candidate higher-order leg from L5: choose_frontier_commit
- RECURRING COMPOSITION PATTERN. All three unlock strategies now share the same
  pipeline: flood movement, expand `_use_macro` transitions, classify/score the
  outcomes, then select or schedule a next commit. They differ only in frontier
  policy: immediate-growth DFS (L1), reach-set best-first search (L2--L4), or
  special-surface greedy growth with bounded structural stalls (L5).
- A candidate higher-order leg is
  `choose_frontier_commit(node, reach, candidate_fn, score_fn, accept_fn)`: the
  candidate policy chooses reachable positions, `_use_macro` evaluates each
  transition once, and the score/accept policies choose the next commit. This
  would capture the recurring "generate outcomes, rank, commit" composition.
  It remains a documented candidate because forcing the DFS, priority queue,
  and greedy staged loop through one controller would risk changing validated
  ordering and stopping behavior; the shared transition is the safe common
  skill to extract now.
