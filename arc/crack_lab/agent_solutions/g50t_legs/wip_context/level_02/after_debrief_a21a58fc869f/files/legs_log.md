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
