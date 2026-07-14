# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## Level 1
ls20 level 1 is a sliding-block maze: a 5x5 tile (colors 12/9) slides exactly
one tile-length per action (1=up,2=down,3=left,4=right) along a color-3
"track" network; a move only succeeds if the whole destination 5x5 is track.
The win condition (a color-5 "room" entrance that looks blocked from one
route) actually opens only after the block has passed over a specific other
track cell first (a hidden switch/key), so naive shortest-path-by-position
search can miss it -- reachability isn't just a function of block position.
`find_winning_path` (BFS deduped on the FULL raw frame, not on any
hand-picked object position) sidesteps this correctly and found the level-1
solution (13 moves) in ~2000 explored frames / ~17s. Lesson for future
levels: prefer full-frame-hash dedup over guessing which object's position
is "the state" -- hidden switches/keys are easy to miss otherwise.
solve_by_search(env) (BFS+replay) solved level 1 with zero level-specific
code in players.py; keep reaching for it first on new levels before writing
anything bespoke, and only fall back to hand-written legs if the state space
is too large for a bounded BFS.

## Debrief: play_level_1 vs. earlier players

`players.py` in this workspace has only one player, `play_level_1`, so
there is no intra-file duplication to extract -- it is already a single
call, `solve_by_search(env)`, i.e. already thin composition. The `legs.py`
behind it factors cleanly into three non-overlapping legs
(`find_winning_path`, `execute_path`, `solve_by_search`) with no repeated
logic between them, so no refactor was needed this round.

Diffed against a sibling leg workspace for a cross-game pattern instead
(wa30, which has ~3 levels of players already):
- wa30's `move_and_use(env, action, steps)`: run a fixed action N times,
  then press one hardcoded follow-up action (`use`). Direction-specific
  wrappers (`ascend_and_use`, `descend_and_use`, ...) are all instances of
  this one shape with `action` bound.
- wa30's `follow_plan`/`relay_box_from_west`: run a literal scripted
  sequence of (action, count) steps -- no search, no branching.
- this workspace's `solve_by_search`: propose a *searched* sequence (BFS on
  clones) and only commit it to the real env if a success predicate
  (`levels_completed` increases) holds, rather than a fixed script.

**Recurring shape across both games:** every leg above is "produce a
sequence of actions, then commit it" -- they differ only in how the
sequence is produced (hardcoded script vs. BFS search) and whether commit
is unconditional or gated on a predicate. Candidate higher-order leg:
`propose_then_commit(env, propose, is_win=None)`, where `propose(env)`
returns one action sequence (a literal list for scripted legs, or a
search's result for BFS-style legs) and `is_win(env_after)` optionally
gates whether to keep it (default: always commit, recovering scripted-leg
behavior). Not extracted yet -- doing so would mean touching wa30's
`legs.py` too for a pattern only 2 instances deep across games; worth
revisiting if a third game/workspace repeats this shape.
