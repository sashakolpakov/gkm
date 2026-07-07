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

## Level 2

ls20 level 2 also fell to the existing generic leg: `solve_by_search(env)`
found and replayed a winning action sequence with no game-specific code
added anywhere. No new leg mechanics (hidden switches, etc.) needed to be
special-cased for this level within the BFS horizon.

## Debrief: play_level_2 vs. earlier players

`play_level_2` is byte-for-byte identical to `play_level_1` -- both are
exactly the single line `solve_by_search(env)`. There is no repeated
*logic* to extract here: the only thing shared between the two players is
already a call to one existing leg, not duplicated inline code, so
`players.py` remains a set of thin one-line compositions and no refactor
of `legs.py` was needed this round (same conclusion as the level 1
debrief).

**Recurring composition pattern (candidate higher-order leg):** every
player defined so far -- level 1 and level 2 -- is the *identical* call
`solve_by_search(env)`, with zero per-level parameterization. That
suggests `solve.py`'s dispatch loop doesn't actually need a
`play_level_K` function to exist per level at all: it could fall back to
one default player (`solve_by_search`) whenever `players.py` has no
`play_level_K` override, and `players.py` would then only need to define
an override for a level that requires tuned `actions`/`max_states` or a
genuinely different (non-BFS) leg. Not implemented here -- the task is
scoped to "no behavior change," and defaulting the dispatch would change
`solve.py`'s control flow (currently: missing player -> stop). Worth
promoting to a real default once a level *does* need an override, so we
can see the override shape before generalizing the fallback.

## Level 3

`solve_by_search`'s plain `find_winning_path` (dedup on raw frame bytes)
stalled on level 3 with no result even at 8000 states / ~5 minutes -- far
worse than levels 1-2. Cause: level 3's frame carries a HUD/counter region
(rows 61-62, ~96 cells, overlapping what first looked like a maze feature)
that changes on its own every step regardless of the action taken. With
that noise included in the hash, no two frames ever compare equal, so BFS
dedup never fires and the frontier explores a near-tree-shaped blow-up
instead of a bounded graph.

Fix: added a new leg, `detect_noise_mask(env)` -- probes each action once
from the current state, picks the one with the smallest immediate pixel
diff (likely a blocked/no-op move), then repeats just that action for ~120
ticks on a clone; any cell that changes anyway is noise. `find_winning_path`
and `solve_by_search` now take an optional `mask` param and blank those
cells out before hashing (default `None` is a no-op, so levels 1-2's
already-validated behavior is unchanged). With the mask applied,
`solve_by_search(env, max_states=40000, mask=detect_noise_mask(env))` found
level 3's winning path (39 moves) in ~39s.

Lesson: before trusting "BFS explored N states with no hit," check whether
the frame has a self-ticking region -- a per-level `detect_noise_mask` probe
is cheap (four clone steps + ~120 more) relative to the cost of a search
that can never dedupe. Reach for `detect_noise_mask` + masked
`solve_by_search` first on any future level where plain `solve_by_search`
fails to find a path within a reasonable state budget, before writing
anything level-specific.
