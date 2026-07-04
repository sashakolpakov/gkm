# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## ls20 (LS20) -- core mechanic
The avatar carries a (shape, colour, rotation) state. Transform tiles cycle one
component when stepped on; a TARGET tile requires a specific combo AND the avatar
to stand exactly on it (it blocks movement until matched). A level clears when
all targets are satisfied. Directions: 1=up 2=down 3=left 4=right, 5=noop. A
step-counter UI drains every move, so raw frames never repeat -> useless as a
search key; the compact game state (avatar pos + shape/colour/rot + done-mask) is
the right dedup key.

## Legs
- `state_key` / `plan_to_next_level` / `run_plan` / `advance_one_level`
  BFS over avatar states on clones to the shortest path that advances one level,
  then commit it. General: it self-discovers which transform tiles to visit.

## Level 1
`advance_one_level(env)` alone. Solves in 13 moves, replay-validated. (One target,
one rotation tile: rotation off by one, shape+colour already matched.)

## Level 2
Object-moving / carry level (NOT a transform level). Big colour-4 maze; avatar
can only go up/left/right (no down); action 2 is the interaction (attach/release
a carried sprite). The win condition depends on where SPRITES end up, so the
avatar-only `state_key` collapses distinct worlds and BFS wrongly reports the
goal unreachable (whole reachable space = 81 states, no solution). Adding every
sprite's (name,x,y,rot) to the dedup key (`full_state_key`) makes the true space
~576 states; BFS then finds a 45-move solution. `play_level_2` =
`advance_one_level(env, key_fn=full_state_key)`. 45 moves (58 total), replay-ok.

Refactor done (real reuse, not churn): threaded a `key_fn` param through
`plan_to_next_level` / `advance_one_level` (default = `state_key`) and added the
general `full_state_key` leg. Both players share ONE generic BFS; the only
per-level knob is which dedup key captures "what matters" for the goal.

## Debrief: candidate higher-order leg (updated after level 2)
Two players now exist and the pattern is confirmed across both:

    play_level_1(env) -> advance_one_level(env)
    play_level_2(env) -> advance_one_level(env, key_fn=full_state_key)

Every `play_level_K` is a SINGLE `advance_one_level` call whose only per-level knob
is which dedup `key_fn` captures "what matters" for that level's goal. There is no
cross-player code duplication to hoist -- each player is already a one-line thin
composition, and the shared skill (`advance_one_level` over the generic BFS) is
written exactly once in `legs.py`. Nothing was refactored this round: a refactor
would be pure churn with no shared code left to extract.

Recurring composition pattern (candidate higher-order leg): a per-level player =
`advance_one_level(env, key_fn=<per-level choice>)`, dispatched by level number.
The natural promotion is `advance_levels(env, n=None, key_fn=...)` -- repeatedly
call `advance_one_level` until `n` levels clear (or the game terminates / no plan
is found), returning the count advanced. STILL PREMATURE, and for a concrete
reason now visible with two data points: the multi-level loop lives in `solve.py`,
and each player itself advances exactly ONE level -- no player repeats a loop. A
blind `advance_levels(env, key_fn=K)` would also be wrong because the correct key
VARIES per level (`state_key` for L1, `full_state_key` for L2), so it needs a
per-level key selector, not a single fixed key. Promote only once a single player
must clear several sub-levels, or a data-driven `{level: key_fn}` dispatch table
actually recurs; until then `advance_one_level` + `solve.py`'s dispatch loop is the
right altitude.


## Level 3
Same family as level 1 (avatar-configuration level), NOT a carry level despite 97
sprites (nearly all are static maze/wall cells). `play_level_3` =
`advance_one_level(env)` with the default avatar-only `state_key`. BFS finds a
41-move clearing path in ~7s / <8000 clone-steps. `full_state_key` also solves it
(39 moves) but costs ~10x more search (67s) for no benefit, so the cheap key wins.
Key insight reconfirmed: a BFS path that raises levels_completed is
simulator-verified, so the dedup key only risks MISSING solutions, never inventing
false ones -- prefer the cheapest key that still finds a path. 99 moves total,
replay-ok. No new legs needed; pure reuse.

## Debrief: candidate higher-order leg (updated after level 3)
Three players now exist and the pattern holds cleanly across all of them:

    play_level_1(env) -> advance_one_level(env)
    play_level_2(env) -> advance_one_level(env, key_fn=full_state_key)
    play_level_3(env) -> advance_one_level(env)

New this round: `play_level_1` and `play_level_3` have BYTE-IDENTICAL bodies
(`advance_one_level(env)` with the default `state_key`). That looks like repeated
code but is NOT duplication to hoist -- it is two call-sites of the ONE shared leg,
which is exactly the thin-composition target. The BFS skill is still written once
in `legs.py`; the players carry only per-level intent (comments) plus the single
knob that varies: which dedup `key_fn` captures "what matters" for the goal
(avatar-only for configuration levels 1/3, full-sprite for carry level 2). Nothing
was refactored -- there is no shared implementation left to extract, so a refactor
would be churn.

Recurring composition pattern (candidate higher-order leg): a per-level player =
`advance_one_level(env, key_fn=<per-level choice>)`, dispatched by level number in
`solve.py`. With three data points the key selector is clearly a 2-valued function
of the level FAMILY (configuration vs carry), not of the level number -- levels 1
and 3 share a key precisely because they share a family. The natural promotion is
therefore NOT a fixed-key `advance_levels(env, key_fn=K)` but a data-driven
`advance_by_table(env, {family_or_level: key_fn})` (or a `pick_key(env)` leg that
sniffs whether the goal depends on sprite positions and picks the cheapest
sufficient key automatically). STILL PREMATURE: each player advances exactly one
level, the multi-level loop already lives in `solve.py`, and a hand-written
per-level `key_fn` is cheaper than a heuristic sniffer while the table has only two
distinct values. Promote once a single player must clear several sub-levels, OR the
`{level: key_fn}` mapping grows enough that repeating it at call-sites recurs.
Until then, `advance_one_level` + `solve.py`'s dispatch loop is the right altitude.
