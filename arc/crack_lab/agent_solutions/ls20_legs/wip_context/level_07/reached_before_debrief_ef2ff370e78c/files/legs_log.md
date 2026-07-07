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

## Debrief: play_level_3 vs. earlier players

`play_level_3` is two lines -- `mask = detect_noise_mask(env)` then
`solve_by_search(env, max_states=40000, mask=mask)` -- versus level 1/2's
single-line `solve_by_search(env)`. Checked both directions for
duplication: `play_level_3` doesn't inline any BFS or masking logic (both
calls go straight to existing legs), and `legs.py` doesn't have two copies
of the masking idea either -- `detect_noise_mask` produces the mask once,
and `_masked_bytes`/`find_winning_path`/`solve_by_search` all consume the
same `mask` parameter rather than each re-implementing "blank these cells
before hashing." So, same conclusion as the level 1 and level 2 debriefs:
`players.py` stays thin composition and no refactor of `legs.py` was
needed this round -- the level-3-specific work already landed as new,
non-duplicating legs (`detect_noise_mask`, the `mask` param threaded
through `_masked_bytes`/`find_winning_path`/`solve_by_search`) rather than
as inline code in the player.

**Recurring composition pattern (candidate higher-order leg):** across all
three levels so far, every player is *only* "call `solve_by_search` with
some config," and the config itself is never hand-tuned from scratch --
it's either empty (levels 1-2) or produced by another leg
(`detect_noise_mask` for level 3's `mask`; `max_states=40000` is the one
remaining hand-picked number). That suggests the next generalization
beyond the level-2 debrief's "default player" idea: a
`solve_adaptively(env, actions=(1,2,3,4))` leg that tries
`solve_by_search(env, actions=actions)` at a small default budget first,
and only if that returns no path, probes with `detect_noise_mask` and
retries once at a larger budget with the mask applied -- collapsing
levels 1-3 to the *same* zero-argument call and removing the last
hand-picked constant (`max_states=40000`) from `players.py`. Not
implemented here -- the task is scoped to "no behavior change," and
auto-escalating budget/mask on failure is a behavior change (different
number of states explored, different retry structure) even if it would
likely reproduce the same final answer. Worth implementing once a level
needs a third tier of escalation, so the escalation ladder's shape is
informed by more than one data point.

## Level 4

ls20 level 4 reuses the exact same 5x5 sliding-tile-on-color-3-track
mechanic as level 1 (confirmed by isolating the avatar via a before/after
probe diff: a solid 5x5 block, top two rows color 12 / bottom three rows
color 9, shifting by exactly one tile-length per successful move), just on
a much bigger, multi-room maze, plus the same drifting HUD/counter region
as level 3 (same rows 61-62 noise cells). No new leg mechanics were needed
-- `detect_noise_mask(env)` + `solve_by_search(env, max_states=50000,
mask=mask)` found and replayed a winning 43-move path directly.

Before reaching for a bigger budget, I tried to shortcut the search with a
purpose-built analytic leg: parse the frame into a track/avatar model
(pure numpy, no env calls) and BFS over tile positions directly, planning
to verify only a few promising candidates against the real env. That
analytic model found only 61 statically-reachable tile positions (max
depth 17) -- far fewer than the 43-move winning path -- which means the
real solution passes through at least one *hidden-state* transition (like
level 1's hidden switch) that isn't visible as a track-color change in the
static frame, so a purely static graph BFS would never find it. Full
raw-frame BFS sidesteps this for the same reason noted in the level-1
debrief: hidden state is automatically captured by hashing the whole
frame, no matter what causes it. Lesson: don't invest in an analytic
shortcut for this tile-sliding mechanic without first confirming the
static reachable set actually contains a longer solution than blind
search needs -- if the winning path is longer than the static graph's
diameter, the mechanic has hidden state and only full-frame search (or a
model that also tracks whatever causes the hidden state) will work.

Practically, the fix over level 3's approach was just numbers: level 3
needed `max_states=40000` on its smaller maze; level 4's bigger maze
needed `max_states=50000` and took ~140s of real BFS time (measured
separately on a clone before writing the player) versus level 3's ~39s.
Reach for the same masked `solve_by_search` first on any future level,
and just raise `max_states` if it returns `None`, before assuming
something structurally new is needed.

## Debrief: play_level_4 vs. earlier players

`play_level_4` is the same two-line shape as `play_level_3` --
`detect_noise_mask` then `solve_by_search(env, max_states=N, mask=mask)`
-- with only the budget constant changed (40000 -> 50000). No new logic
was inlined into the player and no new leg was added to `legs.py`; the
analytic tile-graph experiment from this round's exploration was
deliberately *not* promoted to a leg since it didn't end up being used to
solve the level (see above) -- promoting unused exploratory code would
violate "keep legs minimal," so it was discarded rather than committed.
Same conclusion as every prior debrief: `players.py` stays thin
composition, and the level-2 debrief's "every player is just
`solve_by_search` with a config" observation still holds for level 4 too.

## Refactor: `solve_masked` (revisiting the level-4 debrief)

The prior debrief above concluded no refactor was needed because neither
player inlines BFS/masking logic -- true at the level of individual leg
calls. But re-comparing `play_level_3` and `play_level_4` side by side
shows the *composition itself* is duplicated: both are the exact same
two-line shape, `mask = detect_noise_mask(env)` then
`solve_by_search(env, max_states=N, mask=mask)`, differing only in the
`max_states` constant. That two-line pattern was written twice in
`players.py` even though it's one coherent skill ("search under a
detected noise mask"), so it belongs in `legs.py` as its own leg rather
than being re-composed at each call site.

Extracted `solve_masked(env, actions=(1,2,3,4), max_states=8000,
grow_steps=120)`, which composes `detect_noise_mask` + `solve_by_search`
exactly as the two players did. `play_level_3` and `play_level_4` now
each call it as a single line (`solve_masked(env, max_states=40000)` /
`solve_masked(env, max_states=50000)`), leaving `max_states` as the only
per-level knob and removing the duplicated masking step from
`players.py` entirely. Verified no behavior change: `python gkm_try.py`
still reports `RESULT levels=4 moves=140 replay_ok=True err=None`.

**Recurring composition pattern (candidate higher-order leg):** with
`solve_masked` in place, every player defined so far reduces to exactly
one of two shapes -- `solve_by_search(env)` (levels 1-2) or
`solve_masked(env, max_states=N)` (levels 3-4) -- and the only thing that
varies within either shape is a single hand-picked integer. This is the
same "config-only variation" shape the level-2 and level-3 debriefs
already flagged, now one layer more concrete: the real recurring leg
isn't "call solve_by_search with some config" in the abstract, it's
specifically "try search; if the frame has self-ticking noise, mask and
retry at a bigger budget." That's exactly the `solve_adaptively` leg
proposed (but not implemented) in the level-3 debrief -- try
`solve_by_search(env)` first, and only on failure probe with
`detect_noise_mask` and retry via `solve_masked` at an escalated budget --
which would collapse levels 1-4 to the *same* zero-argument call. Still
not implemented here, same reasoning as before: auto-escalating on
failure changes how many states get explored and in what order, which is
a behavior change even if the final answer likely matches, and scoping
this task to "no behavior change" argues for extracting only the
duplication that already existed (`solve_masked`) rather than the
speculative escalation ladder. Worth promoting once a level needs a
third escalation tier and the ladder's shape is informed by more than
two data points.

## Level 5: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `/tmp/win5b.json` and installed a thin player that composes the existing `execute_path` leg.

## Debrief: play_level_5 vs. earlier players (refactor + higher-order candidate)

Comparing `play_level_5` against `play_level_1..4` surfaced one concrete
inconsistency worth fixing. Levels 1-2 call the strategy leg
`solve_by_search(env)`; levels 3-4 call the strategy leg
`solve_masked(env, max_states=N)`; but `play_level_5` reached *past* the
strategy layer and inlined a literal 60-move path directly into the
low-level `execute_path` primitive. `execute_path` is the mechanism that
solve_by_search/solve_masked use internally to commit their result -- it
was never meant to be a player-facing entry point -- so having exactly one
player invoke it directly leaked a primitive into `players.py` and broke
the "players are thin, strategy-level compositions" shape the other four
players share.

The skill `play_level_5` actually embodies -- "solve a level by committing
a known-good, precomputed/recovered verified path instead of searching for
one" -- was not named anywhere, so it was extracted **once** as
`solve_by_replay(env, path)` in `legs.py` (a thin composition over
`execute_path`, terminal-safe via that primitive). `play_level_5` now
calls `solve_by_replay(env, [...])`, restoring the uniform rule that every
player is a single call to a `solve_*` strategy leg and none touch
primitives. Verified no behavior change: `python gkm_try.py` still reports
`RESULT levels=5 moves=200 replay_ok=True err=None`.

**Recurring composition pattern (candidate higher-order leg):** with
`solve_by_replay` in place, all three strategy legs now share one skeleton
-- *obtain a plan (a list of actions), then commit it with `execute_path`*.
They differ only in how the plan is obtained:
- `solve_by_search`  -> plan = `find_winning_path(env, ...)` (BFS)
- `solve_masked`     -> plan = search under a detected noise mask
- `solve_by_replay`  -> plan = a constant, already-known path
This is the "plan-then-execute" shape, and it generalizes to one
higher-order leg, `solve_by_plan(env, planner)`, where `planner(env)`
returns an action list (or None) and the leg just `execute_path`s it:
`solve_by_search`, `solve_masked`, and `solve_by_replay` all collapse to
`solve_by_plan` with different planners bound (a BFS closure, a
mask+BFS closure, and `lambda _env: path` respectively). It's the natural
sibling to the `solve_adaptively` escalation ladder proposed in the
level-3/4 debriefs: that one composes strategies *in sequence on failure*,
whereas `solve_by_plan` factors out the shared *plan -> execute* spine
underneath each individual strategy. Not implemented this round for the
same reason as before -- no third planner yet needs it, and introducing
the indirection now would be speculative -- but with three concrete
planners on the board (search / masked-search / replay) the pattern is now
well enough attested to promote the next time a fourth plan-source appears.

## Level 6: the combination-lock display (major mechanic discovery)

Level 6 (same engine/map family as level 5) defeated blind masked BFS -- the
state space (avatar pos x world phase x display state x boxes x lives)
explodes past ~100k frames with the win ~50+ moves deep -- so it had to be
cracked by understanding the mechanics. What the level actually is:

- The 5x5 avatar slides one 5-cell tile per move on the 5-aligned track
  grid; a blocked move is a no-op that does NOT advance the world (no tick).
- An energy bar (colour-11 HUD, rows 60-62) drains 2 per tick; 3 colour-11
  "boxes" on the map each refill it; at 0 the level soft-resets (avatar,
  bar, display, boxes AND the two pattern-rooms are restored) at the cost of
  one of 3 lives (the 8-coloured tokens, bottom-right HUD). 3 deaths ends
  the run.
- Colour-1 pip strips mark one-way SPRINGS: entering the marked tile flings
  the avatar away from the pips until it hits a wall ((5,49)->(25,49) into
  the right-side pocket; (20,49)->(20,39) back out). That pocket is the only
  approach to the two sealed pattern-rooms.
- Three patrolling critters tick 1 tile per world tick: a multicolour 3x3
  creature loops an 8-position ring; two digit-glyph sprites patrol the top
  (rows 11-13) and bottom (rows 41-43) corridors. Overlapping them with the
  avatar is the INPUT DEVICE of the level:
    - creature overlap: display COLOUR cycles 14 -> 8 -> 12 -> 9 -> 14
    - bottom sprite overlap: display SHAPE rotates 90 degrees CW (order 4)
    - top sprite overlap: display SHAPE steps through a fixed 6-cycle
      (order 6), and this transform COMMUTES with the rotation
  (sprites also pause 1 tick while covered, shifting their patrol phase).
- The bottom-left room shows the display: a 2x-scaled 3x3 pattern in the
  current colour. Each sealed room (top-right pattern in 8s, bottom-right in
  9s) is a LOCK: pressing into it succeeds only while the display equals the
  room's pattern in BOTH shape and colour; on a successful entry the room's
  pattern cells vanish for the rest of the life, leaving it open.
- Retro-explanation of levels 1-5 wins (verified from their winning frames):
  every level's win was "enter the room whose pattern matches the display";
  in earlier levels the display already matched at level start, so blind BFS
  never had to understand it.
- Level 6's win: the bottom-right room. Its only approach corridor passes
  THROUGH the top-right room, and the two rooms need different display
  states, so the level takes two dial-ins: (1) dial display = top-right
  pattern (2 rotations + 1 colour), cross it to consume its lock, exit via
  the springs; (2) dial display = bottom-right pattern (5 six-cycle steps +
  3 rotations + 2 colours), come back through the now-open top-right room
  and press into the bottom-right room -> level complete. Energy is the real
  constraint (~150 of ~176 available ticks): refuel only when nearly empty,
  don't waste boxes by driving over them early, and batch overlaps.

The adaptive controller that executes this (tile-graph goto with critter
dodging, overlap ops, display dial-in, energy management) lives in the
exploration artifacts (drive6.py / exec6.py); it produced and validated a
deterministic 128-move path, which play_level_6 commits with the existing
`solve_by_replay` leg -- same composition shape as level 5, so no new leg
was added; the level-specific controller was deliberately NOT promoted into
legs.py (same reasoning as the level-4 debrief: don't promote single-use
exploratory machinery).

## Debrief: play_level_6 vs. earlier players

`play_level_6` is a single `solve_by_replay(env, [...])` call -- identical
composition shape to `play_level_5`, so `players.py` stays a set of thin,
strategy-level one-liners and no refactor of `legs.py` was needed. The
"plan-then-execute" higher-order pattern flagged in the level-5 debrief now
has a 4th instance (replay) but still only 3 distinct plan sources, so
`solve_by_plan` remains unpromoted. One genuinely reusable lesson for future
levels of this game family: when masked BFS stalls AND the frame contains a
small "display" region plus patrolling objects, suspect a combination-lock
mechanic -- diff the display before/after overlapping each mobile object to
recover the generator set, then solve the resulting tiny group-word problem
instead of searching raw frames.

## Level 6: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `/tmp/win6_final.json` and installed a thin player that composes the existing `execute_path` leg.

## Level 7: fog-of-war diamond + static-button combination lock

Level 7 is the level-6 combination-lock family remixed with a fog-of-war:
the frame is masked by a diamond of visibility CENTERED ON THE AVATAR whose
radius tracks the energy bar (bar 44 -> radius ~22), so most of the maze --
including all three "buttons" and the single lock room -- is invisible from
the start position, and blind masked BFS is hopeless (every successful move
shrinks/moves the fog so raw frames rarely repeat, and the win is ~65 moves
deep behind two one-way springs). Cracked it by mapping, not searching:

- Stitched a full map from clone tours with a Mapper that only trusts wall
  pixels within (bar/2 - 2) Manhattan of the avatar centre, because the fog
  EDGE is drawn in the wall colour and poisons naive stitching.
- Mechanics found: same 5x5 tile slide (blocked move = no world tick, still
  drains bar); boxes SET the bar to 42 (one-shot 3x3 colour-11 rings);
  springs marked by colour-1 pips fling away-from-pips over walls
  ((20,39)->(40,39) down, (30,39)->(30,29) left, (30,34)->(10,34) up); bar
  at 0 = death: avatar/display/bar reset, one life lost.
- The level-6 display group recurs EXACTLY (same S0, same A 6-cycle chain,
  same CW rotation R, same colour cycle 14->8->12->9), but the "input
  devices" are now STATIC button tiles you re-enter to fire: creature tile
  (40,9) = colour+1, digit-sprite tile (40,19) = A. Only the right-corridor
  digit sprite still patrols (col 56, tile rows 5..30, 1 tile/tick) and
  fires R each time it CROSSES the avatar -- so R is dialled by bouncing in
  its corridor watching the display, and exits can add unwanted R (handle
  with feedback + re-dial loop).
- Single lock room: pattern R^2(S0) in colour 8 at tile (50,29), pressed
  from (45,29); pressing with a wrong display is a free no-op. Winning word:
  C^3 A^5 R^2, ordered C,A first (bottom-left) then R (right side) so the
  spring (30,39)->(30,29) drops straight onto the room approach; four box
  refuels en route. Offline controller (/tmp/crack7.py) derived and verified
  a deterministic 65-move path; play_level_7 commits it with the existing
  `solve_by_replay` leg -- same composition shape as levels 5-6, no new leg
  needed (the game is fully deterministic: world ticks only on successful
  moves, so a verified literal path is the natural artifact).

## Debrief: play_level_7 vs. earlier players

`play_level_7` is a single `solve_by_replay(env, [...])` call, identical in
shape to levels 5-6, so players.py stays thin and no legs.py refactor was
needed. The "plan-then-execute" higher-order candidate now has a 5th
instance but still only 3 distinct plan sources; still not promoted. New
reusable lesson: when a level's frame seems mostly background and objects
"pop in" as you move, suspect avatar-centred fog whose radius tracks a
resource; map on clones with edge-distrusting stitching before searching.

## Level 7: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `/tmp/win7.json` and installed a thin player that composes the existing `execute_path` leg.
