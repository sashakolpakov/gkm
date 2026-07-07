# Leg-library debrief log

Recurring composition patterns and repeated novelty.

Candidate higher-order leg: "reposition -> ascend_and_use -> reposition -> use".
Level 1 uses this as mirrored branch service on the left and right, suggesting a
parameterized branch-delivery leg over horizontal directions and distances.

Candidate higher-order leg: "travel to a lane -> directional sweep -> use".
Level 2 repeats this as alternating corridor traversals that end in activation,
which suggests a parameterized sweep-and-activate leg over direction, distance,
and optional vertical setup.

## wa30 level 3 (current clean rerun; replay-validated)

Level 3 extends the same activate/carry world with a helper on the far side of a
vertical barrier. The current clean solver reaches L3 by composing only simple
movement/use legs:

- Move to each left-side box from the west.
- Press `USE` to attach it.
- Carry it east to the wall/relay boundary with `relay_box_from_west`.
- Release it where the helper can take over.
- Step away where needed, then idle with `use(env, 33)` so the helper finishes.

The important continuity for later levels is not the exact path but the reusable
pattern: the player cannot personally finish every delivery, so it must place
objects into a helper-reachable handoff zone and then yield turns to the helper.
Level 4 should look first for the same "handoff into helper reach, then yield"
shape before starting broad search.

## wa30 level 4 (replay-validated)

Structure: the avatar is fenced inside a central arena of six boxes. Three courier
agents (colour 12) patrol outside; boxes that are released near/into the fence
"ripen" (border 4->5) and a courier fetches each one and places it in a socket
(9-border/2-core) slot; slot count equals box count (one outer box is handled by
the couriers automatically). A depleting bar on row 63 is a hard step budget, so
no idle waiting mid-plan and no USE-spam while yielding (USE re-grabs a nearby
box and blocks ripening).

New legs (kept minimal and general):
- `grab_push_release(face, push, steps)` -- bump-face an adjacent box, USE to
  attach, push/carry it, USE to release. This is the core box-handling verb; the
  whole level is seven calls of it plus `follow_plan` repositioning.
- `yield_until_level_up(idle_action)` -- yield turns with a harmless move until
  the level counter increments (generalizes level 3's blind `use(env, 33)` wait).

Mechanics worth remembering: carried boxes stay on the grab side and can turn
corners; a box leading into the fence embeds one cell into it; couriers walk
through the fence band and can fetch a released box sitting just inside it; the
top fence next to the tall interior wall was a dead drop zone -- prefer bottom /
left / right handoffs.

## Level 4: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json` and installed a thin player that composes the existing `execute_path` leg.

## Level 4 debrief: refactor into shared legs

The recovered raw path for level 4 decoded cleanly into vocabulary the earlier
players already used -- no new primitive skill was needed:

- Seven repetitions of "approach (follow_plan) -> grab_push_release", then a
  single "yield turns until the level counter increments" tail.
- Level 3's `relay_box_from_west` turned out to be the same grab/carry/release
  skill minus the facing bump, so `grab_push_release` gained `face_steps=0` and
  optional `depart_action`, and the relay leg now delegates to it. The skill is
  written exactly once.
- `play_level_4` is now thin composition: seven `ferry_box(...)` calls plus
  `yield_until_level_up(env, DOWN)`; it reproduces the verified action sequence
  byte-for-byte (69 actions, level-up on the 15th yielded DOWN).

Candidate higher-order leg (recurring composition pattern): "ferry a batch,
then yield" -- i.e. `for approach, face, push, n in boxes: ferry_box(...)`
followed by `yield_until_level_up(idle_action)`. Levels 3 and 4 are both
instances: reposition to each movable object, grab it, carry/push it into a
helper-reachable handoff zone, release, repeat for every object, then cede
turns so the helper agents finish the level. A parameterized
`ferry_all_then_yield(env, boxes, idle_action)` over a list of
(approach_plan, face, push, push_steps) specs would make future courier/helper
levels a single data-driven call. `ferry_box` (approach + grab_push_release)
was added this pass as the first step toward that leg.

## wa30 level 5 (replay-validated; reaches LEVEL 5)

Same courier-handoff world as L4, one level harder. Discovered by clone probing:

- Avatar is a 4x4 mover: colour-14 body + colour-0 head (head marks facing).
  Every action moves it exactly one 4-cell; USE grabs the box it faces
  (border 4->3 when adjacent-highlighted, ->0 when grabbed) and the box then
  translates rigidly with the avatar. The world lives on a 16x16 grid of 4x4
  cells.
- A vertical colour-5 wall (cell col 9) splits the map; a corridor gap at cell
  rows 7-8 joins the right arena to a big colour-9/2 socket on the left.
- ONE courier (colour 12) patrols the corridor, auto-fetches boxes from the
  right and seats them in the socket. Six boxes; the win fires only when all
  six are seated (socket coverage never reaches 0 -- 6 four-wide boxes can't
  tile the offset 2-core -- so the goal is "all boxes delivered", not "socket
  full").
- Row-63 colour-7 bar is a hard step budget (~128 steps; ~0.5 cell/step).

Key insight: a single courier round-trip (~16-22 steps) x6 barely fits the
budget, and feeding ALL boxes to one handoff cell CONGESTS it (later ferries
fail because the cell is still occupied). Rotating between TWO handoff cells
(7,10) and (8,10) keeps the pickup point clear so the courier seats one per
pass while the avatar keeps supplying -- all six land inside budget and the
level completes.

New legs (general "courier handoff" skill, grid-of-4 perception from frames):
- `_cells` / `_bfs_path` / `_bfs_pair` / `_nav_to` / `_carry_pair` -- coarse
  cell perception + obstacle-aware pathing (avatar alone, and avatar+carried
  box as a rigid pair), replanned each step because the courier moves.
- `grab_and_deliver(env, box, drop)` -- bump-face + USE grab, then rigidly
  carry the box onto a target cell and USE release. The core box verb.
- `ferry_all_to_courier_then_yield(env, drops=...)` -- self-pacing loop: while
  boxes remain, ferry the nearest free box to the nearest EMPTY handoff cell
  (rotating over `drops` to avoid congestion); if all drops are busy, cede a
  turn so the courier clears one; finally park clear of the boxes and yield
  until the level counter increments. This realises the debrief's predicted
  higher-order leg "ferry a batch, then yield" as a live-frame controller.

`play_level_5` is a single call to `ferry_all_to_courier_then_yield`.

## wa30 level 5 debrief: refactor into shared legs

Comparing `play_level_5` against the earlier players confirms one dominant
composition shape that now recurs across *every* courier/helper level (3, 4, 5):

> **ferry a batch of boxes into a helper-reachable handoff zone, then yield turns
> for the helper to finish.**

- `play_level_3`: repositions to each west box, `relay_box_from_west`, then an
  idle wait (`use(env, 33)`).
- `play_level_4`: seven `ferry_box(...)` calls followed by `yield_until_level_up`.
- `play_level_5`: a single `ferry_all_to_courier_then_yield(...)` -- a live-frame
  self-pacing loop that ferries the nearest free box to a rotating handoff cell
  and finally parks and yields.

The repeated *code* was in `play_level_4`, which open-coded the batch loop and
the trailing yield inline. That loop-and-yield is now written ONCE as the leg
`ferry_all_then_yield(env, specs, idle_action)` (a `for spec in specs:
ferry_box(...)` batch followed by `yield_until_level_up`). `play_level_4` is now
thin, data-driven composition -- a list of `(approach, face, push, steps)`
tuples handed to that one leg -- and reproduces the verified action sequence
byte-for-byte (RESULT unchanged: `levels=5 replay_ok=True`).

Candidate higher-order leg (now realised twice): **"ferry-batch-then-yield"** has
two sibling forms sharing the same skeleton:
- STATIC / pre-planned: `ferry_all_then_yield(specs, idle)` -- deterministic when
  box positions and courier reach are known ahead of time (level 4).
- DYNAMIC / live-frame: `ferry_all_to_courier_then_yield(drops=...)` -- re-reads
  the frame each pass, picks the nearest free box and an empty rotating handoff
  cell, and cedes turns when congested (level 5).

The natural unification for future courier levels is a single
`ferry_all_then_yield(env, plan, idle_action)` whose `plan` may be either a
static list of ferry specs OR a live-frame policy (nearest-box -> rotating-drop),
with the same "grab -> carry into helper reach -> release, repeat, then yield"
core. Both current legs are instances of that shape; the static one is the new
shared leg extracted this pass.

## wa30 level 6 debrief: refactor into shared legs

Comparing `play_level_6` against the earlier players, the player itself is
already thin -- one line: `clear_agent_then_deliver(approach, [(box, drop), ...])`
= a scripted `follow_plan` prep + `use` (thread the wall gap to the parked
self-mover and clear it), then a data-driven batch of grab-carry-release
deliveries. The real duplication was NOT in the players but INSIDE the leg
library: `play_level_6` had spawned a whole second copy of the level-5
perception/pathing machinery, so several core skills were written twice.

De-duplicated this pass (each skill now written ONCE):

- **Grid BFS shortest-path.** `_grid_bfs` was byte-for-byte identical to
  `_bfs_path` (only an `act`/`ac` rename). Deleted `_grid_bfs`; `_avatar_nav`
  now calls the single `_bfs_path`.
- **Avatar navigation loop.** `_nav_to` (courier world) and `_avatar_nav`
  (fenced-goal world) were the same "walk empty-handed avatar to goal,
  replanning every step" loop, differing only in which perception supplied the
  obstacles. Extracted `_walk_avatar_to(env, goal, read_obstacles, cap)` plus
  two tiny adapters (`_obstacles_courier` adds the moving courier; `_obstacles_grid`
  is walls+boxes). Both nav legs are now one-line wrappers.
- **Rigid-pair carry search.** `_rigid_carry` had re-inlined the exact
  ok()-constrained pair BFS that already existed as `_bfs_pair` (which
  `_carry_pair` uses). Routed `_rigid_carry` through the shared `_bfs_pair`,
  keeping the start-cell feasibility guard. The two carry LOOPS stay distinct
  because their world policies differ (courier = extra obstacle + retry a
  transient block; fenced-goal = a block means genuinely stuck), but the search
  itself is now written once.

The two intentionally-separate pieces that remain per-world are the two
perception readers (`_cells` vs `_grid_scan` classify a "box" cell with
different predicates -- `len(u)<=3` vs `2 not in u` -- and `_cells` also reads
the head and courier), and the two grab-and-deliver verbs (`grab_and_deliver`
= nearest reachable grab orientation, courier-aware; `carry_box_to` = try every
orientation with clone look-ahead to confirm a gap-threading carry is routable
before committing). These are different policies over the SAME shared
primitives, not duplicated skills.

Behaviour unchanged: `python gkm_try.py` still reports
`RESULT levels=6 moves=401 replay_ok=True`.

Candidate higher-order leg (the recurring composition pattern across ALL
players so far, now including L6): **"prep, then batch grab-carry-release, then
settle."** Every level is:

    1. an optional scripted PREP (`follow_plan` + `use`) that positions the
       avatar and/or arms the mechanism (L1 branch setup, L2 lane entry, L6
       gap-thread-and-clear-the-self-mover);
    2. a BATCH of the one core verb -- grab an object, move it (push, rigid
       carry, or relay), release it into a goal/handoff cell -- iterated over a
       list of objects; and
    3. a SETTLE tail -- either yield turns so a helper finishes
       (`yield_until_level_up`, `_park_and_wait`) or a final activating move.

`clear_agent_then_deliver` (L6) and `ferry_all_then_yield` (L4) /
`ferry_all_to_courier_then_yield` (L5) are all instances of this skeleton; they
differ only in the prep, in which grab-carry verb the batch uses, and in the
settle. The natural unification is a single higher-order leg

    deliver_batch(env, prep, verb, items, settle)

where `prep` and `settle` are small action plans (or callables) and `verb` is
whichever grab-carry-release skill fits the world (push / rigid-carry-static /
rigid-carry-live). Future levels should first try to express themselves as one
`deliver_batch(...)` call before reaching for search.

---

## Debrief after clearing wa30 L7

L7 gives an *uncatchable* self-mover (colour 15) that relentlessly hauls every
loose box to its own store; USE has no effect while it patrols. The trick is to
let it stall (it grabs a box it cannot seat), USE to clear the now-frozen mover,
then rigid-carry every box into the west goal container. `play_level_7` was one
line -- `clear_frozen_mover_then_fill(...)` -- but its leg had re-inlined the
"deliver boxes onto goal cells" loop that `clear_agent_then_deliver` (L6)
already expressed with `carry_box_to`. So, as with the L6 debrief, the real
duplication lived INSIDE the leg library, not in the (already thin) players.

De-duplicated this pass (each skill now written ONCE):

- **Deliver-to-goals verbs.** The two levels' delivery loops both rigid-carry
  boxes onto goal cells via `carry_box_to`; they differ only in how boxes are
  matched to slots. Extracted them as two tiny, single-definition verbs:
  `deliver_pairs(env, moves)` (explicit `(box, drop)` pairs -- L6) and
  `fill_targets_nearest_first(env, targets)` (nearest still-free box per slot
  -- L7, previously inlined in the leg).
- **Neutralise-then-deliver combinator.** The L6/L7 skeleton "disable a blocking
  autonomous agent, THEN deliver boxes into the goal container" is now the
  higher-order leg `neutralize_then_deliver(env, neutralize, deliver)`. Both
  `clear_agent_then_deliver` and `clear_frozen_mover_then_fill` are now thin
  compositions of it: they supply only their own `neutralize` half (thread-gap +
  USE vs idle-until-frozen + USE) and pick which delivery verb to hand it.

The two `neutralize` bodies stay per-level because the mechanic differs (a
*parked* mover is cleared by walking to it and USE-ing; an *uncatchable* mover
must first be stalled with `idle_until_mover_frozen` before the same USE works).
These are different policies over the SAME shared delivery primitives.

Behaviour unchanged: `python gkm_try.py` still reports
`RESULT levels=7 moves=466 replay_ok=True`.

Candidate higher-order leg (refining the earlier `deliver_batch` skeleton).
Across ALL players the shape is still **"PREP, then batch grab-carry-release,
then SETTLE."** L6 and L7 reveal that for autonomous-agent levels the PREP is
specifically **NEUTRALISE a blocking agent** and the SETTLE is empty (a filled
container is itself the win, so no yield tail is needed). Thus:

    neutralize_then_deliver(env, neutralize, deliver)
        == deliver_batch(env, prep=neutralize, verb=carry_box_to,
                         items=<goal cells>, settle=<none>)

is the specialisation of `deliver_batch` for "there is an autonomous blocker;
disable it, then just carry every box home." `neutralize` is the pluggable prep
(clear-parked vs idle-then-clear), and `deliver` is `deliver_pairs` or
`fill_targets_nearest_first` depending on whether the box->slot assignment is
known. Future autonomous-agent levels should first try to express themselves as
one `neutralize_then_deliver(...)` call, choosing a neutralise half and a
delivery verb, before reaching for search.

## wa30 level 8 investigation (extensive; win condition not cracked)

Level 8 is a two-region combined puzzle. The 16x16 cell world is split by two
wall bands (row6 with a gap at C4-5; row9 with a gap at C9-10) into a TOP region
(rows0-5), a narrow MIDDLE band (rows7-8, where the avatar starts at (8,1)), and
a BOTTOM region (rows10-15). Row63 is a color-7 countdown bar (~0.42/step, hard
episode timeout ~150 steps; the global run_program step_cap also leaves only ~134
real moves for level 8 after the 466-move L1-7 checkpoint).

Objects: avatar=14 (sokoban grabber: bump-face + USE grabs a 4-border/9-core box,
rigid-carry, USE releases). Each region has ONE self-mover (color 15, evasive
patroller) and ONE courier (color 12). Right side has two 9-framed containers
(TOP: cells C11-14,R2-3; BOTTOM: C12-14,R12-14) with color-2 interiors. Left side
has two color-2 regions (a top-left pen holding 4 penned boxes behind an L-fence,
and a bottom-left 3x3 solid block). 13 boxes total (4 penned + 4 top-free + 5
bottom-free).

Mechanics discovered by experiment:
- Helpers are DETERMINISTIC (identical under any avatar action) and move every
  turn. Self-movers evade orthogonal capture (stay diagonal), so a reactive chase
  never catches them.
- A self-mover CAN be removed by USE when it is orthogonally adjacent to the
  avatar; a bounded beam finds a ~20-move sequence that parks the avatar in a wall
  gap so the wandering mover steps adjacent and is USE-cleared (both movers
  removable this way). Couriers (12) CANNOT be removed by USE (they are permanent).
- With movers present, boxes are cycled in a dynamic equilibrium: couriers seat a
  few boxes into the right containers and movers dump a few at the left regions,
  but everything plateaus (~2 per side) and idle never completes.
- With BOTH movers removed, couriers steadily seat boxes into the right containers
  (fill rises), but plateau ~6-7 boxes/container; the corner penned box (2,1) is
  the only one couriers cannot reach.
- Avatar-released boxes inside a container flash color-3 for one frame then revert
  to color-4 (no persistent "avatar-seat"); couriers do the real seating.

Win condition NOT found despite exhaustive search. Ruled out (none trigger the
level counter):
- Filling right containers to their max (movers off, couriers + freeing penned
  boxes: reached 84/72 px ≈ 7/6 boxes, still parked out of the way to the crash).
- Beam maximizing color-4 across ALL four goal regions with movers ON reached 156
  px = all 13 boxes distributed into the regions (free-box count 0).
- Beam (movers off) reaching free-boxes=0 / fill=144.
- Beam maximizing left-region fill only (movers on) to 108.
- Removing both movers (agents 15 -> 0 px) alone; couriers un-removable.
- A staged union heuristic (remove movers THEN maximize both left+right fill) hit
  movers=0 AND fill=156 simultaneously — still no win.
- Exact-frame BFS to depth 11 (120k states) found no shallow win.

Conclusion: box placement in every reachable configuration, and full agent
neutralisation, both fail to increment levels_completed within the ~134/150-step
budget. The true objective remains opaque from raw frames; this matches the
recorded auto-solver failure for level 8 (auto_solve_attempts.json). play_level_8
therefore composes the furthest-progress behaviour (search-clear the evasive
self-movers, then yield for the couriers) via a new general leg, but does not yet
clear the level.

New general leg added this pass:
- `clear_evasive_agents(env, color, budget, beam_width)` — a bounded clone beam
  that drives the pixel-count of an evasive agent `color` to zero using avatar
  actions (positioning at chokepoints so the wandering agent steps adjacent, then
  USE). Generalises the L6/L7 "USE-clear a blocking agent" to agents that evade a
  reactive chase and must be caught with lookahead. Reusable for any level with a
  removable-by-USE evasive mover.

### L8 refinement: goal tiles found, but win still not triggered

Follow-up probing revealed a real signal: a box the AVATAR rigid-carries onto
certain colour-2 cells flashes colour-3 and PERSISTS there (a "locked" box),
whereas on other cells the colour-3 reverts to colour-4 within a frame. Mapping
every colour-2 cell (movers pre-cleared) yields ~14 such tiles, concentrated in
the LEFT structures: the bottom-left 3x3 block (8-9 tiles), the top-left pen's
C3 fence column ((2,3),(3,3),(4,3)), and the top container's right-edge interior
((2,13),(2,14),(3,14)); the bottom container yielded none.

Caveats that stopped this from closing the level:
- Persistence tracks courier-UNREACHABILITY as much as "goalness": tiles that
  couriers can walk to have their colour-3 box relocated (reverts to 4), so the
  colour-3 set is not a clean goal indicator.
- Only AVATAR placement locks (colour-3); movers/couriers dropping a box on the
  same tile leave it colour-4. Covering all ~14 tiles by avatar carries (~15-25
  moves each) is far beyond the ~134-move budget.
- With movers ON, helpers cover the LEFT goal tiles (coverage rose to 5-7/14 on
  idle) but cycle boxes off again (equilibrium); a beam maximising goal-tile
  coverage with movers active plateaued at 7/14. With movers OFF the couriers
  only fill the RIGHT containers, so left coverage stays ~0.
- No coverage level (7/14 movers-on, or right-container fill 84/72 movers-off)
  triggers the level counter.

So the mechanic is now understood (avatar locks boxes onto goal tiles), but the
completion threshold is not reachable within budget by any strategy tried, and
the helper/avatar division needed to cover ALL required tiles inside ~134 moves
was not found. play_level_8 is left as the clean reusable composition
(`clear_evasive_agents` + `yield_for_couriers`); reaching L8 remains open.
