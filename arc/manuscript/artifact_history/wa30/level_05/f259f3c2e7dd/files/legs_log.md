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
