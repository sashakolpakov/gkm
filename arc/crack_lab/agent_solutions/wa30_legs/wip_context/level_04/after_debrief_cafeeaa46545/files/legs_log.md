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
