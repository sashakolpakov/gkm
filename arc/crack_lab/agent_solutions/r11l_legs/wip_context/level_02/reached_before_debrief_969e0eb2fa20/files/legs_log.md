# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## r11l level 1
Sole action = coordinate click `step(6,x,y)` (x=col,y=row). Two draggable
endpoints; a BOX is always their midpoint. A click moves the ACTIVE endpoint;
clicking on the other endpoint's marker selects it. WIN = box lands on the
hollow-diamond RING centre.
New legs: `detect_rope` (finds active endpoint / 3-endpoint / ring),
`place_box_on_ring` (put both endpoints on ring row at Rc±d so midpoint=ring).
Composed by `play_level_1`. Solves in 3 clicks.

Refactor: the raw action `env.step(6,x,y)` is now written ONCE in the
primitive `click(env,row,col)`. Two semantic skills wrap it:
`move_active_to` (drop the active endpoint) and `select_endpoint` (activate a
marker). `place_box_on_ring` composes them; `play_level_1` stays a one-line
thin composition.

## Candidate higher-order leg (recurring pattern)
Across the r11l endpoint puzzles the same shape recurs:

    detect_state(env) -> handles + target
    move_active_to(A)              # place handle #1
    select_endpoint(other)         # hand off control
    move_active_to(B)              # place handle #2 so a DERIVED point == target

i.e. "drive two coupled handles so their derived feature (here the midpoint
BOX) lands on a target." A higher-order leg would capture this:

    place_derived_on_target(env, detect, solve_positions)
        active, other, target = detect(env)
        posA, posB = solve_positions(active, other, target)
        move_active_to(env, *posA)
        select_endpoint(env, other)
        move_active_to(env, *posB)

`place_box_on_ring` is the midpoint instance, with `solve_positions` =
"both on target row at columns Rc-d, Rc+d". Future levels (different derived
feature or constraint) reuse the skeleton by supplying a new
`solve_positions`.

## r11l level 2
Multiple colour-coded rope systems share ONE pick/drop cursor. A colour C owns
a system iff it has BOTH a filled box (a `6` wrapped in C) and a hollow-diamond
RING (isolated C pixels touching only background 5). The box always sits at the
ROUNDED CENTROID of that colour's endpoints (level-2 had a 3-endpoint system in
colour 12 and a 2-endpoint system in colour 15). WIN = every box centred on its
OWN-colour ring (verified: cross-colour placement gives NO reward; both boxes on
same-colour rings simultaneously fires the level reward).

Key mechanic discovered by probing: each drop is bounded by a ROPE LENGTH from
the current box, so a far box must be WALKED (move one endpoint toward the ring,
the box drifts, repeat) — a single long jump is rejected. A near box is placed
in one shot by a symmetric STRADDLE (endpoint offsets summing to zero => centroid
== ring). Two gotchas that cost time:
  * Selecting/placing must be validated by the ACTIVE (0-diamond) marker, but the
    winning drop COMPLETES the level and mutates the frame, so straddle verify must
    also accept "levels_completed increased" as success (else it rejects the very
    move that wins).
  * Placing one box on its ring changes the other rope's reachable set. Walking
    (which clusters endpoints tightly around the ring) keeps a system out of the
    others' way; scattering (large straddle offsets) can block a neighbour. Hence:
    walk-first (farthest gap first), then straddle-snap.

New legs: `box_center`, `active_pos`, `endpoints_of`, `ring_center`,
`ring_systems` (perception), `walk_box_to` (rope-walk a centroid onto a target),
`straddle_box_to` (exact symmetric snap, level-completion aware), and the
orchestrator `place_boxes_on_rings`. `play_level_2` is a one-line composition.
Reused nothing from level 1 directly (different derived feature: centroid of N
endpoints vs midpoint of 2), but the family matches the "drive coupled handles so
a derived point hits a target" pattern noted earlier.
