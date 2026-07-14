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
