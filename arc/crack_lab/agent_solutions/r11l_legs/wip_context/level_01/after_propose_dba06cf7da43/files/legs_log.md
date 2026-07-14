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
