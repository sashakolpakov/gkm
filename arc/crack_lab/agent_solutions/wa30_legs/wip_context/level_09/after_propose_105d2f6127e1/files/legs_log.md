# Leg-library debrief log

(Notes intentionally left minimal.)

## wa30 L8 debrief -- the neutralise-then-deliver shape (now shared by L6/L7/L8)

Comparing `play_level_8` to the earlier players, every end-game level (L6, L7,
L8) is the SAME two-phase shape:

1. **NEUTRALISE** the autonomous agent that would otherwise keep the goal from
   staying filled (it steals boxes back out / relentlessly re-hauls them).
2. **DELIVER** every arena box onto the goal-container cells with the shared
   rigid `carry_box_to`; a filled container is the win.

Only the two halves vary per level:

| Level | neutralise                              | deliver                        |
|-------|-----------------------------------------|--------------------------------|
| L6    | thread the wall gap + USE (one parked mover) | `deliver_pairs` (known box->slot) |
| L7    | idle until the uncatchable hauler stalls, then USE | `fill_targets_nearest_first`      |
| L8    | `chase_and_clear` each band's same-speed stealer | `fill_dual_sockets` (two sockets + couriers) |

The recurring composition pattern -- the candidate higher-order leg -- is
therefore **`neutralize_then_deliver(env, neutralize, deliver)`**: it runs an
arbitrary neutralise phase then an arbitrary deliver phase. L6 and L7 already
routed through it; the L8 refactor now does too (its leg previously inlined the
clear-loop + fill call). The "clear the blocker, then fill the container" shape
is now written ONCE, and each `clear_*_then_*` leg is a thin binding of its two
level-specific halves onto this combinator, while the players stay one-line
compositions.

Next-step candidate: if a future level's neutralise phase itself must be
retried after a partial delivery (agent respawns), generalise to a
`while not delivered: neutralize; deliver` fixpoint variant of the same shape.

## wa30 L9 debrief -- seal-the-chokepoint + hand-over-the-band (70-move clock)

L9 is the neutralise-then-deliver shape AGAIN, but under a hard ~70-move level
clock, so the phases must be pipelined rather than sequenced lazily:

* Win = every one of the 9 boxes seated in a container.  Helpers do most of
  it: a west courier auto-fills the big 3x3 container; a parked NE courier
  serves the top-right container but is cut off from all boxes by an
  avatar-impassable diagonal band (colour-2 texture).
* NEW MECHANIC -- the box CAN sit on the band: rigid-carry a box one cell onto
  the band and release; the cut-off courier walks down, fetches and seats it.
  No new leg needed: `carry_box_to(env, box, band_cell)` already expresses it.
* NEW MECHANIC -- the maze stealer path-plans every turn and FREEZES the
  instant no route to its target exists.  Its bottom-maze region drains
  through the single cell (9,3); occupying that cell freezes it, and walking
  INTO the one-wide corridor keeps the route sealed, so the frozen mover can
  be approached and cleared with the standard facing USE at leisure.  This is
  the new leg `pin_maze_mover(env, seal_cell)` -- a timing-robust alternative
  to `chase_and_clear` (which stays as its fallback if the mover escapes).
* The courier alone is ~15 turns too slow for the six west boxes, so the
  avatar carries one west box itself (`deliver_pairs`) and then
  `yield_until_level_up` cedes the last turns to the courier.

Player is 4 lines: deliver_pairs (3 right boxes incl. the band handoff),
pin_maze_mover, deliver_pairs (west help), yield_until_level_up.

Harness note: run_program caps REAL moves at 600; L1-8 checkpoint replay was
588, leaving 12 -- so every earlier level was re-validated through a
deterministic per-level path shortener (levels restart from a canonical state,
so segments compose freely).  L1-8 now replay in 529 moves and L9's player
adds 64 (593 total).
