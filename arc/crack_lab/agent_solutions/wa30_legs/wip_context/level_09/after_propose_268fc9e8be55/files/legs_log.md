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

## wa30 L9 debrief -- single-container fill, and the two hard limits

L9 is the same **fill-the-goal-container** end-game shape as L6/L7/L8: a
3x3 (12x12px) colour-9-framed container with a colour-2 interior, fed by ONE
active helper courier (colour 12); a second courier is permanently penned
behind a solid colour-2 band + the colour-9... err colour-5 mid-wall and never
helps.  Exactly 9 loose boxes for the 9 slots.  A self-mover (colour 15)
serpentines the lower maze (it can be frozen by parking the avatar at the
col-3 maze mouth, or killed by cornering it, but neither is free).

New reusable leg: **`fill_goal_container(env, cells, mover=None)`** -- greedy
nearest-box -> nearest-open-cell rigid carry, try-next-on-fail, cede a turn to
the courier when nothing is deliverable, and (optionally) USE-clear a parked
adjacent mover.  This is the general single-container specialisation of the
shared deliver phase; `play_level_9` is a one-line composition of it.

Two hard limits were measured (raw-frame probes + width-900 clone beam):

1. **Fill ceiling = 8 of 9.**  The lone courier is relay-based (it seats a box
   at the west entry then *re-grabs* it to push it deeper, transiently -- and
   sometimes permanently -- UN-filling cells, including the avatar's).  Cell
   (5,5) is courier-only (the avatar can reach every other cell).  Pure courier
   ~4-6; avatar adds ~2-3 but competes; the 70-step colour-7 bar terminates the
   level.  Every search (greedy/min-cost/lookahead/random/beam w<=900, multiple
   heuristics) plateaus at 8 filled cells (interior colour-2 = 9px); the win
   needs all 9, so no fill-win exists inside the timer.  A depth-40 clone BFS
   found no short alternative win either.

2. **Global move cap.**  `run_program` enforces `step_cap=600` REAL env.step
   calls total (clones are free).  The validated L1-8 path is 588 moves, so the
   resumed solver has only 12 real moves for L9 -- far below its 70-step bar.

`play_level_9` therefore reaches the discovered maximum (8/9) when given budget
but cannot complete the container; L9 as configured appears to sit past the
solvable edge under both the per-level colour-7 timer and the 600-move cap.
