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

## wa30 L9 investigation (session notes)

Structure (16x16 cell view): vertical wall at cell-col 9 splitting the board;
gap at cell rows 8-9 lets the avatar cross. Bottom rows 10-15 are a comb maze.
Agents: two couriers (colour 12; left one roams+fills the big container, right
one activates only when a box is dropped in its region), one self-mover
(colour 15) that starts in the maze and reaches the top ~turn 45, then trails
the left courier and steals the box it carries. Avatar = colour 14, actions
1..5 = UP/DOWN/LEFT/RIGHT/USE (1 cell/step). Boxes = 4-frame/9-core (loose
border 4; the most-recently avatar-placed one blinks border 0<->3 = a mere
selection highlight, NOT a "seated" marker). Goal-looking 9-frame/2-core
containers: comp2 (big 3x3-cell, left), comp1 & comp7 (2x2-cell, right).

Level has a per-level budget of exactly 70 engine-moves; the bottom colour-7
bar is that COUNTDOWN TIMER (depletes ~1px/turn regardless of any agent -- I
proved this by trapping the mover, the bar kept draining). Bar-full != win
(reached bar7=0, no level-up).

Win condition NOT found. Ruled out empirically (all lc stayed 8):
 - bar fully painted to colour 4;
 - all 9 boxes placed on goal cells (reached 0 off-goal at ~mv64);
 - comp7 completely full (4/4);  comp1 3/4;  comp2 max 6/9;
 - neutralising agents: USE does NOT remove couriers(12) NOR the mover(15)
   (chase_and_clear left colour counts unchanged), unlike L6/L7/L8.
comp2 is the only container whose capacity (9) matches the 9 boxes, but filling
it to 9 is infeasible in 70 moves: the courier seats only ~5 (left boxes) even
undisturbed, and relaying each of the 3 stranded right boxes across the wall
costs ~24 moves (>70 total).

BUDGET BLOCKER: run_program caps TOTAL real moves at 600. Players 1-8 already
consume 588 (L1=27,L2=54,L3=80,L4=69,L5=124,L6=47,L7=65,L8=122), leaving only
12 for L9 under gkm_try. Reaching L9 would require BOTH cracking the (still
unknown) L9 win AND reclaiming ~58 moves from L1-8 (fattest: L5,L8).
