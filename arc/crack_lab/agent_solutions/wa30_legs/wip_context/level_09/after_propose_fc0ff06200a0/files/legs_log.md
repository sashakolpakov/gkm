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

## L9 (wa30)

Two socket clusters split by a wall with a row-8/9 gap: a west 8-ring around a
2-core and an east quad; two couriers (12) seat boxes autonomously; a stealer
(15) is confined to a bottom comb maze but surfaces at the corridor mouth
(10,3); a colour-7 bar caps the level at ~70 steps, so pursuit or full avatar
delivery cannot fit.

* Same neutralise-then-deliver shape as L6-L8, but the tight budget forces an
  AMBUSH instead of a chase: new minimal leg **`ambush_and_clear(env, cell,
  face_action, mover)`** parks on the maze-mouth cell, idles with safe USEs,
  and only presses the facing action when the mover occupies the faced cell.
* Delivery is `deliver_pairs` handing three east boxes to the east courier
  drop cells + one west box onto the ring, then `yield_until_level_up` while
  the couriers finish inside the budget.
* Checkpoint L1-L8 path re-based onto the per-level seg files (529 moves) to
  fit the 600-move real cap; L9 player adds 61 moves -> 590 total.
