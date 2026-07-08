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

## wa30 L9 debrief -- split-world sockets + track-exit ambush

L9 is again neutralise-then-deliver, but the world is split by a wall: each
half has its own courier (12) and sockets, and the colour-15 stealer laps a
bottom maze, only emerging at the fixed exit cell (9,3) around turn ~40 to
raid the left ring.  Win = all 9 boxes seated; budget (colour-7 bar) ~70.

New leg: **`ambush_mover(env, cell, face_action)`** -- park ON a patroller's
track-exit chokepoint; the same-speed mover (unchaseable on its lap) queues in
the adjacent track cell, where a facing USE clears it.  Complements
`chase_and_clear` (corners a roamer) and `clear_frozen_mover` (stalled
hauler): this one exploits track topology instead of behaviour.

Everything else reused as-is: `deliver_pairs`/`carry_box_to` seat the two
avatar-reachable right sockets AND do the courier handoff (dropping a box on
the shelf row the courier serves -- delivery-to-an-agent needs no new verb),
then help the left courier with its farthest box, then `yield_until_level_up`.
Also rebased checkpoint.json onto the per-level seg files (529 moves for
L1-L8 vs 588) to fit the 600 real-move cap with L9's 61 moves.
