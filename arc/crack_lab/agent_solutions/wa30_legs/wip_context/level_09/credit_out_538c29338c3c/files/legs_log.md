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

## wa30 L9 investigation (UNSOLVED within budget) -- discovered structure

L9 is the same neutralise-then-deliver family (9-frame socket container +
colour-12 courier filler + a colour-15 same-speed stealer), but tuned to be
near-infeasible for the available legs.  Discovered entirely by arena probes:

Geometry (16x16 cell view):
* Avatar = colour 14 (4x4 block, colour-0 facing marker).  Moves 1 cell/step.
* Big goal container = 3x3 9-frame at rows3-5,cols5-7.  Its CENTRE (4,6) is a
  solid colour-2 block that is NOT a seat (same as L8's 3x3 -- L8 won with the
  centre still 2).  So the goal is the EIGHT EDGE seats.
* Two small 1x2 containers on the right: small2 (6,13),(6,14) is avatar-
  reachable; small1 (2,13),(2,14) sits in a top-right pocket sealed by a
  colour-2 "diagonal wall" at row3,cols10-15 -- NO boxes there and neither the
  avatar nor either courier can enter, so small1 is unfillable (decoy).
* A vertical colour-5 wall at col9 spans rows0-7 only; rows8-9 are open, so the
  three EAST boxes (5,11),(7,12),(8,14) are reachable only via a row8-9 detour.
* 9 loose boxes total (6 left of col9, 3 right).  Boxes do NOT regenerate.

Mechanics:
* USE = grab-adjacent-box / release (rigid carry), as elsewhere.  No push, no
  auto-slide into seats.  A carry costs ~15-26 real steps (east ones ~22-26 due
  to the wall detour).
* The lone roaming courier fills only the 5 WEST/centre-column edge seats
  (never the centre, never the east column, never small2) from the 6 left
  boxes, at ~1 seat / 8-12 steps -> tops out at ~5 in the level's lifetime.
  The second courier (1,11) is sealed in the top-right pocket and never acts.
* The stealer (15) patrols the bottom maze while the avatar is elsewhere and
  only reaches the goal region ~t40+, un-seating edge boxes it is adjacent to.
  It NEVER stalls, so USE cannot clear it while it moves (L7's freeze trick is
  inapplicable).  `chase_and_clear` DOES remove it permanently but takes ~40+
  steps.  It freezes (harmlessly) whenever the avatar sits in the col-3 maze
  corridor (it path-targets the avatar and is maze-separated there) -- but that
  pins the avatar and it ignores a box placed in the gap.
* The level TIMER is colour 7 (bottom bar): starts 64, drains ~1/step, and (on
  L9) seat-fills give only +1, so the usable budget is ~64 steps -- far harsher
  than L8's ~148.

Why it resists the shared leg: winning needs all 8 edge seats.  The courier
supplies ~5 west; the avatar must transport+seat the THREE east edges from the
three wall-stranded east boxes (~3x~20 = ~60 avatar-steps) AND neutralise the
stealer (~10+ steps) inside ~64 steps, with theft starting ~t46.  Across ~15
tested strategies (courier-only with mover frozen, avatar east-first + park,
cheap-first + chase-clear, `fill_dual_sockets`, `clear_frozen_mover_then_fill`,
hand-optimised carries) the fill ceiling was ~5-7; 8 was never reached.  The
optimal budget is essentially equal to the timer, so it needs carries ~2x
faster (~10-12 steps) or a second active filler -- neither available.

`play_level_9` therefore routes through `clear_stealers_then_fill_dual`
(top=the 8 big edges, bot=small2, one band clearing the stealer): the correct
family shape, but it does not clear the timer.  Genuine progress would need a
new *fast* carry leg (grab from the box side nearest both avatar and drop, one
BFS, no per-step re-clone) and/or exploiting a mechanic not yet found.

(Separately: the harness checkpoint replays the 588-move L1-8 path, leaving
only ~12 of the 600 real-move cap for L9, so even a working L9 would also need
the L1-8 path trimmed by ~55 moves.)
