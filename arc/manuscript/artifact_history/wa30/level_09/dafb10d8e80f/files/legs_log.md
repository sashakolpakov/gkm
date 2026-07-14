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

## Level 9: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `/private/tmp/claude-501/-Users-sasha-gkm/e3e00be1-d1a5-4095-a6ef-4d720f42d84e/scratchpad/gkm_legs_ws_wa30_fable5_l9_wip/base2.json+/private/tmp/claude-501/-Users-sasha-gkm/e3e00be1-d1a5-4095-a6ef-4d720f42d84e/scratchpad/gkm_legs_ws_wa30_fable5_l9_wip/seg_L9.json` and installed a thin player that composes the existing `execute_path` leg.

## wa30 L9 debrief -- the ferry-batch shape (`ferry_each`)

The previous L9 entry installed a raw 61-action `execute_path` replay.
Decoding that verified path against the earlier players shows it is NOT an
opaque blob: it splits exactly into five repetitions of the ferry skill the
library already had in straight-line form --

    approach (walk to a box) -> USE (grab) -> carry (walk rigidly) -> USE
    (release) -> optional depart step clear

which is `grab_push_release` / `ferry_box` from L3/L4, except that on L9 both
the approach and the carry turn corners.  Refactor (behaviour-identical,
checked action-by-action against seg_L9.json):

* **`grab_carry_release(env, approach, carry, depart=())`** -- the one
  scripted box-ferry skill, now written ONCE with plan-step lists for every
  phase.  `grab_push_release` (and hence `ferry_box`, `relay_box_from_west`)
  became thin straight-line bindings of it.
* **`ferry_each(env, specs)`** -- the batch combinator: for each box spec,
  `grab_carry_release`.  `play_level_9` is now a five-line spec table.

**Recurring composition pattern / candidate higher-order leg:** almost every
box level is `ferry_each` plus an optional epilogue:

| Level | composition                                              |
|-------|----------------------------------------------------------|
| L3    | per-box approach+relay (ferry_each shape) + final `use`  |
| L4    | `ferry_all_then_yield` = ferry batch + yield-for-helper  |
| L5    | live-frame ferry batch + yield (`ferry_all_to_courier_then_yield`) |
| L9    | `ferry_each` alone -- no helper, the avatar seats boxes itself |

So the candidate higher-order leg is
**`ferry_batch_then(env, specs, epilogue=None)`**: run the per-box
grab-carry-release batch, then an arbitrary epilogue (`yield_until_level_up`
for courier levels, a switch-`use` for L3, nothing for L9).
`ferry_all_then_yield` is that combinator with `epilogue=yield`; L9 is it
with `epilogue=None`.  If a third epilogue flavour appears, fold the three
`ferry_*` entry points into this one combinator.

Bookkeeping: the resume checkpoint now carries the shorter validated L1-8
base (base2.json, 535 moves) so the 61-move L9 player fits the 600
real-move cap: full run = 596 moves, RESULT levels=9 replay_ok=True.
