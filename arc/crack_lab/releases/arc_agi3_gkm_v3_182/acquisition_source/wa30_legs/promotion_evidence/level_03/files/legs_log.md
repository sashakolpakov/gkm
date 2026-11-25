# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## Level 1: carry pieces onto marked cells
- Reproduced by `probe.py`: actions 1/2/3/4 move the color-14 avatar up/down/left/right by four pixels; 5 alone has no visible effect. Other pieces remain stationary.
- `probe_mechanic.py`: from entry, `[1,1,5]` picks up the piece above the avatar. Each directional move translates both cells while preserving their relative offset; 5 releases it. Approaching highlights a piece with color 3.
- `probe_goal.py` and `probe_win.py`: filling one or two of the three red cells gives no reward. Releasing the third piece onto the final cell increments levels_completed to 1. A held piece hides the red marker too, so release is required.
- `candidate_three_objects.json` preserves the independently clone-verified 27-action discovery path.
- `probe_leg.py` reproduced the first planner failure from pristine entry: it selected an already delivered piece, uncovering its original marker. The leg now preserves initial marker locations and excludes delivered pieces from pickup candidates.
- Final harness: `RESULT levels=1 moves=28 replay_ok=True err=None`.
- `fill_marked_cells` plans paired-cell translations and clone-verifies that every delivery decreases the remaining marker count before committing.

## Level 2: cooperate with an autonomous carrier
- `probe_level2.py` reproduces the color-14 avatar's four movement actions and the independent color-12 carrier advancing on all five actions.
- `probe_mechanics.py` and `probe_contexts.py` reproduce the carrier collecting, transporting (color 5), and releasing yellow pieces onto marks. Five pieces share a six-cell marked region.
- The original delivery planner fails clone verification because the carrier changes source and destination occupancy during the route.
- `probe_cooperate.py` reproduces a winning clone: retrieve the distant piece, deliver it, leave the delivery area, then yield while the carrier delivers the others. Completion increments levels_completed even with one marked cell unused.
- Reused `fill_marked_cells` with an optional cooperative mode: test bounded delivery-and-yield alternatives on clones and commit only a complete winning path. Clone probes are throttled below 300 steps/s.
- `play_level_2` only composes that existing leg.
- Final harness: `RESULT levels=2 moves=96 replay_ok=True err=None`.

## Level 3: stage pieces across a divider
- `probe_l3.py` reproduces the entry after the supervisor's 96-action prefix: avatar at (9,4), autonomous carrier at (3,12), and five pieces. Coordinates are four-pixel tile (row,column).
- All five cloned actions advance the carrier; actions 1–4 move the avatar in the four directions. Repeated rightward movement stops the avatar at column 7 against the column-8 divider.
- Waiting 31 actions leaves reward at 2 while the carrier moves the two divider pieces into the goal region at (6,13) and (8,13). The three pieces on the left remain inaccessible to it.
- Side pickup `[1,1,1,1,4,5]` from entry followed by `[4,4,4,5]` stages the piece at (5,8), with the avatar at (5,7). Moving down and yielding lets the carrier collect it and deliver it to (6,14); reward remains 2 with two pieces still on the left.
- Added `handoff_across_barrier`: infer the divider, preserve pickup offsets in routing, stage each reachable piece, leave pickup range, and verify collection. Exclude goal pieces from further manipulation and clone-verify complete reward before committing.
- Existing `fill_marked_cells` cannot express this handoff: its above-avatar cargo cannot reach across a full-height divider, and its red-mark detector includes divider cells. Earlier players remain unchanged.
- `play_level_3` only composes the new leg. `candidate_barrier_handoffs.json` preserves the validated full campaign path.
- Final harness: `RESULT levels=3 moves=187 replay_ok=True err=None`. Independent probe replay: `PROBE_RESULT 3 187 True None`.
