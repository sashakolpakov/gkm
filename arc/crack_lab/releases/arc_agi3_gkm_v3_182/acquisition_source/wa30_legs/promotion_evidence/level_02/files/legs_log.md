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
