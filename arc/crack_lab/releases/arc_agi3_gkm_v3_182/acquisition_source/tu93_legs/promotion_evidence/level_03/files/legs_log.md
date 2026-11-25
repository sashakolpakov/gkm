# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## Level 3

Reproduced from the supervisor checkpoint via `probe_level3.py`:
color 9 moves north under action 1, while actions 2/3/4 are blocked at
entry. Three color-8 components contain 24 pixels total.
`probe_reuse_level3.py` confirms on a clone that `clear_linked_sentries`
reduces that count to zero without completing the level; then
`traverse_linked_tiles` raises levels_completed from 2 to 3.
Reuse the level-2 composition without adding a new leg.

Committed harness verification: `RESULT levels=3 moves=47 replay_ok=True err=None`.
