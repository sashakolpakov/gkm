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

## Level 4

Reproduced the checkpoint entry at levels_completed=3 with 47 real actions.
`probe_level4.py` tests all four advertised actions on independent clones:
only action 4 moves color 9 at entry, from (40,21) to (40,27), while
color 12 moves from (22,39) to (28,39). Blocked actions leave both in place.
`probe_mechanics.py` confirms that three eastward moves are fatal, then
tests the existing composition on a separate pristine clone.
`clear_linked_sentries` removes color 8 while preserving color 9 and
leaving levels_completed=3; `traverse_linked_tiles` then raises it to 4.
No new leg is needed: reuse the level-2/3 composition.

Committed harness verification: `RESULT levels=4 moves=64 replay_ok=True err=None`.
