# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## Level 2: linked colour rows

Reproduced from pristine entry with `probe_level2.py`: actions are (5,6,7).
Selecting a bottom swatch then clicking a socket places its colour in either row;
7 undoes the latest placement. No moving avatar was observed.
Legend: [12,15,8,9,14,11,6]. Socket columns: [22,28,34,40], rows 22 and 36.
Upper (34,22) is a colour-14 reference to the lower colour-14-bordered row.
Filling sockets in reading order and submitting with 5 did not increase reward.
Expanding that reference recursively gives positions
[(22,22),(28,22),(22,36),(28,36),(34,36),(40,36),(40,22)];
matching the legend to those positions and submitting increased levels from 1 to 2.

Added `match_nested_colour_rows`, which derives the recursive order and delegates
to `match_colour_row` with an optional socket order. Existing placement checks
provide dense progress (one more matching socket per placement), followed by a
clone reward check before committing. Player 2 only composes that leg.
Resumed and fresh harness runs both reported levels=2, moves=24, replay_ok=True.
Full fresh candidate path: nested_rows_level2_candidate.json. Checkpoint untouched.
