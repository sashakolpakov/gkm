# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## tn36 level 1

Added `clear_binary_program`: sample independent binary controls, click lit
segments, then execute. Dense progress is the remaining lit-segment count.
The ten controls are at x=21,26,31,36,41, y=42 and45. Initial lit segments
are x=26,36,41 on both rows. Clearing these six and clicking (36,55) wins.
Verified by `python gkm_try.py`: levels=1, moves=7, replay_ok=True.
The player only composes the reusable leg.

Clone branches gave misleading intermediate pawn positions. Those observations
were discarded as planning evidence; the final seven-click path was reproduced
on a fresh real entry and independently replay-validated. The earlier five-click
candidate failed and must not be adopted.
