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

## tn36 level 2

Reused `clear_binary_program`; no new leg was needed. The left selector at
(21,58) displays a downward arrow and, in each of four columns, the six
segment states [off,on,on,on,on,off]. `probe_sequence.py` reproduces that
observation on clones. Copying that pattern to the four right columns clears
the eight segments at x=39,44,49,54 and y=33,48, then submits at (46,58).
`probe_copy_demo.py` reproduces the successful fresh-entry sequence and saves
`candidate_copy_down_program.json` separately from the supervisor checkpoint.

Clone success predictions are unreliable here: `probe_level1.py` fails to
advance a clone with the known level 1 sequence, while
`probe_level1_real.py` advances the real environment with the same sequence.
Use clone observations of segment toggles cautiously; validate full paths.

Final harness verification: `python gkm_try.py` reports
`RESULT levels=2 moves=16 replay_ok=True err=None`.
