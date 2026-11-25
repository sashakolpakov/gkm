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

## tn36 level 3

Reused `clear_binary_program` without changing the leg. Clear eight segments:
(34,33), (34,48), (39,36), (44,36), (49,36), (54,36), (59,33),
(59,48); submit at (57,58). The resulting six column words, reading
segments at y=33,36,39,42,45,48, are 011110, 101111, 101111,
101111, 101111, 011110. Mismatched segments are the dense progress count.

`probe_level3_verified.py` reproduces the segment words from a pristine
level entry. Its clone remains at levels=2 after submission, even though
the identical real sequence succeeds; clone reward predictions are not
reliable evidence here. Do not infer a movement instruction mapping from
those clone outputs. The completed program was validated by the real
harness: RESULT levels=3 moves=25 replay_ok=True err=None.

The initial six-word direction guess and the 18 two-bit instruction guess
failed. Rapid player rewrites also exposed Python bytecode timestamp/size
cache reuse; the bounded candidate sweep varied source sizes to ensure
that each candidate was actually loaded. No such sweep is used by the
final player. The only final level logic is an existing-leg composition.

## tn36 level 4

Reused `clear_binary_program`; no new leg was needed. In right-column order
x=34,39,44,49,54,59, the successful words are 011011, 101110,
001111, 001111, 001111, 001111, reading y=33,36,39,42,45,48.
All segments initially show 1. Clearing the twelve selected segments and
submitting at (57,58) completes the level. Remaining mismatched segments
provide dense progress; the player only composes the existing leg.

Reproduction: `probe_copy_patterns.py` reads the four selector demonstrations
on clones as 101110, 111011, 001111, 011011. Repeating any one demonstration
six times fails from a fresh real entry. `probe_route.py` tests combinations:
UUUURG and UUUUGR fail; GRUUUU succeeds, where G, R, U denote the words
011011, 101110, 001111. These letters are probe mnemonics; physical meanings
and intermediate clone positions are not established evidence. The clone
of the successful sequence does not advance, while the real sequence does.
The successful full action path is saved separately in
`candidate_up_right_grow.json`; the supervisor checkpoint was not modified.

Final verification, both resumed and with GKM_FRESH_REPLAY=1:
RESULT levels=4 moves=38 replay_ok=True err=None.
