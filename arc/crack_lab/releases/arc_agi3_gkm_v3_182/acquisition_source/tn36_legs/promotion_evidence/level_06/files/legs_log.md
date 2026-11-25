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

## tn36 level 5

Reused `clear_binary_program`; the player adds no inline level logic and the
shared leg is unchanged. In x order 34,39,44,49,54,59, set the six words to
001111,001111,001111,111011,000000,010111 (y=33,36,39,42,45,48),
then submit at (57,58). Clear fifteen initially lit segments. The remaining
selected lit segments are the dense progress count during program construction.

`probe_level5_verified.py` reproduces the pristine entry at reward 4, the
advertised coordinate action, the avatar/target components, the exact edited
words on a clone, and the successful live execution. It also independently
validates the saved path. The clone remains at reward 4 with the same program;
its success prediction is unreliable, as independently reproduced using the
known level 1 solution during this attempt. No movement or reward prediction
from a failing clone is needed by the final leg composition.

The target component is colour 15, bbox (20,47,29,58), area 64. Its bitmap is
the size-two version of the downward-facing goal outline seen at level 4.
The successful program was suggested by passing the narrow opening while
small, growing, changing colour, and turning. Only the complete six-word
program is relied upon as execution evidence; intermediate live poses are
not exposed after failed submissions.

Preserved independently validated path: `candidate_level5_verified_program.json`.
Final harness verification: RESULT levels=5 moves=54 replay_ok=True err=None.
The supervisor-owned checkpoint was not changed.

## tn36 level 6

Completed by four submissions through persistent intermediate checker tiles.
`play_level_6` only composes legs: reuse `clear_binary_program` for the initial
edit, then use the new minimal `write_binary_program` to restore or clear bits.
The earlier clearing leg cannot restore bits needed by later programs.

Columns are x=34,39,44,49,54,59; bit rows, least significant first, are
33,36,39,42,45,48. Submit at (57,58). The four verified word sequences are:

1. (61,61,61,61,61,1): pawn reaches board (row 6,column 5).
2. (62,30,30,1,1,1): pawn reaches (4,4).
3. (30,30,30,30,61,1): pawn reaches (0,5).
4. (62,62,62,60,62,1): reward increases from 5 to 6.

These correspond to RRRRR, LUU, UUUUR, LLLDL, with word 1 as filler.
`probe_stages_finish.py` reproduces the four real submissions and validates
its path independently. `probe_stages.py` reproduces the negative comparison:
ending the last program with word 56 instead of 1 leaves reward at 5.
`probe_live_selectors.py` reproduces the four reference movement words and
left-canvas displacements from a pristine level entry. `probe_insert.py`
reproduces word 1 at each slot among five right words, always reaching (6,5).

Dense progress: reach each checker tile in order, then the goal. Ordinary
wall-respecting maze distances to the goal are 12 at entry, then 9,6,5,0.
During each edit, the number of mismatched segments decreases to zero.
Intermediate tile positions are essential: a single submission does not
complete this route. No inference about changing pawn colour is needed by
the final solver.

Clone execution remains unreliable: `probe_compare.py` reproduces identical
real/clone frames through the known level-5 edits, followed by divergent
reward at submission. The final route relies on real observations and
independent replay validation, rather than clone success predictions.
The direction-only and zero-word sweeps were exploratory; their failures
are not relied on as exhaustive claims about reachable campaign states.

Preserved candidate: `candidate_level6_verified_staged_program.json` (101
coordinate actions, independently validated by `probe_stages_finish.py`).
Supervisor checkpoint untouched. Both `python gkm_try.py` and
`GKM_FRESH_REPLAY=1 python gkm_try.py` report:
`RESULT levels=6 moves=101 replay_ok=True err=None`.
