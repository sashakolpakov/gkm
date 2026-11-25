# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## Level 3: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+proposer_last.log` and installed a thin replay player for the recovered suffix.

## Level 4: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+proposer_last.log` and installed a thin replay player for the recovered suffix.

## Level 5: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+proposer_last.log` and installed a thin replay player for the recovered suffix.

## Level 6: composed and replay validated

`play_level_6` reuses only the four existing legs; no new leg was needed.
The source-colored reference (6) has center (row=6, col=30). The verified
staged component bounds (inclusive row/column) are:

- First bend: (32, 35, 37, 40).
- Second bend: (26, 20, 31, 25).
- Vertical bar: (26, 14, 37, 16).
- Gold-gap bar: row 11..13, blue columns 23..28 and 32..37.

Reproduction: `python probe6_verify_legs.py` replays the pristine checkpoint
onto clones, verifies the saved 41-action suffix, and independently composes
the same layout from existing legs. Moving either bend one horizontal step
from that winning layout fails. Selection via the existing component-center
leg works for both bends. These are observed facts; the exploratory routing
models were hypotheses and their generated candidates did not verify.

Final campaign verification: `python gkm_try.py` reported
`RESULT levels=6 moves=166 replay_ok=True err=None`.
Saved suffix: `level6_routed_candidate.json`. Full candidate path:
`sp80_levels_1_to_6_candidate.json`. The supervisor checkpoint was not edited.
