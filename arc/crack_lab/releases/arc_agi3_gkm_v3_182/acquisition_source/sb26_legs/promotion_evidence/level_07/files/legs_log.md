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

## Level 3: nested boxes with different widths

Reproduced on clones by probe_level3.py: actions (5,6,7); selecting a palette
swatch changes its outline, clicking a leaf fills it, clicking a reference does
not fill it, and 7 removes the placement. Seven target colours are
[8,14,15,11,6,9,12]. Expanded leaf positions are
[(19,23),(19,35),(25,35),(31,23),(37,35),(43,35),(43,23)].
Filling those leaves then submitting 5 increased clone levels_completed to 3.

Generalized match_nested_colour_rows to read each box's own horizontal extent,
reusing match_colour_row for placement and dense verification. Fresh replay
initially exposed connected outlines in level 2. probe_level2_boxes.py reproduced
the reference connector joining the lower box and crossing the root bottom border.
The reader therefore detects outlines geometrically, allowing a two-cell connector
crossing on a bottom border. Player 3 only composes the existing nested-row leg.

Final resumed and fresh harness runs: levels=3 moves=39 replay_ok=True err=None.
Saved independently replayable path: nested_boxes_level3_candidate.json.
Checkpoint was not edited.

Level 4: reproduced public frame and clone action deltas in probe_level4.py. Keys 5/7 at entry changed one status pixel / nothing. Palette selection followed by socket click placed solid colour leaves; palette 14 placed a hollow reference. Root row has five sockets; row 14 has a fixed literal 14 and two sockets. Clone verified expansion [11,8,reference14,12,15], reference14=[14,9,6], then key 5 advanced reward from 3 to 4. Existing legs cannot insert references or accommodate immutable leaves, so complete_expandable_colour_rows plans these assignments from frame geometry. It verifies increasing correctly placed token outlines and reward on a clone before committing. Harness: RESULT levels=4 moves=54 replay_ok=True err=None. Candidate suffix is preserved separately in level4_expanded_reference_candidate.json; checkpoint untouched.

## Level 4: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_sb26_arc_agi3_n0_fresh_frontier_bxafq8be/level4_expanded_reference_candidate.json` and installed a thin replay player for the recovered suffix.

## Level 5: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_sb26_arc_agi3_n0_fresh_frontier_nog1uhmt/level5_inventory_reference_candidate.json` and installed a thin replay player for the recovered suffix.
