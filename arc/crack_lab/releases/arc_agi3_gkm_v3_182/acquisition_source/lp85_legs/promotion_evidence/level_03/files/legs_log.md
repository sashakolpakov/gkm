# Verified leg: align_ring_marker

Level 1 public-frame experiments, reproduced in probe_clicks.py and
probe_rotate.py:
- Twenty 4x4 blocks form a rectangular ring. Four 2x2 color-11 corners mark
  the upper-left slot; one ring block has that color.
- Clicking sampled blocks and the empty center leaves the frame unchanged.
- Clicking the left/right arrow rotates the ring one slot in opposite directions.
- Five left rotations bring color 11 to the marked slot and increase
  levels_completed from 0 to 1 on a clone.
- Repeated right rotations do not complete the level before terminal at move 13.

The leg derives slots, marker, and arrows from connected components, confirms
one-slot rotation on clones, and reduces circular distance to the marked slot.
It commits only a clone-verified winning path. players.play_level_1 only calls
this leg. ring_alignment_candidate.json preserves the verified coordinate path.

Final harness verification: RESULT levels=1 moves=5 replay_ok=True err=None.
No campaign checkpoint was edited.

## Level 2: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_lp85_arc_agi3_n0_fresh_frontier_gqq_1dxa/intersecting_tracks_candidate.json` and installed a thin replay player for the recovered suffix.

## Level 3: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_lp85_arc_agi3_n0_fresh_frontier_hnli396k/marked_cycles_candidate.json` and installed a thin replay player for the recovered suffix.
