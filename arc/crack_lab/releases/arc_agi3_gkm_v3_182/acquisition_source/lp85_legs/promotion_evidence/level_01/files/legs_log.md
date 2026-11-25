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
