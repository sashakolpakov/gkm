# Leg-library debrief log

Recurring composition patterns and repeated novelty.

Level 6: reproduced entry through the supervisor prefix: 5 levels, 333 moves.
`probe_level6.py` uses only frames, advertised actions, clones, and perception.
Directional action deltas move the selected outline by 3 cells; action 5
switches selection. The central color-1 tile occupies rows/columns 28..35.
Pushing the square right against it three times changes its bounding span
from 18x18 to 27x9 while retaining 72 body pixels. Routing below the tile
then aligning covers all four color-11 rings without increasing reward.
For the cross, two upward contacts from below slide its horizontal stroke
up relative to its vertical stroke. Two leftward contacts from the right
slide its vertical stroke left relative to its horizontal stroke. The
existing visible-outline alignment then covers its four color-9 rings and
raises levels_completed to 6. The new generic deformation leg verifies a
body change relative to the selection marker before committing any moves.
`python gkm_try.py`: levels=6, moves=408, replay_ok=True, err=None.
`save_level6_candidate.py` independently clone-verifies and saves the path
under `candidate_level6_obstacle_deformation.json`; checkpoint is untouched.
