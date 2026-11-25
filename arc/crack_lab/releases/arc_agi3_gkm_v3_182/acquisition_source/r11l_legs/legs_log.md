# Leg-library debrief log

Recurring composition patterns and repeated novelty.

Level 1: public-frame probes reproduced black endpoint (7,36), green
endpoint (27,59), white midpoint (17,47), and outlined target (39,21).
Clicks on free space move black; clicking green swaps the two endpoints.
Repeated clicks and clicks at (0,0) leave geometry unchanged. A click at
(1,1), on color 2, also leaves geometry unchanged. Across multiple placements,
the marker follows the endpoints' integer midpoint. No autonomous geometry
motion was observed during repeated clicks.

Dense subgoals: stage an endpoint eight pixels from the target, swap to
anchor it, then place the other endpoint symmetrically, taking midpoint
error to zero. Pristine clones verified both horizontal and vertical
placements complete level 1. The candidate horizontal path is preserved in
midpoint_candidate.json. Near placements also won, so exact tolerance is
unmeasured; the leg uses an exact symmetric placement and verifies reward
on a clone before committing. No existing legs were available.
