# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## Level 5: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_ft09_arc_agi3_n0_fresh_frontier_tspxtzr0/coupled_miniature_candidate.json` and installed a thin replay player for the recovered suffix.

Level 6 clean reproduction: `probe_level6.py` replays the admitted checkpoint
through the public API and reads four 3x3 miniature clues at (12,6), (36,22),
(20,30), (44,46). Centers are 14; markers 0/2 specify same/different and 3
is omitted. All 22 tiles start at 11. Three clicks per tile on independent
entry clones reproduced the 11/14 cycle and selected-tile/above-tile coupling.
The existing match_coupled_miniatures leg solved an entry clone. Its mismatch
bit vector supplies dense progress and its final clone replay checks reward.
Added only a leg composition in play_level_6. Harness verification:
RESULT levels=6 moves=75 replay_ok=True err=None (13 new real actions).
Full candidate replay preserved in level6_coupled_miniatures_candidate.json;
checkpoint.json was not changed.
