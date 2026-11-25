# Leg-library debrief log

Recurring composition patterns and repeated novelty.

Level 4 clean reproduction (probe_sc25.py): entry follows the unchanged
32-action prefix. Keys 1/2/3/4 move the 4x4 colour-9/10 avatar up/down/left/right.
Clicking palette (4,4) shows four zero-colour stencil cells in the input grid;
completing them shrinks the avatar to 2x2 and movement to two pixels.
After down twice before shrinking, down once and left once align the avatar
with the two-pixel-high left passage. Selecting palette (4,15) and filling its
three zero-colour cells while facing left removes both the left colour-6
fixture and the colour-13 exit barrier. This works from the main room.
Right five, down three, right four then increases clone reward from 3 to 4.
Dense subgoals: avatar area 16 -> 4, fixture and barrier counts -> 0,
then distance to exit. Player uses repeat_key and activate_palette_pattern.
Final harness: RESULT levels=4 moves=57 replay_ok=True err=None.
Candidate level suffix: level4_palette_ray_candidate.json. Checkpoint untouched.

## Level 4: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_sc25_arc_agi3_n0_fresh_frontier_uize5e8g/level4_palette_ray_candidate.json` and installed a thin replay player for the recovered suffix.
