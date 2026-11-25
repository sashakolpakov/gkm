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

Level 5 clean reproduction: `probe_level5_progress.py` starts from the admitted
checkpoint path and uses a clone. White palette completion reduces the moving
9/10 sprite from 4x4 to 2x2; purple completion then places it at (29,39).
Seven left keys and two up keys reach (15,35). Completing the colour-6 vertical
pattern while facing up removes the marker at (16,12) and barrier at (52,16),
even though the orange gap blocks further movement. Four down keys and one left
key reach (13,43); completing that pattern facing left removes the marker at
(4,44) and barrier at (52,20). Two up keys collect the colour-14 key at (12,40).
White completion here grows the sprite; purple completion places it at (51,35).
Six up keys increase levels_completed from 4 to 5. These are existing palette
and repeat-key mechanics, so play_level_5 adds no new leg.

Harness verification: `python gkm_try.py` ->
`RESULT levels=5 moves=105 replay_ok=True err=None`.
`preserve_level5_candidate.py` independently reproduces the 48-action suffix on
a clone and saves the complete 105-action path in
`level5_palette_switches_candidate.json`. checkpoint.json was only read.
