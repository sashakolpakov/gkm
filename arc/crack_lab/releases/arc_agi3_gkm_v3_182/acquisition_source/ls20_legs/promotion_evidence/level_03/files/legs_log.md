# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## Level 2: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_ls20_arc_agi3_n0_fresh_frontier_z2rwf0iy/level2_rotation_candidate.json` and installed a thin replay player for the recovered suffix.

## Level 3 clean observation and replay
- Entry reproduced with `python gkm_try.py`: levels=2, moves=58, replay_ok=True.
- `probe.py` used perception components and clone action deltas: key 1 moves the two-colour 5x5 avatar up, 2 down, 3 left, 4 right; walls can block movement. Boundary transitions can relocate the avatar nonlocally.
- Small colour-11 board objects disappear on contact and refill the colour-11 status bar. Both are not required for completion.
- `search_probe.py` searched only public cloned frames/actions, throttled to 250 clone steps/second and bounded at 18000 steps. Found the 39-action `level3_bfs_candidate.json` route.
- `verify_probe.py` reproduced that route twice from pristine level-3 entry clones: each increased levels_completed from 2 to 3. In zero-based 5x5 board coordinates (origin pixel row 5, column 9), the route visits (8,4), refills at (5,2), visits (1,8), relocates to (8,9), then key 2 completes the level. The refill at (2,5) remains present immediately before completion.
- Reused `follow_key_runs`; no new leg needed. `play_level_3` only composes that imported leg.
- Final harness: RESULT levels=3 moves=97 replay_ok=True err=None.

## Level 3: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_ls20_arc_agi3_n0_fresh_frontier_b85xxaho/level3_bfs_candidate.json` and installed a thin replay player for the recovered suffix.
