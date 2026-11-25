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

## Level 4 clean observation and replay
- Entry reproduced by `python gkm_try.py`: levels=3, moves=97, replay_ok=True.
- `probe4.py` uses perception components and cloned action deltas. The 5x5 pink/red avatar starts at board (0,9), with cell origin pixel (5,9). Actions 1/2/3/4 move up/down/left/right; some transitions relocate nonlocally.
- `cycle4.py` reproduces the central color tile cycling the HUD 14 -> 8 -> 12 -> 9 -> 14. `glyph4.py` reproduces the other central tile changing the HUD pattern; its first activation matches the target pattern.
- `search4_full.py` uses only public cloned frames/actions, bounded at 16000 clone steps and throttled to 250 steps/second. Its state key preserves the HUD pattern and color. The earlier board-only key in `search4.py` incorrectly merged distinct HUD states and exhausted its reachable quotient without a win.
- `level4_search_candidate.json` preserves the winning 43-action suffix. `verify4.py` independently replays it from a pristine level entry clone: color changes at actions 13/15/17, target pattern matches at action 28, a purple object disappears at action 37, and action 43 increases levels_completed from 3 to 4. The other purple object remains immediately before completion, so collecting both is unnecessary.
- `play_level_4` composes the existing `follow_key_runs` leg; no new leg needed.
- Final harness: RESULT levels=4 moves=140 replay_ok=True err=None.

## Level 4: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_ls20_arc_agi3_n0_fresh_frontier_6ah5roa1/level4_staged_candidate.json` and installed a thin replay player for the recovered suffix.

## Level 5: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_ls20_arc_agi3_n0_fresh_frontier_owwcfm9h/level5_pattern_rotation_candidate.json` and installed a thin replay player for the recovered suffix.

Level 6 verified: action deltas identify the 5x5 two-colour avatar; legal movement advances three smaller objects, while blocked movement leaves them still. Rings disappear permanently on contact. The upper object cycles base patterns, the lower object rotates the displayed pattern, and the multicoloured object changes its colour. A matching displayed pattern clears each panel. The clone path in level6_panels_candidate.json clears all three rings and both panels in 106 moves. players.play_level_6 composes the existing follow_key_runs leg. Harness result: levels=6 moves=300 replay_ok=True err=None. Full campaign candidate: level6_verified_campaign_candidate.json. checkpoint.json was not modified.

## Level 7

Reproduced from pristine checkpoint entry with bounded frame-only clone probes. Keys 1/2/3/4 move up/down/left/right. Rings refill the shrinking bar once. The lower-left colour tile cycles 12, 9, 14, 8; the nearby shape tile cycles six masks. A moving rotator travels the right corridor and rotates the held mask on contact. Its phase advances on successful movement; blocked moves consume bar without advancing it. The exit accepts colour 8 and mask 101/110/011. The preserved 53-action `level7_exit_candidate.json` reaches levels_completed=7 on a clone. The player reuses `follow_key_runs`; no new leg is needed.

## Level 7: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+proposer_last.log` and installed a thin replay player for the recovered suffix.
