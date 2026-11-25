# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## Level 3

Reproduced from the supervisor checkpoint via `probe_level3.py`:
color 9 moves north under action 1, while actions 2/3/4 are blocked at
entry. Three color-8 components contain 24 pixels total.
`probe_reuse_level3.py` confirms on a clone that `clear_linked_sentries`
reduces that count to zero without completing the level; then
`traverse_linked_tiles` raises levels_completed from 2 to 3.
Reuse the level-2 composition without adding a new leg.

Committed harness verification: `RESULT levels=3 moves=47 replay_ok=True err=None`.

## Level 4

Reproduced the checkpoint entry at levels_completed=3 with 47 real actions.
`probe_level4.py` tests all four advertised actions on independent clones:
only action 4 moves color 9 at entry, from (40,21) to (40,27), while
color 12 moves from (22,39) to (28,39). Blocked actions leave both in place.
`probe_mechanics.py` confirms that three eastward moves are fatal, then
tests the existing composition on a separate pristine clone.
`clear_linked_sentries` removes color 8 while preserving color 9 and
leaving levels_completed=3; `traverse_linked_tiles` then raises it to 4.
No new leg is needed: reuse the level-2/3 composition.

Committed harness verification: `RESULT levels=4 moves=64 replay_ok=True err=None`.

## Level 5

Reproduced entry at levels_completed=4 after 64 moves.
`probe_level5.py` uses perception action deltas and object candidates: action 3
moves color 9 six pixels west; blocked actions leave the patrol objects fixed.
`probe_search5.py` reproduces facing marks and moving color-12 objects.
Four straight westward moves are fatal; backtracking changes the timing.
`probe_static5.py` confirms the existing static leg cannot route through the
patrol-occupied tile graph. `probe_verify5.py` independently replays the saved
29-action candidate: visible patrol pixels fluctuate rather than monotonically
disappearing, and entering the goal raises the reward to 5.

Added general `traverse_patrolled_links`: static link-graph distance guides
bounded clone search, retaining moving-object positions and facing marks.
An independent clone replay confirms each complete route before committing.
The level player only composes this leg. Earlier players are unchanged.
Candidate preserved in `level5_patrol_candidate.json`.

Committed harness verification: `RESULT levels=5 moves=93 replay_ok=True err=None`.

## Level 5: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+proposer_last.log` and installed a thin replay player for the recovered suffix.

Level 6 clean clone experiments (2026-11-16):
- `python gkm_try.py` reproduced entry: levels=5, moves=93, replay_ok=True.
- `probe_level6.py` tested all advertised actions on independent clones. Only
  action 3 moves from entry: color 9 goes from (18,48) to (18,42), and color 12
  advances. Actions 1,2,4 change only the bottom move meter at entry.
- `probe_mechanics6.py` reproduces movement in other contexts: 332 reaches
  (24,36), 3322 reaches (30,36), 33224 reaches (30,42). Straight 333 is lethal.
- `probe_clear6.py`: 332243 leaves the sentry tile (30,42) empty permanently;
  3322334 similarly clears (30,24). Patrol overlays can temporarily hide an
  uncleared sentry, so raw color-8 pixel count is an unreliable subgoal.
- Extended existing clear_linked_sentries with optional incremental flanks:
  retain initial sentry centers, search for a center becoming empty or avatar,
  independently replay each route, then commit and repeat. This supplies
  monotone progress without requiring a full all-sentries BFS.
- `probe_incremental6.py` verified all six sentries cleared in 22 actions, then
  existing traverse_patrolled_links reached reward 6 in another 6 actions.
  Clone suffix saved in level6_incremental_candidate.json for independent use;
  checkpoint.json was only read. play_level_6 composes these existing legs.

## Level 6: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_tu93_arc_agi3_n0_fresh_frontier_zp54_71g/level6_incremental_candidate.json` and installed a thin replay player for the recovered suffix.
