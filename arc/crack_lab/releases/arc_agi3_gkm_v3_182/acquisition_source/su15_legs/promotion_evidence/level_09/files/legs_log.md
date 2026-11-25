# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## Level 3: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_su15_arc_agi3_n0_fresh_frontier_rknmzzuf/candidate_two_goal_merge.json` and installed a thin replay player for the recovered suffix.

## Level 7: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_su15_arc_agi3_n0_fresh_frontier_si7pk8u5/candidate_dual_region_routes.json` and installed a thin replay player for the recovered suffix.

Level 8, reproduced from the supervisor prefix (120 moves):
- Initial field: color 8 at (x=23,y=27), color 11 at (4.5,41.5)
  and (14.5,43.5), four color-9 corner regions, three moving color-7 shapes.
- Clone click (8,40) merges the two 11 markers into one 12 at (8,40).
  Nearby clicks pull markers; distant clicks leave them while hazards advance.
- Action 7 after a coordinate move restores the preceding frame (undo).
- A hazard can reduce 8 to 12 and can further damage 12. Idle probes reproduce
  both changes. Keeping two 12 markers safe and placing them in distinct corner
  regions increases reward, even though four regions are available.
- Existing merge_preserving_mass handles the initial merge. New reusable
  reduce_markers_for_placement searches mixed-size pulls until all surviving
  markers have the requested size. Existing place_matching_markers now permits
  fewer markers than regions and assigns to a distinct subset.
- Verified composed player: RESULT levels=8 moves=129 replay_ok=True err=None.
  Reproduce with python gkm_try.py. verify_level8.py separately saves the full
  validated path as candidate_verified_level8_campaign.json. checkpoint.json
  is unchanged. search_level8.py preserves the initial bounded discovery search;
  candidate_reduce_and_place.json is only its suffix from a cloned merged state.

Level 9, reproduced from the supervisor prefix (129 moves):
- Entry: square markers 6 at (18.5,46.5), 6 at (23.5,52.5), and 8
  at (38,51); four moving 7 sprites; three color-9 regions. Header requests
  square colors 12 and 15 plus a crowned color-13 moving sprite.
- `probe_headers.py` reproduces the per-level objective glyphs from the prefix.
  `probe_prior_win8.py` reproduces the prior level's final field observations.
- `probe_small9.py` confirms equal small squares merge into 15 after a pull.
  `probe_combine9.py` confirms two moving 7 sprites merge into 14.
  `probe_crown9.py` confirms two 14 sprites merge into a crowned 13.
  Moving sprites can still damage square markers after these merges.
- `probe_mixed_balanced9.py` finds a reward-increasing 12-action suffix on a
  pristine level-entry clone. It uses sprite templates because touching
  diagonal sprites can form a single connected component before they merge.
- The final suffix places 15 near (12,55), 12 at (53,55), merges the moving
  sprites into 13 near (20,50), then waits one turn for the reward.
  Thus the crown is an existence objective; it need not occupy a region.
- New general leg `assemble_mixed_objectives` searches feasible marker-mass
  partitions, equal-agent merges, and distinct-region assignments. The player
  only composes this imported leg. It commits only a reward-verified clone path.
- Candidate suffix: candidate_mixed_objectives9.json. Full prefix plus suffix:
  candidate_mixed_objectives9_campaign.json. checkpoint.json is unchanged.
- Harness verification: `python gkm_try.py` reports
  `RESULT levels=9 moves=141 replay_ok=True err=None`.
