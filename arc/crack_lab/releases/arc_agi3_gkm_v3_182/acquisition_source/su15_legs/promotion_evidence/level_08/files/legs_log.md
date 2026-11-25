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
