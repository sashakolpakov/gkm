# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## Level 1: switchable platforms (verified)
- Public clone probes identify color 14 as the avatar: keys 1/2/3/4 move it by two pixels up/down/left/right on connected colored platforms. From pristine entry, repeated up stops at (row 38, col 10) until the lower bridge moves.
- Coordinate button (48,36) alternates the color 9 bridge between bbox (20,18,23,21) and (34,8,37,11). Button (48,20) alternates color 8 between bbox (30,12,33,17) and (24,18,29,21). Both effects were reproduced from entry and with the avatar at different pad positions.
- Verified subgoals: lower bridge -> red platform at (30,10) -> purple platform at (30,20) -> upper bridge at (24,20) -> destination color 11 at (20,24). The lower button must be toggled back after crossing it to restore the final connection.
- Dense progress: Manhattan distance from avatar to destination divided by the observed two-pixel stride. Clone A* retains the visible bridge state and omits the observed bottom-row move meter.
- candidate_switch_walkways.json contains an independently clone-verified 20-action route. The reusable search also passed the real harness: RESULT levels=1 moves=20 replay_ok=True err=None.

## Level 3: revealed switch and paired transport (verified clone probes)
- Reproduced entry with the supervisor prefix: level 2, 64 actions. Keys 1/2/3/4 move color 14 by two pixels up/down/left/right when connected walkways permit; blocked keys only change the bottom meter.
- probe_mechanics.py reproduces both reversible bridge controls from entry and multiple avatar positions: (52,18) exchanges horizontal color-8 spans for vertical spans; (52,27) exchanges the left horizontal color-9 connection for three other connections.
- probe_branch.py reproduces transport: at avatar top-left (16,6), clicking (52,36) moves it to (46,12). The same click at entry only changes the paired patterned tiles.
- probe_tile.py reproduces the pickup at (42,16): the small color-15 object disappears and a 47-pixel color-15 button appears in the right panel. This does not directly open the patterned walkways.
- probe_unlock.py reproduces that button (52,45): no effect before pickup; after pickup, it fills 36 missing cells in the patterned walkways, and a second click reverses it. With those walkways enabled, the avatar can climb from (40,16) to (32,16), cross the lower blue bridge, and descend to color 11 at (38,32). The final step raises levels_completed from 2 to 3.
- candidate_level3_branch_pickup.json preserves the independently verified 47-action level-3 suffix. Replay it from pristine level-3 entry, after the supervisor prefix.
- Reuse: play_level_3 composes navigate_switchable_platforms to waypoint (42,16), then to the reward, adding the newly revealed switch only for the second stage. No new leg is needed. A direct goal search and a second-stage search without the revealed button both exhausted their bounds; clone observations falsified the missing-button assumption before the corrected player was tested.
- Final real harness verification: `python -u gkm_try.py` -> `RESULT levels=3 moves=109 replay_ok=True err=None`. Earlier levels and legs.py were unchanged; checkpoint.json was never edited.

## Level 3: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_dc22_arc_agi3_n0_fresh_frontier_47p03idf/candidate_level3_branch_pickup.json` and installed a thin replay player for the recovered suffix.

## Level 4: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_dc22_arc_agi3_n0_fresh_frontier_f99uwy6u/candidate_level4_staged.json` and installed a thin replay player for the recovered suffix.
