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

## Level 5: hand-controlled platform and cyclic bridges
- Reproduced supervisor entry: 4 levels, 195 actions. `probe5.py` uses perception components and cloned action deltas to identify the color-14 walker and two-pixel movement.
- `probe_grab.py` verifies that up three/right three presses bring the hand beside the color-12 platform; (52,35) grips it, and subsequent hand directions relocate it in four-pixel steps around the black corridor.
- `probe_ride.py` and `probe_rider_detail.py` falsify automatic riding: dragging is blocked while the walker occupies the platform. `probe_stage.py` verifies bridging to the middle island (20,26), parking the walker there, returning the platform, and climbing to (6,22).
- `probe_portal.py` verifies that walking to (6,24) then pressing (52,48) transports the walker to (34,10). Dragging the platform left three/down three now connects this portal to the isolated pickup at (18,10); `probe_pickup_left.py` verifies collection and the revealed pink button at (52,14).
- `probe_finish.py` verifies four presses of each cyclic control (56,22) and (47,22), plus the revealed pink switch. Crossing the extended horizontal bridge, descending the right pink bridge, and restoring the left pink bridge reaches the color-11 goal and increments the clone reward to 5.
- Dense progress is staged island/pickup reachability and Manhattan distance for the existing navigation leg. New `manipulate_platform_controls` verifies each drag or cyclic-control press on a clone before committing. Level 5 only composes these two legs.
- `candidate_level5_win.json` preserves the complete independently clone-verified level-5 suffix. Checkpoint state was not edited.
- Integrated harness verification: `python gkm_try.py` -> `RESULT levels=5 moves=305 replay_ok=True err=None` (110 new real actions).

## Level 5: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_dc22_arc_agi3_n0_fresh_frontier__b4isum0/candidate_level5_win.json` and installed a thin replay player for the recovered suffix.

## Level 6: remote pads and a transported bridge

- `explore6.py` reproduces key directions with the color-14 avatar at (52,28): keys 1/2/3/4 move up/down/left/right by two cells when supported. The initial down/right moves are blocked.
- The blue pickup at (48,34) reveals the grab control. The pickup near (18,8) reveals (50,46), which cycles the source portal at (48,18) through four color pairs (`portals6.py`). From the source, (51,25) transports to (52,32), nowhere, (4,4), or (58,34), respectively (`channel6.py`); the same action rotates the long bridge.
- In the isolated room, standing at (56,34), (58,32), (58,36), or (60,34) enables the corresponding arrow control. Other arrows are inactive. Enabled clicks move the hand four cells; grip at the bridge center changes the hand outline and subsequent movement translates the bridge.
- Verified bridge route: hand left four times to grip at (34,10), right five, up three, left two, up three. This places the horizontal bridge at (10,22), joining the two top islands. The avatar leaves the room through its center portal, selects the top-left destination, and crosses to (6,46).
- `candidate_level6_verified_win.json` preserves a clone-verified suffix that increments reward to 6, including exploratory no-ops. The player composes only the two existing legs. `board_width` lets navigation ignore sidebar glyphs when measuring avatar distance while retaining the full frame in search state keys.
- `compare6.py` verifies the portal selector's delayed visible effect: while the avatar covers the source tile, selector clicks leave the frame unchanged, but the following transport reveals the chosen channel. `manipulate_platform_controls(..., verify_each=False)` therefore verifies the complete operation's visible outcome on a clone before committing it.
- Final harness verification: `python gkm_try.py` reports `RESULT levels=6 moves=448 replay_ok=True err=None`. The composed suffix uses 143 real actions after the unchanged 305-action checkpoint prefix.

## Level 6: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_dc22_arc_agi3_n0_fresh_frontier_yveteuv7/candidate_level6_verified_win.json` and installed a thin replay player for the recovered suffix.
