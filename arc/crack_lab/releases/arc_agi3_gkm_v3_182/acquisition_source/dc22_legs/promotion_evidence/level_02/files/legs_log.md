# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## Level 1: switchable platforms (verified)
- Public clone probes identify color 14 as the avatar: keys 1/2/3/4 move it by two pixels up/down/left/right on connected colored platforms. From pristine entry, repeated up stops at (row 38, col 10) until the lower bridge moves.
- Coordinate button (48,36) alternates the color 9 bridge between bbox (20,18,23,21) and (34,8,37,11). Button (48,20) alternates color 8 between bbox (30,12,33,17) and (24,18,29,21). Both effects were reproduced from entry and with the avatar at different pad positions.
- Verified subgoals: lower bridge -> red platform at (30,10) -> purple platform at (30,20) -> upper bridge at (24,20) -> destination color 11 at (20,24). The lower button must be toggled back after crossing it to restore the final connection.
- Dense progress: Manhattan distance from avatar to destination divided by the observed two-pixel stride. Clone A* retains the visible bridge state and omits the observed bottom-row move meter.
- candidate_switch_walkways.json contains an independently clone-verified 20-action route. The reusable search also passed the real harness: RESULT levels=1 moves=20 replay_ok=True err=None.
