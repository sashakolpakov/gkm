# Level 1: reproduced raw-frame observations

All experiments used fresh level-entry clones via the local harness, perception
components, and public frame/step/clone methods. No external sources were used.

- Initial outlined objects have centers (x,y)=(19,31) and (28,31).
  Default action_deltas and separate one-step replays identify keys 1,2,3,4
  as up,down,left,right: the first object moves three pixels; the second stays.
- Clicking (28,31) transfers the black center/control to the second object.
  Its up move is blocked by the corridor wall. Right moves reach x=31, then
  stop at the vertical color-15 barrier (x=33..38).
- From pristine entry, three right moves produce controlled center (25,31)
  and autonomous center (43,31). Further right moves of the controlled object
  stop at (31,31). Thus this observed displacement gets the second object
  across a barrier that direct controlled movement cannot traverse.
- Clicking the displaced object, then right and up, parks it at (46,28),
  the center of the right yellow outline. Reward remains zero.
- Switching to (25,31), then left four times and down, reaches the other
  yellow center (13,34). Reward increases from zero to one on this last move.
  The first parked object stays in place during the return trip.
- Separately, walking directly to the left target from pristine entry gives
  no reward. Both targets occupied is the supported completion condition.

Dense progress: objects across the barrier, then number of occupied target
centers (0 -> 1 -> 2). Candidate replay is preserved in
`candidate_cooperative_targets.json`; probe.py reproduces its observations.

Library: `move` repeats observed cardinal steps; `select_object` transfers
control at an observed center. The player composes these skills for this layout.

Level 2: reused move and select_object only. Foreground probe.py reproduces the clone route from the supervisor prefix. Small shape moves right 4, up 3, left 1 to launch the square and tall shape into the left room. Square staged below tall shape launches it upward across the gap. All four matching outline placements yield levels_completed=2. Harness: RESULT levels=2 moves=52 replay_ok=True err=None. Candidate suffix: candidate_shape_launch.json; full campaign: candidate_shape_launch_campaign.json. checkpoint.json unchanged.

Level 4: reproduced from the validated 85-action entry using probe.py and
perception connected_components/action_deltas. Both color-14 small objects are
selectable; cardinal moves translate the selected one by three pixels. The
unselected object remains still absent contact. Direct downward moves stop at
y=36 above the color-15 barrier at y=38..43. From entry, right three then down
launches the second object from (30,33) to (30,48). Selecting it and moving down
three, left, up launches the cross from (24,54) to (24,33), across the barrier.
The preserved level4_candidate.json suffix moves the cross to (39,33), then
(39,18), then (39,12). Reward stays at three through parking the upper small
object at (51,30), and rises to four when the lower object reaches (39,54).
Dense progress: lower-room access, cross returned, cross outline occupied,
then two small outlines occupied. Reused move/select_object without new legs.
Validation: python gkm_try.py -> RESULT levels=4 moves=134 replay_ok=True err=None.

## Level 4: recovered verified path artifact

The proposer found a winning suffix but did not integrate it before the time budget ended. Harness recovery validated `checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_ka59_arc_agi3_n0_fresh_frontier_bk5eh1cg/level4_candidate.json` and installed a thin replay player for the recovered suffix.
