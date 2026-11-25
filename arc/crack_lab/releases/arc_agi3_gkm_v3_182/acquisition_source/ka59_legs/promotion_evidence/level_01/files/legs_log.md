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
