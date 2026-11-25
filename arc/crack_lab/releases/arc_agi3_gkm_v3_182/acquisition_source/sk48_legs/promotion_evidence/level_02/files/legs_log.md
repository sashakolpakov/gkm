# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## Level 1: ordered chain collection
- Reproduced with public-frame clone probes in `probe_initial.py`,
  `probe_structure.py`, `probe_interact.py`, `probe_click.py`,
  `probe_chain.py`, and `probe_goal.py`.
- Actions 1/2 move the gripper vertically by six pixels. Actions 3/4
  retract/extend its horizontal reach. Reaching a block attaches it;
  attached blocks travel with the gripper and push blocks in their way.
- The bottom strip orders colors 8, 14, 9. Its matching center becomes 0
  when that chain position matches. Retracting beyond zero reach releases
  a chain member. Action 7 reverses a previous effective move in probes.
- Dense progress: matched strip slots, then distance to the next block's
  row and distance of the carried chain from full retraction.
- Attaching all three in strip order raised levels_completed from 0 to 1.
- `collect_ordered_chain` plans on a clone using frame observations and
  replays only a successful plan. `play_level_1` composes this single leg.
- Harness validation: `RESULT levels=1 moves=24 replay_ok=True err=None`.
- Independently adoptable path: `candidate_collect_order.json`.
