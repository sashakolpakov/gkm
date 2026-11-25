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

Level 4: `distribute_chain_to_rails`
- Reproduced entry with the validated 174-action prefix using public steps.
- `probe_level4.py` replays the independently saved 26-action suffix in
  `rail_beam_candidate.json` and prints symbolic cargo positions and markers.
- At suffix step 7, colors 9 and 8 occupy columns 24 and 36 at row 19;
  the first marker in each order becomes hollow without a level reward.
- At step 17, 9 is at (13,24), 14 at (19,24), and both markers for rail 10
  are hollow. Rail 11 still lacks its second marker.
- At step 26, pushing color 12 upward from below completes level 4.
- The new leg reads rail IDs and orders from blobs, then uses a bounded beam
  of public clones. Its dense score combines column distance, vertical order,
  and hollow markers. The player only composes this imported leg.
