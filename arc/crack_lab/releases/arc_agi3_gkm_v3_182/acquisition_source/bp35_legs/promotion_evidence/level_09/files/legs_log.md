Level 1 clean observations (reproduced by local probe.py and harness):
- Public actions are (3, 4, 6, 7); all coordinate probes used in-frame integer triples.
- On pristine clones, 3 moves colour 9/11 avatar from x21 to x15;
  4 moves it to x27. Repeated 3 stops at the left boundary.
- Clicking yellow centres (15,3), (21,3), or (33,15) removes the
  corresponding 14-colour tile. Clicking empty space leaves geometry unchanged.
- After four right moves the avatar enters the gap at x45 and the scene
  scrolls down three tile rows while the avatar stays near y39.
- Clicking an overhead yellow tile causes a further rise. Indestructible
  colour 5 tiles require moving sideways to an opening.
- Action 7 after movement or a tile removal restores the preceding geometry.
- A 15-action clone route reached the green tile (colour 7) and increased
  levels_completed from 0 to 1; nearby air and intermediate removals did not.
- candidate_ascent.json preserves that clone path. probe.py reproduces the
  route from pristine entry using only public frames, clones, and actions.

Reusable leg: ascend_breakable_shaft uses a visible tile graph, assigning cost
for horizontal moves and yellow removals. Upward reach is the dense objective;
when visible, the green tile becomes the target. It replans after each step.

Validation: python gkm_try.py
RESULT levels=1 moves=15 replay_ok=True err=None

## Level 2: forced rise and ceiling hazards

Reproduced with `probe_level2.py`, `probe_trace.py`, `probe_mechanics.py`, and
`probe_skill.py` from the checkpoint prefix using public clone/frame/step calls.
Color 9 responds to keys 3/4 by moving one six-pixel column left/right.
Coordinate clicks remove color-14 tiles. Clearing the overhead tile triggers
forced ascent and camera scrolling, sometimes several rows in one action.
The old shaft leg dies on its sixth level-2 action: after `[4,4,4,
[6,39,33],[6,39,33]]`, another `[6,39,33]` is terminal without reward.
A click on the white object itself does not remove it. Clearing sideways
yellow cells and moving left instead avoids that outcome.

Added `ascend_with_ceiling_hazards`: bounded visible-board BFS models tile
removal and forced ascent, excludes contact with white ceiling objects, and
replans after scrolling. Dense subgoal: reach the top visible row safely;
when the green exit becomes reachable, enter it. Clone-check each chosen action.
The existing shaft leg cannot represent forced-ascent hazard avoidance.

Pristine level-2 clone completion: 40 actions. Full harness result:
`RESULT levels=2 moves=55 replay_ok=True err=None`. Saved the complete candidate
as `bp35_level2_validated_path.json`; checkpoint.json was not edited.

Level 8 clean observations (reproduced by probe_level8.py and probe_growth_plan.py):
- Harness entry: levels=7, prefix=223 moves. Key 3/4 moves color-9 avatar horizontally; 7 waits. Upward gravity resolves after actions.
- Color-15 area-21 squares are branching supports: a click removes the square and grows into adjacent empty cells. Area-12 triangles are lethal. Three horizontal growth clicks stage a safe ascent that otherwise dies after two rights.
- Color-12 area-21 supports toggle to area-5 outlines. Closing passages behind ascent preserves the return floor.
- Upper color-8 switch reverses gravity remotely. Retain the observed exit column while descending; approach its enclosure from above.
- New leg uses bounded symbolic configuration search, height progress, clone rejection of lethal edges, and replanning after camera movement.
- Validated campaign: RESULT levels=8 moves=283 replay_ok=True err=None. Preserved bp35_level8_clone_candidate.json (60 actions) and bp35_level8_campaign_candidate.json (283 actions); checkpoint.json untouched.
