# Unverified frontier brief: wa30 level 9

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The autonomous finish leaves exactly the untouched row-8 left cargo. The avatar has 23 turns after dismissal, and a direct route to the remaining top target takes 21: reach the cargo from its right, pick it, climb to row 2, align over column 7, and place downward. I’m verifying columns 6 and 7 because the courier’s last delivery can choose either top slot.
- The local cargo is successfully carried to the top, but the frame shows it one row above the outlined target: avatar on row 1, cargo on row 2, target on row 3. The column-6 route has one legal action left, so I’m testing the final downward push that should move the cargo into the outline and complete the level.
- At turn 65 the avatar is correctly at row 2, column 6 and still carrying the local block; the issue is only the five-action placement micro-mechanic. I’m probing each single action from that exact state to see which establishes downward facing without dropping above the target.
- The facing hypothesis was wrong; the carried block still lands above the outline. This is now a five-action microstate with no strategic uncertainty. I’m using an exact frame-deduplicated BFS from turn 65, with the real level increment as the goal. At most 3,125 raw sequences exist, and observational deduplication should make it much smaller.
- No five-action suffix can win, so this is not merely a placement orientation issue—the state is still missing another delivery. I’m printing the eight exact target-cell signatures at turn 44, after dismissal, and at turn 65 to identify which courier/manual block never entered the objective.
- The dense metric resolves it: by turn 65, seven target cells are filled and only `(3,7)` is empty. I had aligned the carried local block over column 6. Aligning over column 7 costs one extra move and leaves exactly four actions; I’m running the exact four-action reward search from that corrected alignment.
- The four-action search confirms the corrected alignment is one action too late. There is a safe one-action saving in the pickup route: from column 6, four left moves already reach column 2 and establish left-facing; the fifth was only a blocked repeat. I’m removing it and verifying the pickup, then rerunning the five-action completion search.
- The four-left shortcut changes the carry geometry and never reaches the target, so it’s discarded. I’m checking the other likely redundant move: five upward steps place the avatar one row lower than the verified alignment. That should allow a side/below placement while preserving five completion actions.
- The remaining saving is earlier and mechanical: the first remote delivery uses four upward actions, but the fourth was observed as a blocked carry step in the analogous route. I’m removing that one action and replaying the same two-remote plan; if the three-action thief contact still works one turn earlier, the verified column-7 alignment will have five completion actions.
- The shortened first delivery works and improves the whole schedule: the same `[up, left, interact]` contact dismisses color 15 at turn 46. This gives the verified local-cargo alignment at turn 65 with five actions remaining. I’m running the final five-action reward search from that exact state.
- The one-turn earlier dismissal changes the avatar’s row: it ends at `(7,6)`, so reusing the old two-down pickup route overshot to row 9 and never collected the local cargo. The corrected route needs only one down, saving another action and reaching the top-right alignment at turn 64 with six actions available.
- The lower local cargo is picked, but the remaining row-7 cargo blocks a straight upward carry from column 2. This is geometry, not timing: step right once to column 3, climb, then use four right moves instead of five. The route length stays unchanged and clears the obstruction.

## Preserved local probes

- `probe9.py` (13826 bytes)
- `probe9_current.py` (19683 bytes)
- `probe9_search.py` (13865 bytes)
- `wip_glue_notes.md` (642 bytes)

Run or inspect the smallest relevant probe before writing another one.
