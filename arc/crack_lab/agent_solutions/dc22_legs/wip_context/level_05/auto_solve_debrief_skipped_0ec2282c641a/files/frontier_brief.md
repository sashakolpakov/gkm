# Unverified frontier brief: dc22 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The dock glyph is a one-shot “lock/attach” action: it transforms the platform from 30 to 28 cells at the right-hand dock. This is a real subgoal, not an avatar selector. I’m now rescanning the UI after attachment and retesting the pocket, because the activated dock state was absent from every earlier configuration/search key.
- The dock changes the controls themselves: the old up/right buttons go inert, and a new active button appears on the left half of that UI row at `(44,29)`. It moves the attached platform together with the large color-12 assembly (126 changed cells). I’m tracing its phases now; this is the cooperative compound-object mechanism the frame-key searches missed.
- Three presses of the new button slide the entire docked compound left. On phase 3 it fully consumes the small color-15 switch—the first verified dense subgoal completion. I’m scanning the post-switch UI for the next newly enabled control before moving any other structure.
- The completed switch reconfigures the control row again: two new controls appear at `(54,29)` and `(60,29)`. The first changes the compound in place/possibly reverses it; the second affects rows 10–25, indicating a new downward branch into the lower chamber. I’m tracing both on clones and will follow the branch that increases reachability.
- The downward branch is correct: three presses lower the compound until the color-12 column reaches the left teleport corridor. The geometry now suggests the next dense subgoal—shift that lowered column right until it joins the starting pocket. I’m testing the existing right control from this lowered state and then probing avatar reachability.
- At the lower-right position, only the compound’s left control remains active. The avatar can now enter the color-12 column from the starting pocket, so this matches level 4’s synchronized-builder pattern. I’m testing `left-control, step-left` from both the lower and upper parts of the occupied column to determine the safe crossing phase.
- Occupying the column correctly pins it; the avatar can step only one tile left before the compound refuses to move. The safe strategy is therefore staged cooperation: enter the column, climb into the stationary right corridor, then move the unoccupied compound left to expose its upward control and restore the top bridge while the avatar waits. I’m identifying that lower-left control set now.
- The cooperative route is now fully determined through the first portal: the avatar waits in the right corridor while the empty compound moves left, up, then right; the restored top column connects that waiting cell to the upper teleport endpoint. I’m replaying the whole chain to verify the teleport and establish the lower-region state before solving the final builders.
- The first portal route is verified exactly; the avatar arrives at logical `(17,5)` and descends to `(22,5)` after restoring the lower bridge. The next color-5 span matches the synchronized builder from level 4, so I’m crossing it with `B, right` phases and scanning the far endpoint for the checkerboard color-15 bridge/control hidden by coarse rendering.
- The synchronized B crossing is verified: four `B, right` phases plus four rights reach `(22,14)`. The vertical checkerboard below is not yet passable and exposes no new UI button. Its lower anchor is the A-controlled builder on row 27, so I’m testing A’s phases as the contextual activation for that shaft.
- A’s docked phase does not expose another UI control, so the next unresolved affordance is local: the avatar is now adjacent to the color-15 checkerboard shaft. I’m scanning gameplay coordinates from that endpoint, before and after docking A, to catch a context-only shaft activation that was impossible from the starting pocket.
- The shaft is not directly clickable. One physical condition may still be preventing the second dock: the avatar is occupying B’s far endpoint while A reaches its lower anchor, analogous to how occupied bridges pin earlier transformations. I’m vacating B back to the left corridor, then docking A and rescanning for the compound activation.

## Preserved local probes

- `probe_astar5.py` (2948 bytes)
- `probe_bfs.py` (2907 bytes)
- `probe_bfs5.py` (2851 bytes)
- `probe_branch5.py` (2023 bytes)
- `probe_bridge_context5.py` (3247 bytes)
- `probe_button_pixels5.py` (1043 bytes)
- `probe_cd5.py` (1628 bytes)
- `probe_cd_sequences5.py` (1563 bytes)
- `probe_click_controlfollow5.py` (2313 bytes)
- `probe_click_pairs5.py` (1170 bytes)
- `probe_clickscan5.py` (1191 bytes)
- `probe_compound5.py` (1705 bytes)
- `probe_configs5.py` (3105 bytes)
- `probe_context5.py` (2464 bytes)
- `probe_context_clickfollow5.py` (1683 bytes)
- `probe_context_clickscan5.py` (1604 bytes)
- `probe_direct.py` (2044 bytes)
- `probe_dock5.py` (1126 bytes)
- `probe_e_trace5.py` (1180 bytes)
- `probe_entry5.py` (1720 bytes)
- `probe_level1_reward.py` (1135 bytes)
- `probe_level2.py` (4438 bytes)
- `probe_level2_context.py` (5036 bytes)
- `probe_level3.py` (2280 bytes)
- `probe_level3_context.py` (1830 bytes)
- `probe_level3_post15.py` (2142 bytes)
- `probe_level4_context.py` (1860 bytes)
- `probe_level4_routes.py` (2278 bytes)
- `probe_level4_solution.py` (1591 bytes)
- `probe_level4_tail_bfs.py` (1965 bytes)
- `probe_level5.py` (2142 bytes)
- `probe_lower5.py` (4854 bytes)
- `probe_map5.py` (1900 bytes)
- `probe_observe.py` (3074 bytes)
- `probe_platform5.py` (1588 bytes)
- `probe_reach_stage5.py` (1658 bytes)
- `probe_select5.py` (1379 bytes)
- `probe_sequence.py` (2217 bytes)
- `probe_stage5.py` (1117 bytes)
- `probe_sync_compound5.py` (1696 bytes)
- `probe_teleport5.py` (1282 bytes)
- `probe_turns5.py` (1194 bytes)
- `probe_ui5.py` (774 bytes)
- `probe_ui_contextscan5.py` (2284 bytes)
- `wip_glue_notes.md` (10887 bytes)

Run or inspect the smallest relevant probe before writing another one.
