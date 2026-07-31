# Unverified frontier brief: dc22 level 4

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the stated clean-room surface. I’ll begin with `solver_index.md`, then `frontier_brief.md` if present, and inspect only targeted ranges plus the documented harness/API files.
- The index shows no prior frontier brief content, so I’m establishing the current level and reward state now with the required harness run.
- I’ve reached level 3 reproducibly through the current solver. Next I’m using a small observational probe: object components, action deltas, and a compact 4×4-cell symbolic map—no raw-frame transcript or hidden state.
- The movement experiment identifies the avatar as the 2×2 color-14 object at row 18/column 30: actions 1 and 3 move it up/left by two pixels; 2 and 4 are blocked in the starting context. The right panel contains three coordinate controls, so I’m testing each independently and in combinations, while tracking bridge-cell changes and avatar reachability.
- The control experiments reveal a reusable bridge mechanic: the first control rotates a color-8 bridge around its center; the second reconfigures several color-9 connector cells; the third flips two color-6/7 endpoint patterns. I’m now doing a bounded clone search over just four moves plus those three verified controls, with avatar-to-target distance as the dense progress signal.
- The direct-clone search hit the harness’s known deep-clone limit, so I switched to the documented path-replay strategy. The first search already verified a real subgoal: rotating controls 1 and 2 opens a route from the starting island down the vertical bridge, reducing target distance from 22 to 16.
- The replay search is still running within its fixed 6,000-state cap. I’m letting this bounded pass finish; if it exhausts, I’ll reduce the action graph using the verified bridge phases rather than widening the search.
- The reduced search exhausted 576 distinct `(avatar position, bridge phases)` states without reward. That is useful negative evidence: simply routing the discovered avatar to the color-11 marker is not the whole objective. I’m now checking partial-state changes and reachability relative to every special object, especially the color-15 object and the paired color-6/7 endpoints.
- A new mechanic is now verified: the color-6/7 endpoint control acts as a teleporter when the avatar is staged on its endpoint, connecting otherwise disjoint regions. The avatar can reach and overlap the color-15 object, but reward still does not fire. Because that overlap may collect or alter hidden-in-the-frame state, I’m restoring a cropped full-frame search key so the search preserves such partial progress instead of merging it away.

## Preserved local probes

- `probe_bfs.py` (2907 bytes)
- `probe_level3.py` (2280 bytes)
- `wip_glue_notes.md` (1168 bytes)

Run or inspect the smallest relevant probe before writing another one.
