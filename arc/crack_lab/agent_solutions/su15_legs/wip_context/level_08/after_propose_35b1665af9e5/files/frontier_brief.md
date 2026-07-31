# Unverified frontier brief: su15 level 8

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace and the raw harness surface. I’ll begin with `solver_index.md`, then use `frontier_brief.md` if present, reproduce any observations I depend on, and keep probes compact and clone-bounded.
- The baseline is reproduced: the solver completes 7 levels in 133 moves with a valid replay. I’m now using the smallest preserved level-8 probes to independently verify the object/action model and the near-goal metric before changing solver code.
- The preserved probe is slow because it replays the full seven-level solver before cloning level 8; it has not produced observations yet. I’m keeping it bounded and will stop it if it remains stuck, then use the validated checkpoint through `gkm_try.py` in a smaller perception-based probe.
- The reproduced level-8 root has coordinate action `6` only (action `7` is not exposed in this level), four 9×9 corner rings, one 7×7 color-8 square, and three 8-pixel color-7 bodies. I hit the unavailable-action guard in the first probe, so I’m removing that invalid branch and rerunning only the documented action set.
- The coordinate experiment confirms two interacting systems: a nearby click relocates the 7×7 square, while each ordinary turn advances all three color-7 bodies diagonally; clicking a body can hold or redirect that body while the other two advance. The preserved 19-click prefix independently reaches the claimed near-state: square staged in the upper-left ring and 21 of 24 body pixels on original ring cells. I’m now resolving the last three pixels by enumerating the exact near-state body shapes and distinct click outcomes, rather than broad raw-frame search.
- The bounded diverse beam reproduced the previous 23/24 state at depth 8, with the square still staged; it is exploring only 40 symbolic states per depth and is rate-limited to 300 clone steps/s. I’ve captured the exact suffix, so even if the current frontier does not finish, the next probe can start from this stronger verified state.
- The 23/24 state is now fully characterized: the bottom two bodies are exact 8/8 fits, while the top-right body has one west-pointing pixel just outside its ring. The last three verified moves nearly synchronize all three, so I’m exhaustively testing only those final three control turns from the earlier 18/24 state; this is a small, exact timing check rather than a broader heuristic search.

## Preserved local probes

- `probe.py` (2087 bytes)
- `probe_beam8.py` (3293 bytes)
- `probe_beam_clean8.py` (4425 bytes)
- `probe_bfs23.py` (2174 bytes)
- `probe_clean8.py` (5740 bytes)
- `probe_dots.py` (903 bytes)
- `probe_finish8.py` (4381 bytes)
- `probe_frontier23.py` (3109 bytes)
- `probe_level4.py` (5557 bytes)
- `probe_level5.py` (1909 bytes)
- `probe_level5_sequences.py` (2002 bytes)
- `probe_level6.py` (1230 bytes)
- `probe_level8.py` (2994 bytes)
- `probe_patterns8.py` (913 bytes)
- `probe_root8_actions.py` (1824 bytes)
- `probe_route.py` (881 bytes)
- `probe_search.py` (2205 bytes)
- `probe_search8.py` (4984 bytes)
- `symbolic_probe_l2.py` (2882 bytes)

Run or inspect the smallest relevant probe before writing another one.
