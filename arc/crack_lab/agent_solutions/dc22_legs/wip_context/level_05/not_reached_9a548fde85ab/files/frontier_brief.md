# Unverified frontier brief: dc22 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and begin with the indexed surfaces, then reproduce any prior observations before relying on them.
- Baseline reproduced: four levels complete in 183 moves, so the missing work is exactly level 5. I’m now using the smallest preserved probe patterns to enter level 5 on a clone and summarize its objects and action effects.
- The first probes show six real controls with distinct mechanics: two horizontal shuttles, a two-axis movable platform, a cycling set of four bridges, and a direction/state toggle. Because these interact combinatorially, I’m switching to a bounded observational search keyed only by the gameplay area; its dense progress metric is the avatar’s newly reached connected regions, with reward used only as the terminal check.
- The bounded search is still running; I’ve capped it at 12,000 observational states and depth 100. Its output is buffered, so I’m letting the current bounded pass finish rather than widening it.
- The search reproduced a concrete first subgoal: one bridge-cycle click lets the avatar leave its starting pocket and reach `(30,28)` via `left, click E, left, up, up`. I’m staging there and narrowing the next probe to the bridge-cycle and direction-toggle controls, which are the only mechanics affecting this local corridor.
- At the staged subgoal, the four local controls alone cannot open a new region; including the two remote shuttles is necessary. The narrowed pass is progressing through those combined states and remains within its fixed 7,000-state cap.

## Preserved local probes

- `probe_bfs.py` (2907 bytes)
- `probe_bfs5.py` (2851 bytes)
- `probe_direct.py` (2044 bytes)
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
- `probe_level5.py` (1921 bytes)
- `probe_observe.py` (3074 bytes)
- `probe_sequence.py` (2217 bytes)
- `wip_glue_notes.md` (10566 bytes)

Run or inspect the smallest relevant probe before writing another one.
