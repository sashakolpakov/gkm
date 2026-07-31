# Unverified frontier brief: dc22 level 4

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside this workspace and begin with the indexed summaries, then reproduce any needed observations through bounded clone probes before editing the level-4 player.
- The baseline is reproducible: the current solver completes 3 levels in 119 moves. I’m now inspecting only the level dispatch, reusable legs, and the preserved bounded probe so I can reproduce the level-4 mechanics without expanding the search surface.
- The reproduced frame corrects an important ambiguity in the brief: level 4 begins with only one visible coordinate control, centered near `(51,19)`; the later controls are not yet exposed. The avatar is the 2×2 color-14 block at `(row 18, col 4)`, and actions 2/4 move it down/right while 1/3 are initially blocked. I’m refining the logical 2×2 tile map and testing that one visible control in isolation.
- The control scan found three genuine outcomes: one top control and two independently clickable halves of the lower control. The earlier bounded search omitted the lower-right half, explaining its tiny state space. I’ve corrected that observational action set and am running a fixed 6,000-state replay search with full gameplay-frame keys.
- The corrected search made genuine dense progress—moving the avatar from `(18,4)` to `(28,26)`—but replay reconstruction consumed the harness step budget before reaching reward. The path’s repeated lower-right clicks suggest turn-sensitive cooperation with the two color-1 objects, so I’m switching to a short, direct sequence probe that tracks those objects and the avatar after every action instead of widening BFS.
- The direct probe verifies the lower-left affordance is different from the lower-right one: each click translates both 2×6 color-1 platforms two pixels to the right, while the lower-right control constructs the two long crossings in four phases. The avatar is now staged on the far side of the upper crossing. I’m searching only from that verified subgoal, preserving the full gameplay frame so platform position and construction phase cannot be merged.

## Preserved local probes

- `probe_bfs.py` (2907 bytes)
- `probe_direct.py` (2044 bytes)
- `probe_level1_reward.py` (1135 bytes)
- `probe_level2.py` (4438 bytes)
- `probe_level2_context.py` (5036 bytes)
- `probe_level3.py` (2280 bytes)
- `probe_level3_context.py` (1830 bytes)
- `probe_level3_post15.py` (2142 bytes)
- `probe_observe.py` (3074 bytes)
- `probe_sequence.py` (2217 bytes)
- `wip_glue_notes.md` (5574 bytes)

Run or inspect the smallest relevant probe before writing another one.
