# Unverified frontier brief: dc22 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace and use only the indexed harness/API surface. I’ll begin with `solver_index.md`, then reproduce any inherited observations from `frontier_brief.md` before relying on them.
- The brief contains only unverified level-5 hypotheses. I’m now establishing the real baseline, then I’ll inspect and rerun the smallest level-5 probes that generated those claims.
- The action map and initial geometry are now reproduced: the avatar is the 2×2 color-14 block at logical `(17,16)`; actions 1/2/3/4 are up/down/left/right when passable; the six coordinate controls produce the bridge/platform transformations shown by the compact map. The visible dense target is the color-11 tile at `(25,5)`, but it is separated by staged corridors, so I’m inspecting the preserved bounded search and its exact state key before deciding whether to repair or replace it.
- The deduplicated best-first run is still active but has emitted no buffered frontier yet. While it remains within its fixed 25,000-state cap, I’m treating it only as a verifier; the next probe will target staged reachability and teleport activation directly if this pass exhausts.
- A more plausible mechanic emerges from the reproduced geometry: control E moves an entire 2×2 bridge between the starting pocket and the western pocket, so toggling it while the avatar occupies that bridge may shuttle the avatar across the barrier. I’m testing that directly, along with F only after reaching a color-6 endpoint; this is the context-sensitive check the global search was obscuring.
- E does not carry the avatar, and the verified start pocket is genuinely closed under the six assumed controls. That falsifies the prior “complete action surface” assumption. I’m now scanning the visible 64×64 frame at bounded 2-pixel resolution for every coordinate interaction that causes a real delta, including gameplay objects—not just the right-side panel.
- The scan confirms exactly six clickable regions—no hidden seventh control. The remaining causal object is the large color-8 assembly: C and D move it only upward/right toward the color-15 targets at row 9. I’m treating target coverage as the next dense subgoal and testing order-sensitive C/D placements for state changes or newly revealed connectivity.

## Preserved local probes

- `probe_astar5.py` (2948 bytes)
- `probe_bfs.py` (2907 bytes)
- `probe_bfs5.py` (2851 bytes)
- `probe_clickscan5.py` (1191 bytes)
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
- `probe_map5.py` (1900 bytes)
- `probe_observe.py` (3074 bytes)
- `probe_platform5.py` (1588 bytes)
- `probe_sequence.py` (2217 bytes)
- `probe_stage5.py` (1117 bytes)
- `wip_glue_notes.md` (10887 bytes)

Run or inspect the smallest relevant probe before writing another one.
