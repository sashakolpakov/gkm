# Unverified frontier brief: dc22 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and begin with the indexed interfaces, then reproduce any frontier observations with compact clone probes before editing the level-5 composition.
- Baseline is confirmed: `levels=4`, `moves=183`, and replay is valid. I’ll now inspect only the level-5 probe and the documented harness/player ranges, then rerun that probe to validate its observations rather than trusting the brief.
- The reproduced action map matches the brief: actions 1–4 move the color-14 avatar by two pixels when unblocked; coordinate controls A/B shuttle lower bridges, C/D translate the large platform, E cycles four local bridges, and F toggles four paired tiles. The prior narrowed BFS is already bounded to 7,000 compact states, so I’m rerunning it to verify the staged reachability result and see whether it yields a complete path.
- The reproduced staged search confirms the avatar reaches several nearby cells, but no new region yet; it has explored about 4,000 of its fixed 7,000-state budget. I’m letting this bounded pass finish, then I’ll use its exact failure frontier to reduce the control/state search rather than widening blindly.
- The 7,000-state pass ended cleanly and confirms its limitation: it only saturates the starting/adjacent pocket. The observed goal-colored object is at logical cell `(25,5)`, southwest of the avatar at `(17,16)`, so I’m switching to a bounded best-first clone search whose dense score is avatar distance to that object while retaining all six verified controls. This directly implements the requested sparse-reward strategy.
- The best-first probe has improved the dense metric from 38 to 30 pixels and found the verified route into the left edge of the starting complex (`(36,26)`), but it is now exploring a plateau rather than claiming success from visual change alone. The cap remains 25,000 states; if it does not break southward, I’ll inspect the plateau’s control-state geometry and retarget the next reachable waypoint.
- The best-first run exposed a search-implementation issue before reaching its cap: repeated shorter-depth updates created duplicate queue entries while the unique observational graph stalled near 9,600 states. I stopped it rather than spending clone budget on duplicates. I’m now checking whether the six initial controls are sufficient or whether a later state reveals another coordinate affordance—the most likely reason a complete graph over the initial controls would still fail.
- The deduplicated run is behaving correctly now: every observational state is expanded at most once, and it has passed 6,000 unique states without clone-chain duplication. Its dense best remains `(36,26)`; I’m waiting for a definitive terminal path or finite-graph exhaustion before changing the action surface.

## Preserved local probes

- `probe_astar5.py` (2948 bytes)
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
- `probe_map5.py` (1900 bytes)
- `probe_observe.py` (3074 bytes)
- `probe_sequence.py` (2217 bytes)
- `wip_glue_notes.md` (10566 bytes)

Run or inspect the smallest relevant probe before writing another one.
