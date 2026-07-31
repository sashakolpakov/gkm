# Unverified frontier brief: su15 level 9

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace and use only the indexed harness/API surface. I’ll first read `solver_index.md`, then `frontier_brief.md` if present, run the baseline, and reproduce any level-9 observations with compact clone probes before editing the leg library.
- The brief records a plausible but explicitly unverified four-body/cutter theory. I won’t rely on it yet; I’ll reproduce the level start and action effects through the preserved compact probe and fresh bounded clones.
- The reproduced start confirms the coordinate-only surface, four independent color-7 bodies, two 2×2 solids, one 7×7 solid, and three ring targets. The preserved staging sweep also shows that naïvely moving solids lets autonomous bodies collide with or merge during transit, so the solution must coordinate seating and body steering; I’m now measuring each click’s exact one-turn effect.
- The first dense search is slower than the clone budget warrants and has not found even the verified split subgoal yet. I’m stopping that broad branch set and will constrain it to the body-control actions that directly change trajectories, then search split-first and completion-second.
- A direct attempted cut exposed the real timing constraint: after seating the small object, one autonomous body is already adjacent to the 7×7 solid and destroys it on the next turn. I’m shifting the dense subgoal earlier—first find a one-click trajectory change that leaves both solids intact through the fixed five-click staging route, then handle the split from that safe state.
- A safe prefix now exists: steer the lower-right body upward, then stage the 3×3 solid at the lower-left ring and the 7×7 solid at the lower-right ring. The constrained search has preserved both objects for two additional body turns; I’m pruning equivalent trajectory branches more aggressively to reach the split event without spending thousands of deep clones.
- The dense subgoal is clearer now: the four color-7 bodies combine pairwise into two color-14 bodies while the two seated solids remain intact. This matches a merge-and-deliver objective, not a cutting objective. I’ve verified a safe prefix through that pairwise merge; the remaining task is routing the two color-14 bodies together and then into the unoccupied ring.
- The safe prefix now deterministically reaches exactly two color-14 bodies with both final solids seated. A direct A* from that state is still clone-heavy, so I’m narrowing to the eight visible control points per body and prioritizing only moves that reduce their Chebyshev separation or produce the color-13 merge.

## Preserved local probes

- `probe_l9.py` (6618 bytes)
- `probe_l9_actions.py` (1633 bytes)
- `probe_l9_manual.py` (2149 bytes)
- `probe_l9_prestage.py` (4492 bytes)
- `probe_l9_search.py` (4623 bytes)
- `probe_l9_variants.py` (2585 bytes)
- `wip_glue_notes.md` (586 bytes)

Run or inspect the smallest relevant probe before writing another one.
