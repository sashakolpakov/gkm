# Unverified frontier brief: sk48 level 4

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the clean-room surface, begin with `solver_index.md`, then reproduce any prior observations from `frontier_brief.md` before probing level 4. I’ll keep experiments clone-bounded and only commit a leg-composed player once the mechanic is verified.
- Baseline is reproduced exactly: level 4 is reached after 133 moves with replay validation intact. I’m now checking the existing level-4 probe and leg conventions, then I’ll rerun only the compact experiments needed to verify avatar motion, action semantics, and token-order progress.
- The preserved probe is valid but its clone BFS is too expensive for the requested budget—it has not yielded within a minute. I’m stopping that search and will replace it with a replay-based, compact symbolic search so clone depth does not accumulate.
- The split transition is now independently verified step by step: moving to row 13 while retracting at the row-19 barrier leaves token `12` behind, proving the maze reorders by selectively detaching tokens. A 2,500-state breadth search reached a useful four-row staging state but not the win; I’m switching to best-first search using an explicit distance-to-`8,12,9,14` train metric, which should spend the remaining probe budget on physically meaningful states.
- The dense search has reduced the target distance from 10 to 4 within about 1,000 unique physical states. Its best state has the desired suffix `9,14` already adjacent and has isolated `8` and `12` in separate barrier rows, so the metric is steering toward the intended staged assembly rather than mere frame changes.
- The closest staged state is reproducible: all four tokens are on row 19 at columns `9@24, 8@30, 12@36, 14@48`. That is one displaced token plus one gap from completion, and I’m searching locally from this verified checkpoint instead of repeating the full state tree.
- The local basin is larger than expected and has not improved the distance yet. I’ll let this bounded run reach its cap, then inspect the cheapest frontier transition; if it remains flat, I’ll adjust the dense measure to reward correct adjacency/order rather than raw Manhattan distance.
- The map exposes the key correction: level 4 has two vertical barriers, creating three token regions only in the upper half. The displayed request is also spatially split into `[8,12]` and `[9,14]`, so “one contiguous four-token train” may be the wrong dense goal. I’m comparing the exact pre-reward states of levels 1–3 to determine whether reward checks token order alone or the displayed grouping/placement.
- The corrected pair-placement metric behaves much better. It quickly found a state with `8,12` adjacent on the left and `14` parked at the far right; only `9` remains on the wrong side. This supports the split-pair objective inferred from the reward examples.
- The search is now exploring how to ferry the remaining `9` past the left barrier without disturbing the completed pair. I’m keeping the run capped at 8,000 unique states; if it does not finish, the preserved near-goal path gives a focused subproblem for one final local search.

## Preserved local probes

- `probe_level2.py` (2280 bytes)
- `probe_level3.py` (5753 bytes)
- `probe_level4.py` (4414 bytes)
- `probe_level4_trace.py` (3428 bytes)
- `wip_glue_notes.md` (293 bytes)

Run or inspect the smallest relevant probe before writing another one.
