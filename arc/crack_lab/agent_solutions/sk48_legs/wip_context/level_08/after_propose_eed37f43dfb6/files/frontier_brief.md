# Unverified frontier brief: sk48 level 8

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Verified parent budget

- Exact parent boundary: level 7 at 584 actions.
- Remaining real-action budget under the harness cap: 16 of 600.
- This budget is verifier evidence. If it is insufficient, optimize earlier composed legs rather than searching an uncommittable suffix.

## Prior proposer progress

- At suffix depth 9, horizontal residual is down from 9 to 5 with the vertical train still exact. The best branch is the transposed analogue of the earlier successful horizontal shuttle, so I’m continuing it.
- The progress is exactly transposed from the earlier horizontal-complete run. I’ll reuse the verified mapped suffixes rather than re-search them: shuttle the remaining row inward, retract it, form `9→14`, then bring that pair one lane from color-6. This reaches the symmetric near-goal state at move 55 with nine moves left.
- At move 55 the completed vertical train is intact, and `9→14` is already compact one row above color-6. The horizontal head only needs its three-cell arm fully retracted, then one upward lane move should join the adjacent pair. I’m testing this five-action finish.
- The five-action finish falsifies the upper-side state: color-6’s highest reachable row is one lane below `9→14`. I’ll return to move 32 and add the symmetric reachability constraint—keep both horizontal tokens on or below color-6—before optimizing adjacency.
- No branch has yet moved color-9 to the reachable lower side while preserving the complete vertical train through depth 7. I’m continuing until that reachability count changes; if it cannot, the vertical train itself must briefly be loosened during the handoff.
- The constraint cannot improve while `8→12` is held rigidly exact, so preserving that train is over-constraining the handoff. I’ll prioritize moving both horizontal tokens to color-6’s reachable lower side first, allowing the vertical train to loosen temporarily, then restore it.
- At depth 5, the relaxed search succeeds in the key reachability subgoal: both `9` and `14` are now on color-6’s lower/reachable side. The vertical train is temporarily loosened, as expected. I’ll continue from this viable topology.
- By depth 8 the search has both horizontal tokens on the reachable side and has already restored the exact `8→12` vertical train. This is the correct topology; only the horizontal residual remains.
- At suffix depth 12, the viable lower-side topology is preserved, `8→12` remains exact, and horizontal residual improves to 6. I’ll reseed directly at this 44-move state, leaving a 20-move search horizon.
- From the viable state, five more moves reduce horizontal residual to 5 while keeping both reachability and the vertical train exact. The branch remains monotone.
- At suffix depth 10, one horizontal collector edge is exact: color-9 is attached to color-6, while color-14 remains to be added. The vertical train and lower-side reachability remain intact.
- The run hit its wall-time cap at a useful state: after 55 level-8 moves, `8→12` is exact, `9` is attached to color-6, and color-14 is three lattice steps away on the reachable side. I’ll reseed there and exhaust the final nine legal moves.

## Preserved local probes

- `measure_levels.py` (3655 bytes)
- `probe_level8.py` (3775 bytes)
- `search_level8.py` (4880 bytes)
- `test_prefix_deletions.py` (2673 bytes)
- `trace_level8.py` (3148 bytes)

Run or inspect the smallest relevant probe before writing another one.
