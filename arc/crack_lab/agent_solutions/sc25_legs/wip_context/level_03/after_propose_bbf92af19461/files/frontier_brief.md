# Unverified frontier brief: sc25 level 3

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and use only the indexed harness/perception surface. I’ll first inspect `solver_index.md`, then `frontier_brief.md` if present, and reproduce any observations I rely on with bounded symbolic probes.
- The baseline reliably reaches level 3’s start (`levels=2`, 22 moves), so the missing piece is isolated to `play_level_3`. I’m now reading only the indexed player/leg bodies and the documented harness entry points needed to construct a level-3 clone.
- The first experiment identifies the avatar unambiguously: the adjacent color-10/color-9 4×4 object at `(22–25,35–38)` moves left/right/up/down under actions `1/2/3/4`; bare non-coordinate action `6` is not part of that movement probe. The board also contains a familiar 3×3 coordinate panel with three black cells, so I’m testing that existing selection leg separately before exploring movement paths.
- The three black panel clicks are visually inert, and a 5,000-state movement-only search finds no reward, so completion needs a context-sensitive interaction or staging step. I’m switching to compact targeted probes: direct routes toward each salient object and coordinate action `6` on each object/panel, measuring reward plus object/color-count changes rather than raw pixels.
- The coordinate mechanic is now mapped: the lower 3×3 grid (`x≈25/30/35`, `y≈50/55/60`) is clickable, and each cell writes the corresponding upper 3×3 output cell. This exactly matches the existing `select_grid_cells_of_color` leg; my earlier inert clicks were on the output panel. I’m reproducing the existing leg on a fresh clone, then searching only the post-selection movement phase.
- The grid leg correctly clears the transposed black output row, but no reward follows immediately. Movement colors reveal that the avatar reorients on every move (color 9 becomes the leading half), so orientation at contact matters. I’m replacing deep-clone search with the scaffold’s replay-based BFS and an observation key that excludes only the visible move-budget bar; this preserves avatar pose and all puzzle objects while avoiding duplicate states caused solely by elapsed turns.
- A structural detail now matters: the reachable floor ends at `x=33`, a color-13 block occupies `x=34–36,y=27–30`, and the socket lies immediately to its right. The earlier “toward obstacle” route stopped adjacent without issuing the push. I’m testing the explicit subgoal path to stand at `(30,27)`, orient right, and push the block into the socket—both before and after the verified grid-selection leg.

## Preserved local probes

- `probe_level2.py` (7657 bytes)
- `probe_level2_interactions.py` (4048 bytes)
- `probe_level3.py` (4545 bytes)

Run or inspect the smallest relevant probe before writing another one.
