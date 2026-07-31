# Unverified frontier brief: su15 level 9

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly within the workspace and the documented harness surface. I’ll begin with `solver_index.md`, then use `frontier_brief.md` if present, and keep every level-9 experiment clone-only until the final player is ready.
- The baseline is still running after a one-time local plotting-cache initialization. Once it reports the current level, I’ll reproduce the brief’s safe prefix with the smallest preserved probe and inspect only its relevant source ranges.
- Fresh probes confirm the brief’s core mechanics: action `6` is coordinate selection, clicks on a colored body redirect that body while every body advances one turn, equal color-7 bodies merge into color 14, and the solids merge/move toward the rings. The dense target is therefore concrete: preserve two seated solids, reduce `4×color-7 → 2×color-14 → 1×color-13`, then seat color 13 in the remaining ring. I’m now tracing the preserved safe prefix and only branching over actual body control points.
- The preserved broad A* reproduces the exact safe state—two color-14 bodies at `(13,48)` and `(46,11)`, with color-15 seated lower-left and color-8 seated lower-right—but its 4,000-state clone expansion is too expensive. I’m stopping that branch and replacing it with a compact velocity-state search that tests only the eight meaningful control points and prioritizes the first verified color-13 merge.
- The first beam confirms a subtle but important constraint: locally reducing center distance can worsen the next turn because the unselected body advances on its autonomous trajectory. The search is progressing, but symbolic extraction dominates runtime. I’m keeping this run bounded; if it does not merge within a few more depths, I’ll cache one state extraction per child and narrow the beam around solid-preserving states.
- The narrowed search exposes the level’s cooperation timing: by turn 10, a moving color-14 body reaches the seated color-8 square before the two bodies can meet. That collision dislodges rather than irreversibly destroys the solid, matching the earlier “reseat while routing” mechanic. I’m allowing that temporary displacement and continuing toward the denser color-13 subgoal; the final leg will reseat both solids only after the merged body has passed.
- A useful simplification emerged: the color-14 merge path itself is deterministic and only 17 control clicks. Before spending more probes on solid recovery, I’m checking the actual sparse reward by routing the resulting color-13 body into the remaining ring. If the earlier two deposits are latched, that should complete immediately; if not, the exact post-merge frame will tell us which solid must be restored.
- The reward check did not complete because the level has a hard turn limit: the merged body reached `(41,23)` and the environment terminated before it could enter the left ring. This rules out the long post-merge detour. I’m testing the more promising target permutation now—put the large square in the upper-left ring and leave the lower-right ring for the color-13 body, which the verified merge already produces nearby.
- The target permutation is viable only if staging order changes: the preserved small-first route cannot carry the large square to the upper-left ring without an autonomous body striking it. I’m now testing the existing level-6 affordance—stage the large square first—across the 32 possible one-click body deflections, while keeping the lower-left small deposit and the lower-right final ring fixed.

## Preserved local probes

- `probe_l9.py` (6618 bytes)
- `probe_l9_actions.py` (1633 bytes)
- `probe_l9_beam.py` (3716 bytes)
- `probe_l9_layouts.py` (2067 bytes)
- `probe_l9_manual.py` (2149 bytes)
- `probe_l9_orders.py` (1973 bytes)
- `probe_l9_prestage.py` (4492 bytes)
- `probe_l9_route.py` (8261 bytes)
- `probe_l9_search.py` (4623 bytes)
- `probe_l9_variants.py` (2585 bytes)
- `wip_glue_notes.md` (586 bytes)

Run or inspect the smallest relevant probe before writing another one.
