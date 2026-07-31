# Unverified frontier brief: su15 level 9

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside this workspace and begin from `solver_index.md`, then reproduce any prior observations through bounded clone probes before changing the solver.
- The baseline run is still starting up (Matplotlib is building a local cache). There is no `frontier_brief.md`, so I’ll derive level 9 from fresh observations only.
- Baseline reliably reaches the level-9 start (`levels=8`). I’m now adding a compact observational probe that runs the existing solver to that point, reports blobs and color counts, and tests both opaque actions on clones at representative coordinates.
- The first coordinate sweep is too broad for the clone cost at this depth, so it hasn’t yielded useful output yet. I’m bounding the probe more tightly to object centers and a few empty cells, which should expose action semantics with far fewer cloned steps.
- Fresh level-9 evidence: it is coordinate-only (`actions=(6,)`), and every effective click advances autonomous objects. The playfield contains three 9×9 ring targets, one 7×7 solid square, and three small moving bodies. I’m comparing those body trajectories and the square’s response to clicks, then I’ll decide whether the existing general merge/cutter leg already matches.
- The mechanics are now reproducible: clicking a 2×2 color-6 square merges the pair into a 3×3 color-15 square; clicking elsewhere advances four independently moving color-7 bodies; clicking within a body changes that body’s trajectory. This is a new four-body configuration, so I’m using a shallow symbolic search to find verified body merges before encoding a reusable leg.
- The first two body merges are verified in three clicks total, reducing four color-7 bodies to two color-14 bodies. The next replay-style search is reconstructing too many paths, so I’m stopping it and switching to a node-retaining shallow BFS for this short stage.
- The dense search found the key chain: four color-7 bodies merge into two color-14 bodies, then into one color-13 body. Along that verified trajectory, the merged body cuts the 7×7 square to 5×5 and both 2×2 squares to 1×1; the three ring targets disappear, strongly indicating level completion. I’m confirming the reward transition, then I’ll encode this as one adaptive reusable leg.
- Reward did not fire: merging all bodies first is a dead end because the resulting color-13 cutter destroys the unseated squares and ring structure. The earlier level-8 leg confirms the intended pattern—seat solid objects first, then merge and route moving bodies while protecting seated objects. I’m testing the six assignments of the two solids to the three rings before any body merge.

## Preserved local probes

- `probe_l9.py` (6549 bytes)
- `wip_glue_notes.md` (586 bytes)

Run or inspect the smallest relevant probe before writing another one.
