# Unverified frontier brief: s5i5 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside this workspace and begin with the indexed surface, then reproduce any prior observations before relying on them. After locating level 6, I’ll use compact clone probes and keep all player logic as leg composition.
- Baseline is reproduced at `levels=5`, so the task is cleanly isolated to level 6. I’m now reusing the prior probe pattern only as scaffolding, while generating fresh level-6 observations from the documented clone surface.
- Level 6 is a coordinate-only mechanism with three coupled “large” controls and six directional sub-controls. The movable assembly contains the only attached color-13 cell; the detached four-cell plus near `(52,34)` is the likely target. I’m measuring dense progress by the attached marker’s distance to that target and probing each control’s repeat trajectory before searching combinations.
- The clone search has now produced and reward-verified a 45-click solution. Its dense progress reached the marker directly above the target before the final docking click. I’m tightening that path with a breadth-first pass over the same 7k-state reachable space, then I’ll encode the resulting staged arm-threading mechanic as one reusable leg.
- The shortest-path pass is still running within the bounded state cap; clone construction is the slow part, not an unbounded search. The already verified solution remains available, so this optimization won’t block implementation if it yields no improvement.
- Implemented and verified level 6. - Added reusable `dock_three_link_arm_through_partitioned_chamber` leg in [legs.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_s5i5_budgeted_medium_screen/legs.py). - Added `play_level_6` as leg-only composition in [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_s5i5_budgeted_medium_screen/players.py). - Preserved compact level-6 probe and bounded search scripts. Validation: ```text RESULT levels=6 moves=173 replay_ok=True err=None ```

## Preserved local probes

- `probe_level3.py` (2988 bytes)
- `probe_level4.py` (3456 bytes)
- `probe_level5.py` (3516 bytes)
- `probe_level6.py` (3311 bytes)
- `search_level4.py` (2167 bytes)
- `search_level5.py` (3054 bytes)
- `search_level6.py` (2339 bytes)

Run or inspect the smallest relevant probe before writing another one.
