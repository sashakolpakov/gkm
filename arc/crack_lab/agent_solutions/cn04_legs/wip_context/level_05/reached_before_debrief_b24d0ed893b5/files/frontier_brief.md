# Unverified frontier brief: cn04 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the clean-room surface, begin with `solver_index.md`, then inspect only targeted ranges and reproduce any frontier observations before relying on them.
- Baseline is confirmed at level 5 (`levels=4`). I’m now mapping the raw level through compact blob/action summaries; no prior observation file was present to trust or reproduce.
- The key mechanic is now experimentally separated: actions 1–4 translate the selected rigid figure by three pixels, action 5 rotates it, and coordinate action 6 transfers selection. Level 5 has three movable figures plus fixed geometry; I’m checking the prior level’s reward transition to pin down whether the goal is connectivity or a specific placement before planning.
- The prior-level transition confirms a frame-native dense signal: the 32-pixel bar at row 0 changes color as placements improve. A first arbitrary assembly changed 6 pixels without reward, so I’ll optimize that exact progress count on bounded clones, one selected figure at a time, and verify the final reward independently.
- The bar is a move counter, not progress—I rejected that hypothesis after reproducing its regular increment under irrelevant repeated moves. The verified level-4 win geometry instead supports the real dense measure: the number of spatially disconnected figure groups. I’m now searching only for placements that reduce that count, then checking reward.
- A second falsification sharpened the objective: merely making one connected group does not win. In the verified level-4 solution, occupied area drops from 495 to 378 before the last move, proving that deliberate figure overlap—not simple contact—is the useful dense measure. I’m switching the bounded search to maximize overlap while preserving reward checks.
- The overlap search is still running within its fixed state cap; it has not produced a candidate yet. I’m letting this bounded pass finish, then I’ll replay only its best symbolic paths.

## Preserved local probes

- `probe_connectivity.py` (1756 bytes)
- `probe_goal.py` (1035 bytes)
- `probe_level5.py` (1641 bytes)
- `probe_optimize.py` (1585 bytes)
- `probe_paths.py` (1381 bytes)

Run or inspect the smallest relevant probe before writing another one.
