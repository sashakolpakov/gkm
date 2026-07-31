# Unverified frontier brief: lp85 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay within the current workspace and use only the indexed harness/API surface. I’ll first inspect `solver_index.md`, then reproduce any useful frontier observations with compact clone probes before editing the leg library and level-5 player.
- The baseline now reproducibly reaches the level-5 frontier (`levels=4`, 41 moves). The inherited brief described the old level-4 search and is no longer directly relevant; I’ll use the preserved probe style only to obtain a clean level-5 clone, then summarize its objects and click effects symbolically.
- The level-5 frame is a single 20-slot “snake” of persistent colored tiles with two pairs of clickable arrow-like controls: the upper pair changes a small region, while the lower pair changes most of the snake. Each click also advances a one-pixel move meter, so indiscriminate real play would consume the budget. I’m now learning the two reversible permutations on clones and testing explicit structural scores on the tile sequence.
- The learned controls are fixed inverses: one rotates the five outlined tiles, and the other rotates the entire 21-tile serpent. The outlines isolate the actual work area. Only color `1` occurs at least five times, so the feasible dense goal is to load five `1` tiles into those outlined slots (initial progress `1/5`), rather than symmetry across the storage rows. I’m searching that compact permutation model within the visible move budget and will replay the candidate on a fresh clone for the real reward.
- The five-of-a-kind arrangement is attainable in 14 moves but does not raise reward, so that hypothesis is falsified on a clone. I’m returning to the stronger inherited family invariant—matching opposite positions in every horizontal motif—and quotienting the search by global color renaming, since the goal depends on equality patterns rather than color identities. This keeps the bounded search compact while preserving reward-relevant structure.
- Both attractive visual hypotheses are now decisively wrong: a clone with five identical outlined tiles and a clone with all six opposite pairs matched each remain at level 5. I’m going to reproduce the actual rewarded terminal patterns from levels 3 and 4, whose controls are the same overlapping rotations, then transfer only the invariant that is visibly present at those real reward transitions.

## Preserved local probes

- `model.py` (1357 bytes)
- `model_l3.py` (4322 bytes)
- `model_l4.py` (6909 bytes)
- `model_l5.py` (2970 bytes)
- `p1.py` (544 bytes)
- `p2.py` (429 bytes)
- `p3.py` (1572 bytes)
- `p4.py` (1549 bytes)
- `p5.py` (691 bytes)
- `p6.py` (966 bytes)
- `p7.py` (555 bytes)
- `p8.py` (514 bytes)
- `probe_l3.py` (2908 bytes)
- `probe_l4.py` (2646 bytes)
- `probe_l5.py` (3217 bytes)
- `probe_level2.py` (2244 bytes)
- `probe_offline.py` (3762 bytes)
- `probe_pair_l3.py` (2049 bytes)
- `probe_search.py` (2182 bytes)
- `reward_l3.py` (1384 bytes)
- `rewarded_l4.py` (2078 bytes)
- `search_l3.py` (2330 bytes)
- `search_l4.py` (1832 bytes)

Run or inspect the smallest relevant probe before writing another one.
