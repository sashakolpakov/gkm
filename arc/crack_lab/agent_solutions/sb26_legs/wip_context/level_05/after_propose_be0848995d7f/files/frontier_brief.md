# Unverified frontier brief: sb26 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly within the workspace and begin from `solver_index.md`, then use only targeted ranges plus the documented harness/perception surface. I’ll reproduce any inherited observations on bounded clones before relying on them.
- Baseline reproduced: levels 1–4 complete in 54 moves, with level 5 undispatched. The preserved brief’s starting state is valid; next I’m reproducing its level-5 geometry and click effects through the smallest existing probes.
- The level-5 action model is now reproduced: action `6` is coordinate selection/painting, action `5` submits, all eight central dots are writable, and “filled slots / 8” is a valid dense progress measure. Earlier solved assignments are also reproduced; I’m using their diagram geometry to derive the traversal order before trying any level-5 permutation search.
- The visualization probe created five workspace PNGs that the harness correctly treats as taint. They are only my newly generated observations, so I’m removing those exact files before continuing; no solver or user-authored files are affected.
- The earlier leg cannot represent level 5: its contract requires one unique palette color per top symbol, while level 5 has nine displayed symbols, eight slots, and duplicate palette colors. I’m adding one minimal general leg for “arrange a visible palette multiset into diagram cells,” with duplicate branches collapsed and a hard cap of the exact 10,080 distinct assignments; the level player will only call that leg.

## Preserved local probes

- `probe_l5.py` (3146 bytes)
- `probe_level2.py` (4310 bytes)
- `probe_progression.py` (757 bytes)
- `probe_rule.py` (2739 bytes)
- `search_level2.py` (1722 bytes)

Run or inspect the smallest relevant probe before writing another one.
