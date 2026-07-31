# Unverified frontier brief: sb26 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace and use only the indexed ranges plus the documented harness/perception surface. I’ll first reproduce the current level state, then probe level 5 symbolically on clones, implement the smallest reusable leg, and verify the real run reaches at least five completed levels.
- The baseline is reproduced: levels 1–4 complete in 54 moves, and level 5 is the first undispatched state. I’ll now validate the preserved level-5 observations with the smallest probes before trusting their proposed mechanic.
- The mechanic is now reproduced: `6` only acts with coordinates, palette clicks select a tile, a second click paints any of the eight central dots, and `5` is a submit/check action. The central dot count gives a dense measure from 0/8 to 8/8. Importantly, duplicate palette colours are visually distinct tiles, so I’m checking their pixel signatures rather than collapsing them by colour.
- The reusable leg is in place. It detects the palette and equal-sized diagram group from blobs, experimentally groups visually identical tiles by what they paint, and searches only the 10,080 distinct multiset arrangements. `play_level_5` contains only that leg call, as required. I’m running the full replay validation now.
- The exhaustive clone search is still running normally; no error has surfaced. The search is bounded by the exact number of visually distinct assignments, so this run will either produce a replay-validated level 5 or a precise exhausted-candidate failure.
- The run remains within the bounded search and is taking longer than the earlier seven-tile levels, which is expected for the eight-tile multiset. I’m letting this decisive run finish rather than restarting it and losing the accumulated search.
- The first bounded hypothesis has been falsified cleanly: none of the 10,080 final grids using the palette multiplicities wins. That means the duplicate buttons are not a consumable multiset constraint; they are reusable choices or occurrence markers. I’m now probing what a failed full submission reveals and testing code-derived fills, then I’ll narrow the larger state space from that evidence.

## Preserved local probes

- `probe_diagrams.py` (967 bytes)
- `probe_l5.py` (3146 bytes)
- `probe_level2.py` (4310 bytes)
- `probe_progression.py` (757 bytes)
- `probe_rule.py` (2739 bytes)
- `probe_submit_feedback.py` (1463 bytes)
- `search_level2.py` (1722 bytes)

Run or inspect the smallest relevant probe before writing another one.
