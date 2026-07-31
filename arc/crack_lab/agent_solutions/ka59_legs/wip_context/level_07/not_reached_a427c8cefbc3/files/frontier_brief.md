# Unverified frontier brief: ka59 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace and use only the indexed harness/perception surface. I’ll first inspect `solver_index.md`, then reproduce any frontier observations before probing level 7, and keep the final player as leg composition only.
- The clean baseline reaches `levels=6` in 210 moves, so the current solver is correctly entering level 7 but not solving it. The frontier notes concern the prior level and don’t establish level-7 mechanics; I’ll now build a fresh symbolic level-7 observation from the documented clone interface.
- The fresh probe identifies three plausible movable pieces and three size-matched color-4 targets: a large ring `(13,13)→(13,40)`, a horizontal small ring near `(36,52)→(27,25)`, and a vertical small ring near `(55,45)→(28,48)`. Two color-13 corridor agents animate on every arrow command, and ordinary direct routes are blocked by sealed barriers. I’m tightening the probe to verified center-selection and a target-distance search over these persistent geometries.
- The bounded best-first probe is still running; it is searching only arrow moves plus the observed “select the other small ring” action, keyed by the visible 63-row world. Its dense score is the summed distance of the two selection markers and the large ring to their size-matched targets.
- The two small rings can be centered exactly, but that state has only 1,666 reachable visible configurations and none wins. This falsifies the “large color-11 ring moves downward” interpretation: the size-matched pair is instead the large color-4 ring below and the fixed color-11 outline above. I’m restarting the bounded search with the large color-4 ring’s upward distance as the third progress term, so it can discover the required push before the small rings become trapped on their targets.

## Preserved local probes

- `probe_l4.py` (3341 bytes)
- `probe_l5.py` (3037 bytes)
- `probe_l6.py` (5807 bytes)
- `probe_l7.py` (2431 bytes)
- `probe_l7_focus.py` (1670 bytes)
- `probe_l7_search.py` (4160 bytes)
- `probe_l7_select.py` (1549 bytes)

Run or inspect the smallest relevant probe before writing another one.
