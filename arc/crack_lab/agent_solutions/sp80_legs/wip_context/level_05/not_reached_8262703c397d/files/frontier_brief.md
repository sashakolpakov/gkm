# Unverified frontier brief: sp80 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace and the documented harness surface. I’ll begin with `solver_index.md`, then `frontier_brief.md` if present, and use compact clone probes before changing the leg library.
- Baseline is confirmed: level 4, 79 moves, replay-valid. I’ll now reproduce the level-5 observations from the preserved compact probe and inspect only the relevant harness/player/leg ranges needed to understand how it advances and records actions.
- The reproduced evidence sharpens the mechanic: action 6 selects one of four pieces, arrows translate it by 3 pixels, and USE commits. Three sockets are above the playfield and one is left of it, so level 5 plausibly constrains both column and row projections. I’m comparing the known winning layouts of levels 2–4 against their socket geometry to derive that projection rule before searching.
- The known levels confirm the useful dense measure: each bar’s endpoint cells align with socket legs/openings; the winning layouts are projection layouts, not physical contact. I’m sampling a bounded set of alternate known wins now to separate necessary endpoint alignments from incidental choices in the existing fixed solutions.
- The alternate level-2 wins reveal the actual dense objective: the union of movable projections must continuously cover the span between the inward prongs of the outer sockets; surplus pieces may sit anywhere. Level 5 already covers the horizontal span at its starting positions, so I’m testing the new left-edge socket as the analogous vertical coverage constraint.
- A direct level-3 check shows coverage alone is insufficient; the fixed solution’s outer pieces are materially contributing. I’m now perturbing one known winning piece at a time while holding the others fixed. This gives the exact positional tolerances with only a few dozen clone commits and avoids a broad level-5 search.

## Preserved local probes

- `probe_known_sensitivity.py` (2093 bytes)
- `probe_level5.py` (5404 bytes)
- `probe_level5_hypotheses.py` (2004 bytes)
- `probe_projection_rule.py` (1827 bytes)
- `probe_win_sets.py` (1949 bytes)

Run or inspect the smallest relevant probe before writing another one.
