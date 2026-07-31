# Unverified frontier brief: lp85 level 3

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and begin with the indexed API surface. I’ll reproduce any inherited observations before relying on them, then use compact clone probes and keep the final player as leg composition only.
- The baseline is level 2 completed in 13 moves, so level 3 is the only missing dispatch. I’m now inspecting just the player/leg definitions and the harness entry points needed to obtain and clone the level-3 state.
- The first clone experiment found exactly two coordinate affordances: two bottom controls centered near `(23,41)` and `(35,41)`. Clicking ordinary board pieces does nothing. I’m treating these as context-dependent actions and will compare repeated and alternating sequences, tracking the board’s symbolic token layout and reward rather than raw pixel dumps.
- The frame shows two mirrored articulated figures with one control beneath each; each click advances a pose/state and consumes one pixel of a finite top budget. A concrete dense measure is therefore left/right mirror agreement across corresponding colored joints. I’ll search the compact two-action state graph while reporting only improvements in that agreement and checking the actual level reward.
- The bounded search is still running; the arena’s deep clones are slower than expected, so I’m holding the current cap and will stop or tighten it if the next yield does not finish. No real-environment moves are being committed by this probe.
- The first search confirmed mirror agreement is meaningful: it rose from 3/14 to 8/14, but recursive clones became prohibitively deep. I stopped it cleanly. I’m switching to path replay from a shallow level-3 root and a narrow beam ranked by that dense measure, which respects the scaffold’s guidance for expensive clones.
- The controls are not independent: they rotate overlapping joint loops, explaining why order matters. Rather than spend clone budget on a wider black-box tree, I’m inferring the fixed board permutations from a small set of varied contexts. If validated, that gives a fast symbolic model for planning while reserving arena clones only for reward verification.
- The inferred model validated perfectly in three unseen contexts. It is two overlapping 16-position ring rotations (the classic interlocking-rings geometry), sharing two center positions. That also sharpens the dense goal: the remaining 28 positions form 14 mirror pairs, while the two shared positions can hold the odd-count colors. I’ll now search this exact symbolic model for full 14/14 symmetry, then verify the resulting sequence on a clone.
- Full mirror symmetry is not the win condition—the exact model disproved it through depth 18, though agreement still served as useful partial structure. The top indicator appears to leave only a small action budget, so I’m measuring the precise terminal horizon first; if it is short, I can exhaustively verify every legal action sequence on shallow clones without guessing the hidden goal.

## Preserved local probes

- `model.py` (1357 bytes)
- `model_l3.py` (4322 bytes)
- `p1.py` (544 bytes)
- `p2.py` (429 bytes)
- `p3.py` (1572 bytes)
- `p4.py` (1549 bytes)
- `p5.py` (691 bytes)
- `p6.py` (966 bytes)
- `p7.py` (555 bytes)
- `p8.py` (514 bytes)
- `probe_l3.py` (2908 bytes)
- `probe_level2.py` (2244 bytes)
- `probe_offline.py` (3762 bytes)
- `probe_search.py` (2182 bytes)
- `reward_l3.py` (1384 bytes)
- `search_l3.py` (2330 bytes)

Run or inspect the smallest relevant probe before writing another one.
