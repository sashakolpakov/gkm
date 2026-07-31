# Unverified frontier brief: su15 level 8

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside this workspace and the documented harness surface. I’ll begin with `solver_index.md`, then the frontier brief if present, run the baseline, and use compact bounded clone probes before editing the leg library/player.
- The baseline is confirmed: 7 levels, 133 moves, valid replay. I’m now reproducing the level-8 root and near-goal claims from the smallest preserved probes, treating the brief itself as untrusted.
- The root model is now independently confirmed: one coordinate action, a movable 7×7 square, and three 8-pixel autonomous bodies; every click advances the bodies, while clicking a body redirects/holds it. The existing prefix stages the square and reaches 21/24 body-ring overlap, and a bounded suffix reaches 23/24. I’m checking a small exact timing frontier from the stronger staged state.
- The first compact beam confirms the 23/24 state but not completion. One important modeling risk surfaced: the visible playfield state may omit the bodies’ current travel directions, so deduplicating only by body pixels can merge behaviorally different states. I’m checking the compact non-playfield indicators before widening the search.
- The control probe resolved the mechanic cleanly: each body’s eight pixels are directional controls. Clicking one sends that body immediately in the corresponding cardinal/diagonal direction; unclicked bodies continue on their prior headings. I corrected the search metric to exact body-pixel containment in distinct rings (not approximate center distance); the staged prefix is exactly one pixel of translation away, though timing still matters.
- The exact search is now widened while keeping the clone rate capped. I’ve also removed the only “blank” candidate because it moves the already-staged square; every retained branch now preserves that completed subgoal and varies only body direction controls.

## Preserved local probes

- `probe.py` (2087 bytes)
- `probe_beam8.py` (3293 bytes)
- `probe_beam_clean8.py` (4425 bytes)
- `probe_bfs23.py` (2174 bytes)
- `probe_body_controls8.py` (1396 bytes)
- `probe_clean8.py` (5740 bytes)
- `probe_dots.py` (903 bytes)
- `probe_exact_search8.py` (4032 bytes)
- `probe_finish8.py` (4381 bytes)
- `probe_frontier23.py` (3109 bytes)
- `probe_level4.py` (5557 bytes)
- `probe_level5.py` (1909 bytes)
- `probe_level5_sequences.py` (2002 bytes)
- `probe_level6.py` (1230 bytes)
- `probe_level8.py` (2994 bytes)
- `probe_patterns8.py` (913 bytes)
- `probe_root8_actions.py` (1824 bytes)
- `probe_route.py` (881 bytes)
- `probe_search.py` (2205 bytes)
- `probe_search8.py` (4984 bytes)
- `probe_state8.py` (1031 bytes)
- `symbolic_probe_l2.py` (2882 bytes)

Run or inspect the smallest relevant probe before writing another one.
