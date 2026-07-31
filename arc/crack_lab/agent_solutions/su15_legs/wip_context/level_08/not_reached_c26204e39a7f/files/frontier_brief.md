# Unverified frontier brief: su15 level 8

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace and use only the indexed harness/perception surface. I’ll first locate the prescribed index and frontier brief, then run the baseline and probe level 8 symbolically before making any edits.
- The baseline is deterministic at level 7 in 133 moves, so level 8 is indeed the active frontier. I’ll now inspect only the harness entry points plus the existing leg interfaces, then create a compact clone probe that summarizes objects and the two coordinate actions without dumping frames.
- The first level-8 probe is still replaying the bounded planners for earlier levels; it hasn’t emitted any observations yet. I’m letting that clean replay finish, then I’ll reuse the resulting compact state rather than repeatedly paying this startup cost.
- The first action experiment identifies the directly controlled object: action 6 places the 7×7 color-8 square at the clicked coordinate when reachable. Every valid click also advances three independent small bodies, while two 4×4 color-11 squares persist and four ring targets occupy the corners. This strongly resembles earlier merge/staging mechanics in a combined configuration, so I’m testing the existing reusable legs on clones before inventing anything new.
- The dense search has converged on the intended assignment: the 7×7 square occupies the upper-left ring, while the three moving bodies are each within a few cells of the other three rings. Reward has not fired yet because their centers are not all aligned, so I’m narrowing the final steering rather than broadening the search.

## Preserved local probes

- `probe.py` (2087 bytes)
- `probe_dots.py` (903 bytes)
- `probe_level4.py` (5557 bytes)
- `probe_level5.py` (1909 bytes)
- `probe_level5_sequences.py` (2002 bytes)
- `probe_level6.py` (1230 bytes)
- `probe_level8.py` (2994 bytes)
- `probe_route.py` (881 bytes)
- `probe_search.py` (2205 bytes)
- `probe_search8.py` (4984 bytes)
- `symbolic_probe_l2.py` (2882 bytes)

Run or inspect the smallest relevant probe before writing another one.
