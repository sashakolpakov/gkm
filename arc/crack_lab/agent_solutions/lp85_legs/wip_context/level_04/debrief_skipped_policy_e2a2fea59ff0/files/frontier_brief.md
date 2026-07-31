# Unverified frontier brief: lp85 level 4

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay within the current workspace and begin from the indexed API surface, then reproduce any inherited observations before relying on them. I’ll use compact clone probes and keep all level behavior in reusable legs.
- Baseline reproduction reaches level 4 (`levels=3`, 29 moves), so the frontier brief’s level-3 theories are stale and won’t be used. I’m now isolating the level-4 frame on a clone and enumerating its coordinate affordances through symbolic object/delta summaries.
- The level is a four-cross permutation puzzle. Sixteen border handles collapse into structured row/column operations; clicks permute persistent 2×2 tokens rather than creating or deleting objects. I’m extracting the exact symbolic permutation and testing candidate progress measures (cross uniformity and counterpart agreement) before any search.
- The four generators are two global shifts and their inverses; every visually separate handle in the same direction is equivalent. A bounded reward search is now exploring only genuinely distinct frames, with inverse cycles deduplicated; no moves have been committed to the real run.
- The token multiplicities reveal a strong goal invariant: exactly four colors have odd counts, matching the four centers, while every cross has four opposite pairs. That makes 180° symmetry simultaneously feasible and testable (current score 1/16). I’m learning the fixed permutations across varied clone contexts, then searching the cheap symbolic state space for 16/16 and verifying the reward on one clone.
- The learned permutations are two overlapping 20-position rings sharing the four cross centers—the same reusable mechanic as level 3, scaled up. Earlier levels’ solutions are four alternating ring rotations, so I’m testing that compact macro family exhaustively (rotation amounts 0–19) against the symmetry goal instead of expanding millions of single clicks.

## Preserved local probes

- `model.py` (1357 bytes)
- `model_l3.py` (4322 bytes)
- `model_l4.py` (6909 bytes)
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
- `probe_level2.py` (2244 bytes)
- `probe_offline.py` (3762 bytes)
- `probe_pair_l3.py` (2049 bytes)
- `probe_search.py` (2182 bytes)
- `reward_l3.py` (1384 bytes)
- `search_l3.py` (2330 bytes)
- `search_l4.py` (1832 bytes)

Run or inspect the smallest relevant probe before writing another one.
