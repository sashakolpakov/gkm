# Unverified frontier brief: lp85 level 8

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside this workspace and use only the documented harness/observation surface. I’ll begin with the indexed file map, then reproduce any needed frontier observations with compact clone probes before editing the leg library.
- Baseline is confirmed at six completed levels, so level 7 is the active screen. I’m inspecting only the small solver definitions and the prior probe’s documented environment setup, then I’ll characterize level 7 from a fresh clone with symbolic object/delta summaries.
- The screen is a coordinate-only puzzle with two responsive regions. One click changes a compact object near the left structure; the other changes a larger pattern near the right/lower structure. I’m now measuring their exact state transitions, repeat periods, and reward behavior—the dense progress signal will be the symbolic configuration of the affected colored cells.
- The exact first control does alter the left branch; the second advances the shared colored configuration with an eight-step period. I’ve launched a two-control bounded search, but clone expansion is slower than expected, so I’m keeping the cap tight and will switch to replay-from-base keys if it does not finish promptly.
- The puzzle’s mechanics are now clearer: one control rotates the eight-token horizontal ring; the other affects a much smaller exchange near its left end. The earlier BFS key included intermediate animation geometry, inflating the state count. I’m reducing the world to the persistent 2×2 token colors so the search operates on the actual permutation state and can use “tokens in target-relative positions” as dense progress.
- The dense objective is now supported by the earlier verified level family: each motif must become 180°/endpoint symmetric. Level 7 has seven equality pairs across an 8-token row, a linked 3-token column, and a 2×2 square. The odd color counts fit exactly because the row and column share an endpoint. I’m switching from clone enumeration to the verified two permutation operators and will reward-check the resulting symmetry path on a clone.
- The first symmetry hypothesis is structurally consistent but a one-sided breadth-first search is too diffuse—the two overlapping rotations generate a very large permutation orbit. I’m preserving the seven-pair dense target and changing the search to multi-source reverse search from every color-count-compatible symmetric arrangement. That uses the known inverse rotations and should expose whether the target is truly reachable without spending clone budget.

## Preserved local probes

- `model_l3.py` (4322 bytes)
- `model_l4.py` (6909 bytes)
- `model_l5.py` (2970 bytes)
- `model_l7.py` (1569 bytes)
- `probe_l3.py` (2908 bytes)
- `probe_l4.py` (2646 bytes)
- `probe_l5.py` (3217 bytes)
- `probe_l6.py` (2835 bytes)
- `probe_l7.py` (2381 bytes)
- `reward_l3.py` (1384 bytes)
- `rewarded_l4.py` (2078 bytes)
- `search_l3.py` (2330 bytes)
- `search_l4.py` (1832 bytes)
- `search_l6.py` (1562 bytes)
- `search_l7.py` (3442 bytes)

Run or inspect the smallest relevant probe before writing another one.
