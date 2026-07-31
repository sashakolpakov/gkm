# Unverified frontier brief: lp85 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and begin with the indexed API surface. I’ll reproduce any inherited observations before relying on them, then probe level 6 symbolically and keep the final player as leg composition only.
- The baseline is already at the requested frontier: `levels=5`, so level 6 is the active unsolved screen. The prior brief concerns level 5 and provides no level-6 facts; I’ll now generate a fresh level-6 clone via the existing solver and characterize coordinate clicks with compact deltas.
- Fresh probing identifies seven responsive arrow-like controls and no key movement: this is a coordinate permutation puzzle, with colored 2×2 tokens arranged in three visible structures. I’m reducing the frame to those token colors now, measuring each control’s exact cycle and testing reward only on clones.
- The bounded clone search is behaving cleanly and has reached depth 4 with 700 distinct symbolic states; no reward yet. I’m capping it at 20,000 states/depth 18, which keeps the probe within the requested bounded regime and will either produce a verified click path or show that a denser structural target is needed.
- The first BFS exposed a probe-side bottleneck rather than a game limit: it was recomputing connected components for every clone. I stopped it cleanly and replaced the key with direct sampling of the already-observed fixed token coordinates; the search semantics are unchanged, but state checks are now constant-size.

## Preserved local probes

- `model_l3.py` (4322 bytes)
- `model_l4.py` (6909 bytes)
- `model_l5.py` (2970 bytes)
- `probe_l3.py` (2908 bytes)
- `probe_l4.py` (2646 bytes)
- `probe_l5.py` (3217 bytes)
- `probe_l6.py` (2835 bytes)
- `reward_l3.py` (1384 bytes)
- `rewarded_l4.py` (2078 bytes)
- `search_l3.py` (2330 bytes)
- `search_l4.py` (1832 bytes)
- `search_l6.py` (1562 bytes)

Run or inspect the smallest relevant probe before writing another one.
