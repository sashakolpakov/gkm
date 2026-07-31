# Unverified frontier brief: lf52 level 4

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and begin with the indexed surfaces, then reproduce any prior observations on bounded clones before changing the leg library or player.
- The baseline reliably reaches level 4. I’m now instrumenting only that live level through clones: compact components, available actions, and one-step deltas, including coordinate probes at visible object centers.
- The level-4 probe is still reconstructing the first three solved levels; no output has appeared yet. I’m letting that bounded run finish rather than launching overlapping clone searches.
- The direct reconstruction exposed an important harness constraint: deep clones after replay become prohibitively expensive. I’ll use the harness’s documented resume path instead, with a temporary level-4 observation leg, so every fact still comes from the live frame/action surface.
- The first controlled interaction identifies the active piece: only the color-14 piece at `(24,12)` responds to a click. The two color-15 pieces lie exactly one lattice step to its right, so I’m testing the natural capture chain as a dense-progress hypothesis, while separately tracking how the key-moved carrier changes reachable lattice positions.
- The capture-chain hypothesis is now verified more precisely: the color-15 cells are persistent bridge markers, not removable pegs. They let the active peg advance `12→24→36`; that stages two adjacent color-14 pegs so the movable empty carrier can receive a true peg capture. I’m encoding those two observed macro types in a general bounded search leg.
- The general search is exploring too many animation-distinct states, although its action macros are correct. I’m tightening the state key to the puzzle-relevant objects—peg, hole, bridge, and carrier positions—so incidental moving scenery cannot consume the clone budget.
- The staged run revealed the crucial interaction: once the carrier is engaged, later clicks advance the surrounding boards, so stale absolute coordinates miss their targets. I’m switching the probe to a per-action symbolic trace; the final leg will recompute macros from each current frame rather than replay fixed coordinates.

## Preserved local probes

- `probe_level2.py` (6693 bytes)

Run or inspect the smallest relevant probe before writing another one.
