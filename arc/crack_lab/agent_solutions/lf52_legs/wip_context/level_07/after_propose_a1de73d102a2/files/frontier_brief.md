# Unverified frontier brief: lf52 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The inherited brief is only a hypothesis ledger, so I’m treating it as untrusted. The required baseline run is still starting; after it completes I’ll use compact public-frame summaries to independently verify level 7’s carrier, transfer, and relay mechanics.
- Baseline is confirmed at six completed levels. I’m now building one bounded probe that replays the verified six players, then reports level 7 as object components and clone deltas—enough to identify the controllable object and legal click transfers without dumping pixels.
- The clean replay is still running; that cost is in the six established solvers rather than the level-7 clone set. I’m keeping the first observation pass deliberately small: four key clones and a component inventory.
- The from-scratch replay is slower than the checkpointed baseline but remains the clean way to reproduce level 7 without consuming prior artifacts. I’m letting this single process finish so subsequent experiments can be consolidated into fewer launches.
- No partial output is expected because the probe only prints after level 6, but the process has not failed. If this clean replay proves too costly, I’ll fold all remaining level-7 experiments into its eventual callback rather than repeatedly paying the setup cost.
- A direct reuse check is decisive: the level-6 wrapped-relay leg makes no level-7 move, so this is a new configuration rather than a drop-in repeat. I’ll now route a temporary diagnostic leg through the documented `gkm_try.py` callback; this preserves the player’s composition rule and gives fast, reproducible clone evidence at the checkpointed level boundary.
- The first reproduced frame confirms the main mechanics: color 12 is the only key-moved 4×4 object, initially on a horizontal rail; color 14 is the peg; color 8 is a distinct movable bridge; action 6 alone only marks a selection. I’m next mapping the carrier’s bounded key-action reachability and visible wrap states, using frame keys that ignore the action-counter pixel.
- The carrier graph is now verified as a 16-position rail network. It reaches `(24,6)` beneath the peg and `(24,42)` beneath the movable bridge, with lower exits at `(42,12)` and `(42,42)`. I’m testing the full first relay as a single clone sequence and will inspect the post-wrap chamber only if every transfer produces the expected object state.
- The complete first relay is verified, including a crucial coordinate wrap: moving the bridge from bottom column 42 to 54 scrolls the board, relocating the peg/bridge pair to columns 4/10. The final peg jump must therefore use the new coordinates `(54,4)→(54,16)`. I’m now mapping the second chamber’s carrier route to the lower relay.
- The second-chamber reachability probe is the first heavier clone search, but it is capped at 300 observational states and depth 24. Its target is concrete: a carrier position at `(54,22)` that enables the bridge-over-peg load inferred from geometry.
- The in-place traversal exposed a limitation: viewport transitions are not perfectly reversible, so inverse-key backtracking cannot certify routes across them. I’m switching to the scaffold’s path-reconstruction style for one narrow search—carrier at `(54,22)`—with paths rather than recursively nested clones.
- The targeted replay search is still within its 160-state cap; reconstruction is slower because each candidate starts beyond the six-level replay plus the first relay. I’m waiting for this single result rather than broadening the search.

## Preserved local probes

- `probe_level7.py` (988 bytes)

Run or inspect the smallest relevant probe before writing another one.
