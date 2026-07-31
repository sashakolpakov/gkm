# Unverified frontier brief: vc33 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and use the indexed ranges first. I’ll inspect `solver_index.md`, then the frontier brief if present, and reproduce any prior observations with bounded symbolic probes before editing the level-7 composition.
- The frontier suggests a coupled three-reservoir relay, but I’m treating that as unverified. I’ll now establish the real baseline with `gkm_try.py`, then inspect only the documented setup and smallest level-7 probe ranges needed to reproduce its symbolic claims.
- The compact preserved probe defines the right kind of dense measure: unresolved marker pairs first, then chamber crossing, then vertical alignment. I’m reproducing it unchanged now; this is bounded to 1,600 observational states per stage and only clicks visible cyan pads or active gates.
- The reproduction confirms the key mechanics and also exposes the scoring flaw: completing color 15 removes its marker pair, but the next greedy step completes color 14 in a way that leaves color 11 uncrossed and stalls. I’ll inspect the preserved first-traversal search and the current relay legs to reuse their verified action generation/state keys, then build one minimal general coupled-relay leg with a completion-aware search objective.
- The reproduced three-action traversal and alignment confirm that gate crossing is real. They also show that “pair disappeared” is not permanent—subsequent pressure transfers can make it reappear—so the true goal is simultaneous alignment, not pair-by-pair elimination. I’m gathering one compact initial-state/action table next, then I’ll search the small pressure-state graph directly for reward while using total visible mismatch only as an ordering heuristic.
- The action table shows eight persistent 2×2 cyan pressure controls; each changes a fluid boundary by two cells, while active gate blobs add context-dependent crossing actions. I’m adding a bounded best-first probe over only those visible affordances. It will prioritize total marker Manhattan mismatch but accept neutral/regressive states and stop only on the real level reward.
- The reward search is still running within its 20,000-state cap; output is buffered, so there is no intermediate trace yet. The search remains confined to clone actions on the eight visible pads and visible active gates.
- The first cap is too expensive for iteration at this clone depth. I’ll let this bounded run finish once because it may return the full path, but if it exhausts without reward I’ll tighten the state representation and reuse the known three-action gate prefix rather than increasing brute-force breadth.
- The broad best-first run confirms why action order matters: it crossed colors 11 and 14 first and still could not stage color 15 within 20,000 states. I’m now enforcing the experimentally verified rightmost-first subgoal—cross and align color 15, then transfer pressure three times—and searching only the remaining coupled state. This tests the frontier’s claimed relay order without trusting its conclusion.
- The ordered search is materially closer: it has reduced the remaining symbolic mismatch from 76 to 49 in only ~1,500 states, while allowing color 15 to reappear transiently and settle again. I’m keeping the cap unchanged and waiting for the real reward condition.
- The global search reached a one-marker near-solution but spent most of its budget revisiting alternate pressure distributions. The existing horizontal-gate leg already has the correct per-marker geometry; its only mismatch is its built-in ordering. I’ll test it as three single-color subgoals in the experimentally indicated order 15 → 14 → 11, which preserves reuse and should avoid a new search implementation if it completes.

## Preserved local probes

- `probe_level4.py` (13644 bytes)
- `probe_level5.py` (4716 bytes)
- `probe_level6.py` (3765 bytes)
- `probe_level7.py` (7566 bytes)
- `probe_level7_best.py` (3241 bytes)
- `probe_level7_compose.py` (1533 bytes)
- `probe_level7_score.py` (3289 bytes)
- `probe_level7_search.py` (5641 bytes)
- `reproduce_level4.py` (5779 bytes)

Run or inspect the smallest relevant probe before writing another one.
