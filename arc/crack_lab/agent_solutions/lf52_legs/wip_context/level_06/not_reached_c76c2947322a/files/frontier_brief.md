# Unverified frontier brief: lf52 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace and begin with the indexed API map, then reproduce any prior observations before probing level 6. I’ll keep frame analysis symbolic and clone searches bounded.
- The baseline is clean: five levels complete in 238 moves. I’m now locating the harness entry points and the smallest existing-leg interfaces needed to construct a level-6 clone without rereading unrelated code.
- Reaching the level through the real five-level prefix is taking longer than the baseline because the probe reruns the search-heavy level-5 solver. I’m letting that legitimate prefix finish; the probe itself performs only one cloned step per available action.
- The level-6 frame does not match the existing bridge/carrier parser—the reused leg makes no state change. I’m switching to a temporary observational leg so the normal validated checkpoint can expose level 6 through the documented harness, without consuming or inspecting any prior transcript artifact.
- The coordinate probes establish the new configuration’s key mechanic: the color-8 object is a persistent bridge. A peg at `(24,18)` can jump across it to `(12,18)` while the bridge remains, which is the same reusable bridge/carrier leg with a new bridge palette. I’m generalizing that parser minimally and testing the existing dense-progress solver.
- The key-controlled object is a two-slot carrier at the right edge; pushing right simply ejects its slots and does not complete the level, so that is a staging mechanism, not the goal by itself. I’m enumerating only observationally valid peg source/destination pairs over visible slots—about 200 cloned steps—to map every legal macro without assuming the old move rules.
- The bridge itself is coordinate-movable: it can leap over a peg from `(18,18)` to `(30,18)`, while pegs can leap over the bridge. This is a genuine new cooperative mechanic, so I’m adding one minimal reusable leg: a bounded symbolic search over verified leapfrog macros plus carrier key moves, with peg count as the dense progress signal and the real level reward as the terminal test.
- The symbolic search is still running past 30 seconds, so the raw breadth-first ordering is exploring too many neutral leapfrog arrangements. If it does not finish on the next short poll, I’ll retain the same verified macros but prioritize states with fewer pegs and closer cooperative alignments.
- The search found and replay-verified a 12-click sequence that reduces three pegs to one, but the real reward correctly stays zero. The surviving peg is adjacent to the carrier corridor, so the denser subgoal is now clear: leapfrog it onto the carrier, then test carrier ejection on a clone before encoding the final leg behavior.
- Landing the last peg on the first carrier slot causes a camera pan, explaining the earlier apparent carrier motion: the movable color-8 bridge/avatar stays central while the world scrolls. The remaining symbolic arrangement is bridge–peg–empty-carrier in consecutive lattice cells. I’m testing the natural cooperative terminal macro—bridge leaps over the peg into the second carrier slot—on that clone.
- Loading both cooperative pieces is a staging state, not the terminal state. From there, rightward travel scrolls through the world and eventually blocks, confirming a final navigation phase. I’m folding this into the reusable leg: symbolic leapfrog search until the last peg reaches the carrier, observed loading of the bridge into its companion slot, then a bounded key-only BFS keyed on the visible frame with the action-counter row excluded.
- The navigation search is again clone-bound rather than logic-bound. I’m stopping this run and switching that final phase to path-replay BFS from the single staged clone, so the queue stores compact paths and observational keys instead of thousands of deep game copies.

## Preserved local probes

- `probe_level6.py` (978 bytes)

Run or inspect the smallest relevant probe before writing another one.
