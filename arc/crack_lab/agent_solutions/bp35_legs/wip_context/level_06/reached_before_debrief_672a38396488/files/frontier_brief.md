# Unverified frontier brief: bp35 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The search has not emitted another milestone yet, which means it is exploring alternate room-local support states rather than advancing the camera. The run remains inside the 12,000-state cap; if this pass stalls at the 22-band frontier, I’ll restart from its 16-action route with a much smaller tail-only search.
- I stopped the broad pass because deep clone reconstruction became too slow, but it yielded a decisive shorter macro: the same frontier in 16 actions. I’ll now re-root the already bounded tail probe on that route; this removes nine wasted actions and may change the lower hazard timing/state enough to make the previously lethal shaft viable.
- The shortened route confirms the final shaft is spatially lethal, not a move-budget issue. There is still one untested branch at the preceding full support barrier: the earlier attempt committed at column 2, but protected stepping can continue toward columns 1 and 0. I’ll probe those aligned lanes before concluding the upper-switch route is forced.
- This finds the missed branch: activating column 2 leads to the dead-end switch chamber, but activating columns 1 or 0 produces a different, nonterminal chamber on the left side of the barrier. I’ll follow those two verified alternatives through their visible open corridors; column 1 is especially promising because it transfers the avatar into the previously unreachable left region.
- The left chamber’s bottom supports are remotely configurable and persist as physical state, while the onward corridor is otherwise forced. I’ll enumerate the 16 configurations of its four still-small supports, then run the same switch-and-shaft suffix on clones. This directly tests whether staging the unreachable lower region is the required cooperative mechanic.
- The correct lane is now verified: column 1 preserves the avatar through the later switch, while column 2 does not. No extra support configuration is required. The avatar emerges at `(4,2)` in a new chamber with three supports beneath it and a left-side hazard bank; I’ll probe that chamber’s immediate movement and aligned interactions next.
- The post-lane-1 chamber has a forced safe direction: left or touching the aligned lower support kills, while three steps right scroll into another distinct chamber. I’ll continue from that new `(4,5)` frontier and test its two visible shafts plus the supports now above the avatar.
- The new chamber again has a single progress shaft: stepping left from column 5 to column 4 scrolls four bands, while the right edge is closed. I’ll now test how far the avatar can cross the exposed upper hazard bank and whether the newly appearing one-pixel objects are interactive.
- The pattern is now clear and verified chamber by chamber: the avatar advances by entering the only open support shaft, here moving from column 4 to column 2. The next chamber reverses that opening to column 4, so I’ll continue the alternating shaft traversal and watch for the reward object rather than spending moves on inert one-pixel cells.
- The apparent column-4 opening is lethal, so this chamber is another staging puzzle rather than a plain shaft. Its only remaining object bank is six one-pixel color-0 cells below the avatar (plus three ceiling hazards). I’ll preserve target identity and test each object before entering column 4, including an aligned column-2 activation and a column-0 setup.
- Direct interactions in the final chamber are inert, so the safety condition must be staged earlier. The four configurable supports in the left chamber persist through the scrolling world; I’ll now carry all 16 configurations through the entire verified suffix and test the final column-4 descent.
- The cooperative staging hypothesis is confirmed: expanding only support column 5 in the earlier left chamber makes the formerly lethal column-4 descent safe. That transition returns to the same room family with the safe support span extended from columns `2–4` to `2–5`. I’ll continue from this denser-progress state and identify the next edge lane to stage.

## Preserved local probes

- `probe_level6.py` (4255 bytes)
- `probe_level6_barrier.py` (4947 bytes)
- `probe_level6_bfs.py` (3832 bytes)
- `probe_level6_finish.py` (2376 bytes)
- `probe_level6_macro.py` (5346 bytes)
- `probe_level6_tail.py` (4529 bytes)
- `probe_level6_timeline.py` (2269 bytes)
- `probe_stage.py` (2391 bytes)

Run or inspect the smallest relevant probe before writing another one.
