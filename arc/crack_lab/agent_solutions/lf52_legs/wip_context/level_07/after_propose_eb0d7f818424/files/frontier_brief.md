# Unverified frontier brief: lf52 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The second chamber’s rail graph is now fully reproduced at 21 carrier positions. A naïve bridge-first transfer reaches the upper exit but cannot unload there, so I’m checking the other rail endpoints and context-dependent coordinate action rather than assuming the endpoint semantics. This is the key ambiguity before committing a reusable sequence.
- The endpoint sweep is still within the 40-state cap but clone reconstruction is slower than the symbolic graph size suggests. I’m letting this precise sweep finish; if it yields no unload transition, I’ll switch to the alternative ordering (peg relocation before bridge transport) rather than expanding the search.
- The correct cooperation pattern is now verified: the bridge is transported first, the freed carrier returns for the peg, and the peg uses both a fixed support and the movable bridge to enter the upper board. Advancing the bridge over the next fixed support causes a viewport wrap and reveals a new peg plus a second carrier—clear dense progress. I’m mapping this newly revealed relay state next.
- The dual-carrier graph probe is still running under its 80-state cap; the expensive part is clone depth, not graph breadth. I’ll use its reachable carrier pairs to identify whether both agents can service the two separated peg regions, then reduce the rest to explicit relay macros.
- The first BFS target was too permissive because ordinary piece movement changes the visible slot set. I tightened the dense goal to a real reduction in peg count (or level completion); the replay-based search is now exploring that exact condition within 300 symbolic states.
- The newly revealed section is a two-agent ladder: fixed supports at `(36,12)`, `(36,24)`, `(42,12)`, and `(42,24)` let the first peg advance to `(42,30)`, while the long-rail carrier can reach `(30,30)`. I’m verifying that transfer now; it should convert a separated peg into a carrier-held piece and expose the next cooperative handoff.
- The apparent long bridge jump is invalid, which confirms the local lattice spacing rather than a variable-distance rule. I’m switching back to bounded replay search, but cloning directly from the live third-chamber root (the earlier slowdown came from cloning an already-cloned root). The search target remains a verified peg-count reduction.
- Direct-root reconstruction is healthier but still expensive at the 500-state ceiling. I’ll give this exact search one more interval; if it does not finish, I’ll cap it and use the already verified ladder prefix as a forced macro so the remaining search starts after the first peg is carrier-held.
- The forced ladder prefix is verified and the reduced search now starts with peg A already held at `(30,30)`, leaving only the cooperative handoff. It is still computing, but the state cap is now 250 and the search no longer spends states rediscovering the ladder.
- A useful mechanic just resolved the dead end: a carrier-held peg at `(24,54)` can unload directly to the slot at `(36,54)` without an intervening support. That means carrier transfers are their own affordance. I’m now testing the corresponding bridge transfer through the long rail, which is the likely way to service the otherwise isolated right-hand peg.
- The empty carrier does not act as a jump support, so the bridge cannot cross the wide gap that way. I’m doing one focused affordance sweep with peg A held at the rail’s terminal `(36,42)`; this checks every visible destination for the held peg, bridge, and isolated peg and should expose any terminal-specific transfer without another broad search.
- The visible-board solver also confirms there is no purely local peg solution, so the remaining mechanism must involve rail-induced viewport movement or a carrier tunnel. I’m checking the rail boundary paths directly now; prior carrier-only summaries hid whether the board itself shifted when one carrier left the viewport.

## Preserved local probes

- `probe_checkpoint_level7.py` (7358 bytes)
- `probe_l7_next.py` (2694 bytes)
- `probe_level7.py` (988 bytes)

Run or inspect the smallest relevant probe before writing another one.
