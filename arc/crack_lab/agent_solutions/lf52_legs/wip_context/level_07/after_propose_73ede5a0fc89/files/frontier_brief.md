# Unverified frontier brief: lf52 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the clean-room surface. I’ll begin with `solver_index.md`, then any `frontier_brief.md`, and use only targeted ranges plus bounded symbolic probes.
- The baseline run is still initializing its local rendering cache; no game result has appeared yet. I’m waiting on that same process, then I’ll inspect only the relevant harness/API ranges.
- Baseline confirms the existing library reliably reaches six levels. I’m now reconstructing level 7 through the same public run callback and summarizing its blobs plus one-step action deltas; I won’t rely on checkpoint or prior-attempt artifacts.
- Replaying the six solved levels from scratch is slower than the checkpointed baseline, but the bounded probe is still running normally. Once it reaches level 7, the output will be a compact object/action summary rather than a frame dump.
- The level is a composite relay/peg layout, and the public action list is `(1,2,3,4,6)`: four keys plus coordinate action 6. I’m testing each key and each visible piece/slot click independently on clones, which should distinguish carrier movement from peg selection without assuming either mechanic.
- The experiments identify color 12 as the controllable carrier: keys move it on a 6-pixel lattice, while action 6 performs source/destination transfers. The first verified subgoal is to align the carrier at `(24,6)` beneath the peg and fixed bridge, then test transferring the peg onto it and transporting the combined object.
- The first relay is verified: the peg rides the carrier to `(42,12)`, crosses a fixed bridge, and lands at the bottom track. The geometry suggests a second cooperative relay: use the carrier to bring the distinct movable bridge from the upper-right chamber down to the bottom, then alternate peg-over-bridge and bridge-over-peg jumps toward the far-right goal. I’m verifying that bridge transport now.
- The bottom relay triggers a horizontal wrap into a second chamber, so the goal is not the first far-right slot. The frame now exposes a new carrier network and fixed bridges. I’m recomputing carrier reachability in this wrapped chamber before making further transfers.
- The second chamber’s cooperation is now verified: the carrier loads the movable bridge at `(54,22)`, and the peg jumps over that carried bridge to `(54,28)`. Next I’m staging the bridge onto the upper lattice, freeing the carrier, then searching only the carrier’s small key-action graph to bring the peg to the matching vertical relay.
