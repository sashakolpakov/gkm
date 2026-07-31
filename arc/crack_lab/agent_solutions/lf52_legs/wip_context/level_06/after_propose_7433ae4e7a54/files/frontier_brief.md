# Unverified frontier brief: lf52 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The survivor carrier is coupled to the bridge’s obstruction, not merely the empty carrier. I’m testing the cleaner separation: descend, move the bridge fully off the upper rail while leaving both carriers at their original lower columns, then lift. If successful, the survivor should reach row 30 with the bridge absent, allowing horizontal carrier travel.
- The forced-exit choice is the likely mistake: it parks the survivor in the left carrier and strands the bridge. I’m moving the search branch point earlier—to the frame immediately after the verified upper capture, when the survivor and bridge are still adjacent and both empty carriers are available. A live-clone macro search from there can discover the correct relay/exit without re-solving the upper board.
- The immediate-capture frontier is still running within its 600-state/300-second cap and has not reached a second capture yet. This is the highest-value remaining search because it includes both possible survivor exits and bridge reload choices while excluding the already-solved prefix.
- No new capture has appeared by roughly 180 seconds. I’m keeping the cap fixed; if this exhausts, the next move is not more brute force but a deterministic relay probe from the immediate-capture frame, where only two non-reversing macro choices exist.
- The search exposed a crucial correction: immediately after capture, the bridge is already loaded in the right carrier at `(30,46)`, while the survivor is at `(18,40)` and the other carrier is `(30,28)`. I’m now probing the symmetric right-edge relay—unload bridge, move peg and bridge to the right edge, shift the freed carriers, and reverse into a newly positioned carrier.
- The right relay reveals the level’s wraparound topology: moving the survivor past column 52 shifts the viewport, exposing four pegs and two carriers in a new sector. The bridge at `(18,2)` can either continue horizontally or drop through a static bridge into the carrier at `(30,2)`. I’m testing the carrier drop, which is the first verified way to preserve the survivor in the new sector while routing the bridge onward.
- Dropping the bridge into the left carrier leads to a blocked lower edge. The lattice classes reveal the symmetric correct dock: advance the bridge/peg pair across the wrapped upper row until the bridge reaches column 38, directly above the other carrier, then drop it vertically while leaving the survivor at column 32. I’m verifying that route and its lower descent.
- The wrapped sector contains four real pegs in separated maze regions, so the earlier “one upper plus one lower board” model was incomplete. I’m taking one compact symbolic map at this verified post-capture sector to identify the actual carrier rails and board entrances; that should replace blind key branching with finite route choices.
- The map shows two bridge-carrier shafts and a lower horizontal rail; the missed choice is at the rail’s finite right stops. I’m testing Down at bridge columns 38, 44, and 50. This is the same “move horizontally, then descend at a shaft” affordance verified earlier, now in the wrapped sector.
- The direct rail shafts are blocked, so I’m returning to bounded search with a better dense measure derived from the new map: prioritize states that expose more pegs and reduce the nearest orthogonally aligned peg-pair distance, in addition to capture-ready macros. The previous search treated all non-capture carrier states nearly equally and spent its budget on relay cycles.
- The revised heuristic immediately found the wrapped right relay and reduced the nearest aligned peg gap from 64 to 18 within six macros—the exact hand-derived route. It is now exploring from that meaningful frontier rather than cycling among visually similar carrier states.
- No additional gap reduction or capture has appeared yet, but the frontier is still inside its fixed cap. The current best gap of 18 means two pegs are collinear with one lattice slot between them; the remaining task is exposing the correct landing slot/carrier phase.

## Preserved local probes

- `probe_level6.py` (1119 bytes)
- `probe_level6_after_upper_align.py` (723 bytes)
- `probe_level6_after_upper_search.py` (4677 bytes)
- `probe_level6_after_upper_trace.py` (2157 bytes)
- `probe_level6_existing.py` (613 bytes)
- `probe_level6_focus.py` (14424 bytes)
- `probe_level6_lower.py` (4595 bytes)
- `probe_level6_lower_route.py` (4195 bytes)
- `probe_level6_macro_search.py` (3587 bytes)
- `probe_level6_macros.py` (13153 bytes)
- `probe_level6_relay.py` (2268 bytes)
- `probe_level6_right_relay.py` (3916 bytes)
- `probe_level6_search.py` (5292 bytes)
- `probe_level6_selection.py` (1162 bytes)

Run or inspect the smallest relevant probe before writing another one.
