# Unverified frontier brief: dc22 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- Remote exit is fixed: action `1` returns the avatar to tile `(19,4)` regardless of selector state. I’ll now test the large assembly control from that return point. The key question is which phase reconnects the avatar to a new island; I’ll track reachable tile sets rather than raw pixel changes.
- The five-phase reachability sweep is again clone-heavy but remains bounded (at most 500 positional states). I’m waiting for its compact phase summaries; after this I’ll use only targeted paths, since the phase that exposes a new island should eliminate the need for broad search.
- The phase sweep exposes a key correction: deduplicating solely by avatar position hid world evolution. In the verified upward transfer, repeated blocked-looking moves advance the large assembly until the avatar can move—so the assembly is time-dependent on every action. I’ll probe repeated downward actions from the return point under each initial control phase, recording only actual avatar transitions and reward.
- The return tile `(19,4)` is itself the teleporter mouth: moving down re-enters the remote room, so it is not a descent route. The unexplored frontier is above the upper-left island (top island at rows 2–5). I’ll stage at row 8 and repeat upward actions under each large-assembly phase, preserving temporal changes even when the avatar initially appears blocked.
- The map now suggests a teleport hub: paired-color endpoints correspond to isolated hub arms, while the selector changes an endpoint on the starting island. A concrete navigation oversight explains the failed descent probe: `(19,4)` is the teleporter column, but adjacent `(19,5)` lies on the builder route. I’ll shift right before testing synchronized downward movement.
- Both columns feed the same upper teleport mouth, confirming the route is intentionally one-way. The selector may be latched when the hub is re-entered rather than when it is exited, so I’ll test a full cycle—set each pair, exit upward, re-enter downward—and record the resulting hub arm and its available exits.
- The selector does not redirect teleports directly. A stronger interpretation is that the changing 2×2 patterned object is an independently controlled agent: clicks rotate/set it, then ordinary turns may let it move. I’ll hold the avatar blocked in the hub and advance turns, tracking that object’s position and orientation for each selector phase.
- The object is stationary; ordinary turns do not advance it. At this point the remaining interaction space is small but context-dependent, so I’ll run a bounded full-observation search from the hub over movement plus the three verified controls. Unlike the earlier positional search, its key includes the entire visible world state, preserving control phases and timed bridge configurations.
- The full-state search is behaving correctly: it distinguishes bridge phases and has already found a context-dependent new route to tile `(17,2)` using the large-assembly control—something the avatar-only key missed. I’ll let the 800-state bound finish; the search is still within the requested bounded regime and is now exploring genuinely new configurations.
- At 400 states the assembly control is confirmed to move/connect laterally: paths now reach columns 1 and 7 from the upper region. No reward yet, so the remaining bound is testing whether that lateral transfer reaches the top or another teleport endpoint. I’ll stop at the preset 800-state cap if it does not win.
- The search has reached row 12 and continues expanding upward; selector and platform phases are not yet yielding shortcuts, while repeated movement plus the large assembly is. I’m keeping the run to its original cap so its negative result, if any, remains interpretable.
- The bounded search did not win, but it exposed a strong transfer pattern: at row 17, one large-assembly activation lets the avatar traverse laterally from column 3 out to column 0 or from column 6 out to column 8. I’ll stage at the right frontier `(17,8)` and test synchronized rightward continuation under successive assembly phases—the likely route back to the central endpoint island.

## Preserved local probes

- `probe_level6.py` (6750 bytes)
- `probe_level6_stage10.py` (1609 bytes)
- `probe_level6_stage11.py` (1649 bytes)
- `probe_level6_stage12.py` (1674 bytes)
- `probe_level6_stage13.py` (2328 bytes)
- `probe_level6_stage14.py` (1651 bytes)
- `probe_level6_stage3.py` (2458 bytes)
- `probe_level6_stage4.py` (3002 bytes)
- `probe_level6_stage5.py` (2798 bytes)
- `probe_level6_stage6.py` (1501 bytes)
- `probe_level6_stage7.py` (2099 bytes)
- `probe_level6_stage8.py` (1572 bytes)
- `probe_level6_stage9.py` (1599 bytes)
- `wip_glue_notes.md` (294 bytes)

Run or inspect the smallest relevant probe before writing another one.
