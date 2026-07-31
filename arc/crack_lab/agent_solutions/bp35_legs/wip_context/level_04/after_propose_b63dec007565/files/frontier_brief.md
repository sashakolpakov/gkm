# Unverified frontier brief: bp35 level 4

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The frontier suggests level 4 needs a broader interaction model than the earlier tower legs, but those notes are unverified. I’m reproducing the baseline first with the harness, then I’ll inspect only the relevant API bodies and the smallest preserved level-4 probe.
- Reproduction confirms the avatar is the two-tone color-9/11 figure: action 4 moves it right/up the staircase and action 3 moves it left/down, while coordinate action 6 is contextual. The key transition is clicking the lone upper `*` after four right moves; that changes the whole room and creates a second configuration, so camera-height alone is not a sufficient progress metric. I’m now searching that interaction-expanded state space for the actual reward transition.
- The bounded search is still running without a reward hit yet. Its current action generator includes both movement keys and clicks on every visible non-terrain lattice object; I’m letting this one bounded pass finish, then I’ll tighten the state key or derive a staged subgoal if it closes without success.
- This pass is clone-cost-bound and has exceeded the intended exploration rate, so I won’t repeat it. I’m giving it one final short interval; if it remains non-terminal I’ll stop it and switch to compact, staged probes around the post-click configuration.
- The exhaustive pass was stopped because deep cloning became far slower than the requested budget. I’m replacing it with a staged probe at the verified room transition: measure the avatar’s reachable positions, automatic motion of the two hazards, and every coordinate interaction that changes more than the move counter. This should expose the next subgoal without retaining a large clone tree.
- The staged probe found real dense progress: moving left to a removable support and clicking it produces another full-room transition while preserving the avatar. The new room exposes color-8 objects and a different set of supports, so level 4 is a sequence of gravity/support configurations, not a single tower. I’m probing the second configuration’s interaction set now; this is the right abstraction for a small reusable “room-transition” leg.
- In the second configuration, the color-8 objects are verified traps (clicking either terminates), while removable color-14 supports remain safe interactions. I’m switching the search representation, not broadening the action set: each state will be reconstructed from one shallow root and keyed by the visible frame plus clock parity. That avoids the recursive clone history that stalled the earlier pass.
- The second room’s geometry now explains the contextual interaction: color 8 behaves like a gravity toggle. Activating it at column 0 kills because there is no ceiling above the avatar; the same toggle should be safe only after staging the avatar under a removable ceiling at column 1 or 3. I’m verifying both alignments on clones before using that mechanic.
- The context test confirms the gravity model. Column 1 toggles safely, but clearing its ceiling then launches the avatar into the two hanging hazards and terminates. Column 3 has a solid catch ceiling above the removable support, so I’m verifying that route instead.
- The straightforward ascent after toggling is a verified dead end: the remaining removable ceiling is directly below the hanging hazards. I’m now using a much smaller context-aware frontier—movement, every visible gravity toggle, supports aligned with the avatar, and only nearby hazards. This preserves all verified affordances while cutting the irrelevant distant-click branching that made the raw BFS unusable.
- The compact search closed 900 states cleanly but was still branching on aligned supports several rooms away. Every verified support transition occurred immediately above or below the avatar, so I’m tightening only that part to adjacent supports while retaining all visible gravity toggles. This should let the same bounded budget reach meaningful sequence depth.
- The context-bounded search found a 19-action reward path in 601 expanded states. Its key insight is that the shortest route activates a newly exposed toggle after only two right moves, then alternates safe toggles with adjacent support removals. I’m replay-verifying that exact path independently before promoting the mechanic into a reusable leg.

## Preserved local probes

- `compact_search4.py` (2890 bytes)
- `frontier_scaffold.json` (1700 bytes)
- `l1_probe.py` (7220 bytes)
- `l1_search.py` (2988 bytes)
- `level4_prefix.json` (702 bytes)
- `p3.py` (1634 bytes)
- `probe.py` (7635 bytes)
- `probe4.py` (5322 bytes)
- `probes3.py` (9594 bytes)
- `replay_search4.py` (2237 bytes)
- `search4.py` (2299 bytes)
- `stage4_probe.py` (6156 bytes)
- `verify4.py` (901 bytes)

Run or inspect the smallest relevant probe before writing another one.
