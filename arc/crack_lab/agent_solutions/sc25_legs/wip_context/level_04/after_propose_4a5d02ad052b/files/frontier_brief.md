# Unverified frontier brief: sc25 level 4

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The border is a hard 32-action timer: fresh clones become terminal exactly when it is exhausted. That makes the intended solution highly structured. Opening the gates is reproducible, but merely entering the left dock is not the win; I’m now checking whether the cross is a reversible size toggle when submitted inside the dock, which would explain both the 4×4 dock geometry and the nearby 2×2 inset.
- The cross is indeed reversible: submitted after entering the 4×4 left dock, it grows the avatar back to exactly the dock footprint. That still does not reward immediately, so the dock is a staging condition rather than the whole goal. I’m checking whether docking activates the remote actor—first via ordinary directions, then via bounded one-to-three-cell panel commands—using remote-shape change as the dense measure.
- Docking does not activate any panel command of up to six cells, and the remote shape never changes, so it is likely a fixed destination/socket rather than a second controllable avatar. I’m using the already-solved tutorial levels as controlled evidence now: the frame immediately before and after each known reward will reveal which contact or shape relation actually defines “finish,” without inspecting any hidden implementation.
- The solved levels establish the reward condition: the moving avatar must contact the fixed colour-9/10 socket. Level 4’s socket is the right-hand object, while the left-side command station is only a gate mechanism. I’m now running a bounded shortest-path search over verified macro-actions (movement, cross resize, and the two three-cell line commands), keyed by the visible world with the timer excluded; this will test whether a shorter transform/gate route exists before introducing any new mechanic.
- The bounded search is still exploring the visible state graph; it has not produced a verified reward path yet. I’m letting this capped run finish, then I’ll either replay its shortest path or tighten the state abstraction around avatar pose, gate state, panel state, and remaining timer.
- The expensive generic search was dominated by deep clone histories, so I stopped it within its cap. The tutorial evidence suggests a shorter physical plan: approach the bottleneck while still large (4-cell stride), shrink only for the narrow left corridor, open the linked gates, immediately grow again at the corridor edge, then use the restored 4-cell stride to reach the right socket before the sweep. I’m replaying that 35-action candidate now.
- The optimized route reaches the marker just before the sweep, but the next vertical-alignment move still loses at action 33. I’m inspecting that final contact frame to distinguish “marker was collected but gives no time” from “the large avatar was blocked”; one action is the current gap, so anchor/orientation ordering matters.
- The cross only grows inside the opened left station; it is not a free toggle in ordinary corridor space. That clarifies the intended loop. With the earlier large-stride approach, docking and regrowing occurs by action 26, leaving a direct nine-stride run that intersects the marker en route to the socket. I’m testing that full station-mediated route; the station itself appears to delay the sweep compared with the undocked candidate.
- Growing inside the left station traps the large avatar, confirming it is a checkpoint rather than a return route. The remaining physical interpretation is that entering the station resets/pauses the descending hazard; the small avatar can then exit, traverse back through the opened corridor, use the right-side marker as the growth station, and enter the nearby socket. I’m testing that staged two-region route now.
- The station snaps a partially entered avatar fully into its 4×4 footprint, so the trap is intentional. The remaining unexplored affordance is a longer panel command after docking: prior enumeration used an inefficient arrival that timed out after six clicks. I’ve reduced docking to action 26 and am enumerating the remaining seven-to-nine-cell patterns for any reward or remote-core change.
- No panel subset—whether docked small or large—changes the remote object, so panel commands are not the handoff. One documented coordinate affordance remains under-tested: selecting the green/blue mechanisms themselves before issuing their pattern. I’m probing those object clicks from the initial frame; a remote gate selection would eliminate the long left traversal and make the socket reachable within the sweep.
- Direct object selection is inert, so there is no remote gate shortcut. I’m switching to the scaffold’s path-only BFS (designed for expensive deep clones) with an abstract key over avatar pose/size, gate presence, marker presence, and panel state. This will exhaust all documented actions within the visible deadline while avoiding the clone-lineage slowdown of the earlier search.

## Preserved local probes

- `probe.py` (11441 bytes)
- `probe_handoff.py` (4051 bytes)
- `probe_remote.py` (2616 bytes)
- `probe_search.py` (2739 bytes)
- `probe_select.py` (1531 bytes)
- `probe_tutorial.py` (1485 bytes)

Run or inspect the smallest relevant probe before writing another one.
