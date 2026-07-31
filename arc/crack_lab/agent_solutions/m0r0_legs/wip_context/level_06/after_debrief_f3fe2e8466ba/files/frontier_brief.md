# Unverified frontier brief: m0r0 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- Baseline is confirmed at 5 levels and 182 moves with a valid replay. The frontier claims are still only hypotheses; I’m now reproducing the smallest useful observations from the preserved probes and checking the documented clone/action surface before relying on them.
- The reproduced mechanics are now solid: the mirrored 4×4 pair opens the two central barriers only while standing on the upper contacts; coordinate action 6 selects the tiny 2×2 agent; selected motion recolors it and lets it cross the opened right chamber. The dense subgoal is therefore switch occupancy plus tiny-agent reachability, while the edge “cracks” are a turn deadline. I’m narrowing the coordinate-aware search around actual selectable components.
- The broad BFS is too expensive in this harness, so I stopped it before the cap rather than burning the clone budget. I’m switching to compact staged experiments: map the 4×4 geometry, enumerate only meaningful switch/agent handoffs, and use replay-path states instead of retaining deep clones.
- The staged handoff reveals the key asymmetry: placing the tiny agent directly under one main avatar pins that avatar while its mirrored partner moves independently. Each upper contact opens only its same-colored barrier, so the solution must deliberately transfer contact occupancy. I’m testing the minimal transfer sequence now, including whether the tiny agent itself activates a vacated contact.
- A decisive candidate emerged: the tiny agent cannot occupy the contacts itself; its reusable affordance is pinning one member of a mirrored pair. That lets the other avatar traverse the barrier kept open by its pinned partner. I’m checking whether reaching the lone lower contact is the actual sparse-reward event.
- Touching the lower colored body alone does not award the level, so I’m checking the remaining context-sensitive affordance at that exact contact rather than assuming it is passive. This also distinguishes “reach the marker” from “interact while staged.”
- The lower colored body is the second handoff point, not the finish: the first avatar can hold that barrier open while the tiny agent circles underneath and pins it. This permits the other avatar to descend. I now have a concrete cooperative route to put both mains in the lower chamber, then use the tiny agent as a horizontal stop so the mirrored pair can reunite.
- The attempted one-sided crossing resets the level, confirming that the center is a synchronized reunion aperture: both mains must enter it together from opposite sides. I’m testing whether the lower contact latches its barrier after being visited; that persistence is the missing state needed to get both avatars below simultaneously.
- The lower contact is not latched. A more general physical affordance remains: a solid movable agent parked inside a temporarily removed barrier may prevent that barrier from respawning. I’m testing that directly, offsetting the tiny agent so it wedges the gate without blocking the main avatar’s lane.
- The wedge works: the barrier respawns around the tiny agent but leaves a full 4-pixel lane at that column. I’m verifying that moving the tiny agent out preserves the punched lane and lets one main pass; if so, the same operation can be repeated on the opposite barrier before the final synchronized descent.
- The gate cannot be permanently wedged: it fills as soon as the tiny agent exits. Manual staging has exposed a circular dependency, so I’m returning to a tightly capped symbolic search—5,000 states, coordinate choices only on currently controllable components—to find the overlooked handoff without revisiting crack-age variants.
- The single-step clone search remains too slow even at 5,000 states, so I stopped it. I’m compressing repeated directional moves into macros and scoring states by a verified dense measure: both main avatars at the same lower-row reunion lane, eight pixels apart, with selectable/gate state preserved and crack age removed.

## Preserved local probes

- `probe_clock.py` (990 bytes)
- `probe_coop.py` (804 bytes)
- `probe_coord_bfs.py` (1446 bytes)
- `probe_handoffs.py` (2246 bytes)
- `probe_l6.py` (3836 bytes)
- `probe_latch.py` (869 bytes)
- `probe_macro_search.py` (2684 bytes)
- `probe_previous_goal.py` (954 bytes)
- `probe_routes.py` (1556 bytes)
- `probe_search.py` (998 bytes)
- `probe_wedge.py` (925 bytes)

Run or inspect the smallest relevant probe before writing another one.
