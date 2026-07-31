# Unverified frontier brief: cn04 level 2

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside this workspace and the documented harness surface. I’ll begin with `solver_index.md`, then reproduce any inherited observations through bounded clone probes before editing the leg/player composition.
- The baseline is reproduced: level 1 completes in 14 moves and execution stops at level 2. I’ll now inspect only the indexed player/leg definitions plus the public harness API needed to create and observe level-2 clones.
- The first probe reproduces the key mechanics cleanly: actions 1–4 translate the black 90-cell piece, action 5 rotates it, and action 6 is inert in the initial context. The level contains three fixed colored silhouettes and twelve 3×3 pegs; a generic 10k-state visible-frame BFS does not find reward, so history/context matters beyond the current avatar pose.
- The reproduced dense-score probe shows the top-row count is only clipping as the avatar leaves the screen, so I’m discarding it. More useful is the silhouette probe: level 2’s controlled piece has 10 logical cells, and the first fixed figure plus its nearby pivots appears to define a grappling/rotation route—not a simple overlay target. I’m narrowing the model by measuring pose transitions at each reachable peg and the exact reward-triggering contact from level 1.
- A decisive level-1 fact is now reproduced: the winning route never overlaps the colored figure before reward. It first covers the two starting pegs, then reaches the other pair on the first rotation, and wins on the third consecutive rotation. That makes “engage a socket, then complete a three-turn maneuver” the strongest reusable subgoal. I’m testing that maneuver across the ordered peg-chain encoded by level 2.
- The uniform “three-turn maneuver at every edge” is falsified. The transition graph gives a cleaner interpretation: within the first colored socket, one rotation advances `(2,4) → (0,2) → (0,5)`, so those are successive states of one maneuver, not three independent objectives. I’m now testing one entry socket per colored region, with consecutive turns allowed to carry the avatar through the region.
- The one-entry-per-region model is also falsified. There is one untested part of the documented action surface: action 6 may be coordinate-sensitive even though bare `step(6)` is inert. I’m probing it at avatar cells, pegs, colored figures, and background points; this could be the missing selection/attachment mechanic that all key-only searches omitted.
- Coordinate action 6 is the missing mechanic. Clicking a peg belonging to the 90-cell figure changes exactly 180 pixels (the current 90-cell avatar plus that figure); clicking the 162-cell or 45-cell figures changes 252 or 135 pixels respectively. This strongly indicates selection transfers control between figures, which explains why a single avatar cannot reach all relevant structures. I’m now verifying the recoloring and movement of each selected figure, then I’ll plan cooperative subgoals per figure.
- Selection is a reversible control transfer: the previously controlled figure regains its own color, and the clicked figure becomes black; the pegs remain fixed selection handles. I’m checking each agent’s independent reachable state space for a direct win before expanding into cooperative multi-agent search—this separates “one correct agent” from “stage several figures.”
- No single selectable figure can complete level 2 across its entire independent pose space, confirming a cooperative/staging requirement. The level-1 winning rotation is consistent with bringing two figures into a stacked/contact configuration, so I’m testing the lowest-complexity cooperative hypothesis next: align the three colored figures over the original avatar in all quarter-turn orientations, transferring control between them via their fixed handles.
- Simple stacking is falsified. The stronger cooperative model is a relay: level 1 moves the active figure into the next figure’s peg socket and performs the finishing turns; level 2 lays out `A → B → C → D`, with action 6 transferring control after each stage. I’m enumerating which pairs of the next region’s pegs each differently shaped agent can actually cover, respecting its own geometry.

## Preserved local probes

- `probe_level1_finish.py` (2014 bytes)
- `probe_level1_neighborhood.py` (993 bytes)
- `probe_level1_paths.py` (607 bytes)
- `probe_level2.py` (1822 bytes)
- `probe_level2_agents.py` (874 bytes)
- `probe_level2_attachment.py` (1291 bytes)
- `probe_level2_collect.py` (2478 bytes)
- `probe_level2_color_order.py` (1676 bytes)
- `probe_level2_contact.py` (2953 bytes)
- `probe_level2_coordinates.py` (1033 bytes)
- `probe_level2_edge_permutations.py` (2128 bytes)
- `probe_level2_edges.py` (1832 bytes)
- `probe_level2_history_search.py` (2075 bytes)
- `probe_level2_match.py` (1943 bytes)
- `probe_level2_ordered.py` (2248 bytes)
- `probe_level2_pair_transitions.py` (1424 bytes)
- `probe_level2_pairs.py` (1275 bytes)
- `probe_level2_progress.py` (1104 bytes)
- `probe_level2_region_turns.py` (1447 bytes)
- `probe_level2_relay.py` (1864 bytes)
- `probe_level2_relay_coverage.py` (1743 bytes)
- `probe_level2_relay_pairs.py` (1331 bytes)
- `probe_level2_reward.py` (741 bytes)
- `probe_level2_select_orders.py` (804 bytes)
- `probe_level2_selection.py` (1100 bytes)
- `probe_level2_shapes.py` (1080 bytes)
- `probe_level2_socket_turns.py` (1739 bytes)
- `probe_level2_stack.py` (1200 bytes)
- `probe_level2_sweep.py` (1889 bytes)
- `probe_level2_turn_patterns.py` (1784 bytes)
- `probe_level2_use.py` (1293 bytes)
- `probe_mechanic_trace.py` (1716 bytes)

Run or inspect the smallest relevant probe before writing another one.
