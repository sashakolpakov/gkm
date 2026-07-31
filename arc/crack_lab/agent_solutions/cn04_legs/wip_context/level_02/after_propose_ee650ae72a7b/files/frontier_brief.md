# Unverified frontier brief: cn04 level 2

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- Baseline is confirmed: level 1 finishes in 14 moves and level 2 remains. The preserved brief’s key claim to reproduce first is coordinate selection: if action 6 truly transfers control among figures, that changes the search from one-body navigation to staged cooperation.
- The selection behavior is reproduced: clicking a figure’s gray handle recolors that whole figure black, restores the prior figure’s color, and movement/rotation then acts on the newly selected body. The lattice also shows a four-body chain (A–D) with 2/4/4/2 handles, matching a relay or assembly task; I’m now testing which handle-pair transitions are physically realizable.
- The dense metric is now concrete and physically grounded: a completed joint leaves each gray peg visible with the moving body on one side and the target body on the opposite side. Bounded pose searches find exact bridge poses for A→B, B→C, and C→D. I’m adding the level-1 requirement that each mover first engages its own handle set, then testing the three-stage chain.
- The augmented search is larger than the pure pose graph because it preserves which handles have actually been engaged; it is still bounded, but this first 16k-state pass is slower than useful. I’ll take its partial result, then tighten the search around per-joint handle subsets if the full-set hypothesis is too broad.
- The first chain is geometrically plausible but not rewarded, including after handing control to D and exhaustively searching D’s pose space. The missing distinction is visible in level 1: a valid joint is not just an endpoint—it is entered by one rotation onto the two pegs and completed by two more consecutive turns. I’m constraining each relay segment to reproduce that verified three-turn socket maneuver.
- This stricter maneuver search is again in the handle-history state space, but it is testing a sharply defined invariant: own handles engaged, first turn covers the destination pair, third turn leaves a true bridge. I’ll use the first viable mover segments it returns and reject the model outright if any relay body has no such pose.
- Level 1 is symmetric: selecting the colored body and moving it back to the original body also wins, so control transfer is genuine and the primitive goal is docking one body at another body’s socket. The failed forward relay does not rule out the reverse docking order; I’m checking D→C→B→A next, where each later target remains in place until it has been used.
- The forward edge geometry has only a small finite frontier: six A→B bridge poses, one B→C pose, and two C→D poses. That makes exhaustive testing of every bridge variant and every staging order cheap and decisive, so I’m checking that entire macro-space rather than guessing another route.
- The bridge macro-space is exhausted with ordinary body clicks. One action-surface distinction remains: the gray cells are explicit selection handles, and the relay should transfer control through the joint just reached. I’m now replaying the same finite bridge set while clicking the actual bridged peg (including both handle choices), then testing the terminal D handoff.
- Handle-click handoff is visibly equivalent and does not solve. The remaining level-1 invariant is historical: the winning body covers both its departure pegs and the destination pegs before settling beside them. I’m preserving that peg-visit history explicitly for each mover; ordinary pose BFS cannot represent this because returning to the same pose after visiting a peg is observably identical.
- The visit-and-settle search is bounded at 30k augmented states per candidate and is taking the expected longer pass. Its result will tell us whether each proposed edge can even reproduce the full level-1 history; if so, the returned routes are immediately testable as one compact relay.
- The pair-history relay is also falsified. I’m broadening only one dimension: level 1 covers every peg at both ends, while B and C in level 2 each expose four pegs. I’ll build each mover’s ~1k-pose graph once and search masks on that compact graph, which lets me test “visit the entire next socket set, then dock” without another expensive clone explosion.

## Preserved local probes

- `probe_level1_finish.py` (2014 bytes)
- `probe_level1_neighborhood.py` (993 bytes)
- `probe_level1_paths.py` (607 bytes)
- `probe_level1_selection.py` (839 bytes)
- `probe_level2.py` (1822 bytes)
- `probe_level2_agents.py` (874 bytes)
- `probe_level2_attachment.py` (1291 bytes)
- `probe_level2_bridge_variants.py` (2504 bytes)
- `probe_level2_chain_candidate.py` (1930 bytes)
- `probe_level2_chain_search.py` (2740 bytes)
- `probe_level2_collect.py` (2478 bytes)
- `probe_level2_color_order.py` (1676 bytes)
- `probe_level2_connections.py` (2686 bytes)
- `probe_level2_contact.py` (2953 bytes)
- `probe_level2_coordinates.py` (1033 bytes)
- `probe_level2_edge_permutations.py` (2128 bytes)
- `probe_level2_edges.py` (1832 bytes)
- `probe_level2_handle_relay.py` (2682 bytes)
- `probe_level2_history_search.py` (2075 bytes)
- `probe_level2_joint.py` (1447 bytes)
- `probe_level2_maneuvers.py` (2915 bytes)
- `probe_level2_match.py` (1943 bytes)
- `probe_level2_ordered.py` (2248 bytes)
- `probe_level2_pair_transitions.py` (1424 bytes)
- `probe_level2_pairs.py` (1275 bytes)
- `probe_level2_pose_graph.py` (3336 bytes)
- `probe_level2_progress.py` (1104 bytes)
- `probe_level2_region_turns.py` (1447 bytes)
- `probe_level2_relay.py` (1864 bytes)
- `probe_level2_relay_coverage.py` (1743 bytes)
- `probe_level2_relay_pairs.py` (1331 bytes)
- `probe_level2_reverse.py` (2631 bytes)
- `probe_level2_reward.py` (741 bytes)
- `probe_level2_select_orders.py` (804 bytes)
- `probe_level2_selection.py` (1100 bytes)
- `probe_level2_shapes.py` (1080 bytes)
- `probe_level2_socket_turns.py` (1739 bytes)
- `probe_level2_stack.py` (1200 bytes)
- `probe_level2_stage_orders.py` (1695 bytes)
- `probe_level2_sweep.py` (1889 bytes)
- `probe_level2_turn_patterns.py` (1784 bytes)
- `probe_level2_use.py` (1293 bytes)
- `probe_level2_visit_bridge.py` (2751 bytes)
- `probe_mechanic_trace.py` (1716 bytes)

Run or inspect the smallest relevant probe before writing another one.
