# Unverified frontier brief: cn04 level 2

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside this workspace and begin with the indexed API surface, then reproduce any useful prior observations before probing level 2.
- The baseline is clean: level 1 completes in 14 moves and dispatch reaches level 2, but no level-2 player exists yet. I’m reproducing the prior symbolic geometry and ordered-peg observations from the smallest preserved probes before designing the reusable leg.
- The reproduced geometry gives a stronger dense measure than generic peg visitation: level 1 rewards only at one exact placement, after a three-rotation maneuver around the target’s side and support pegs. I’m now testing exact silhouette coverage around each level-2 colored figure; that will distinguish “touch every peg” from “seat the controlled shape into successive sockets.”
- A clean structural pattern emerged: the simultaneously coverable peg pairs around each colored figure form a path graph—three edges around color 14, then three around color 11, then one final edge around color 9. That explains why merely visiting all 12 pegs failed. I’m testing the four possible traversal directions through the two middle path graphs, using “completed ordered edges” as the dense progress state.
- Level 1 is not generally path-dependent: alternate routes and harmless detours all reward at the same final three-rotation maneuver. The remaining ambiguity is the order of the three valid socket edges around each middle figure, so I’m exhaustively testing those 36 compact orderings with cached pose-to-edge routes rather than expanding raw action histories.
- The 36-order probe is still running but remains within the bounded pose graph; it has produced no rewarded ordering yet. I’m letting this finite test complete before changing the mechanic hypothesis.
- The edge-order hypothesis alone is falsified. A key observation is that rotations connect some socket-pair states directly, while others are isolated, suggesting the game records a sequence of context-specific rotation engagements rather than a simple “visit all edges” set. I’m switching to a bounded best-first clone search keyed by avatar pose plus the first-time order of rotated socket engagements, with the number of distinct engagements as the dense progress score.
- The history-aware search is still within its 45,000-state cap and has not emitted a rewarded path yet. Output is buffered by the harness, so I’ll report its final explored frontier and best engagement count once it terminates.
- The search visited all eight inferred socket edges in multiple orders without reward, so those pegs are affordances, not the goal state. I’m returning to the level-1 fact that reward occurs at one geometric seating maneuver and testing the colored figure together with its local pegs as a complete target silhouette; the colored regions alone omitted cells visibly occupied by pegs.

## Preserved local probes

- `probe_level1_finish.py` (2014 bytes)
- `probe_level1_neighborhood.py` (993 bytes)
- `probe_level1_paths.py` (607 bytes)
- `probe_level2.py` (1822 bytes)
- `probe_level2_attachment.py` (1291 bytes)
- `probe_level2_collect.py` (2478 bytes)
- `probe_level2_color_order.py` (1676 bytes)
- `probe_level2_contact.py` (2953 bytes)
- `probe_level2_edge_permutations.py` (2128 bytes)
- `probe_level2_edges.py` (1832 bytes)
- `probe_level2_history_search.py` (2075 bytes)
- `probe_level2_match.py` (1943 bytes)
- `probe_level2_ordered.py` (2248 bytes)
- `probe_level2_pair_transitions.py` (1424 bytes)
- `probe_level2_pairs.py` (1275 bytes)
- `probe_level2_progress.py` (1104 bytes)
- `probe_level2_reward.py` (741 bytes)
- `probe_level2_shapes.py` (1080 bytes)
- `probe_level2_sweep.py` (1889 bytes)
- `probe_level2_turn_patterns.py` (1784 bytes)
- `probe_level2_use.py` (1293 bytes)

Run or inspect the smallest relevant probe before writing another one.
