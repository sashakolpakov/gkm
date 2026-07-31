# Unverified frontier brief: cn04 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- Since no collision-free layout exists, overlap itself is part of the construction. The remaining check is per-figure validity: a joint that looks matched on the last-selected figure can be invalidated for an earlier figure by a third body sharing that cell. I’m inspecting all five figures in the final matched layout, not just the last one.
- Level 5 provides the decisive correction: it wins while its selected figure still has two visible green cells, so green matching is only an overlap indicator, not the objective. I’m measuring the known level-5 win transition directly—occupied area and physical component count immediately before/after the rewarding placement—to identify the true dense goal.
- The reward trace finally identifies the objective: level 5 first becomes one physical component at move 49, but `levels_completed` does not advance until move 54—a five-tick completion delay. Our level-6 layout was only given one post-connection tick. I’m replaying it with the full delay before judging the construction.
- The level-5 trace shows the useful progression is component merging: 4 → 3 → 2 → 1, followed by a short exact-registration adjustment. I’m now reproducing that strategy on level 6 with bounded clone BFS—one selected component at a time, first reducing physical components, then searching only the small final registration neighborhood for reward.
- The bounded merge search is still running; it is exploring selected-pose states, not the full five-piece state space. I’m letting the current bounded pass finish, then I’ll either replay its merge prefix or tighten the component key if one branch exhausted the cap.
- The merge prefix is verified: `5 → 4 → 3 → 2 → 1` in 32 actions. The broad finish search was wasteful, so I’m deduplicating the five selectable figures in that merged frame and giving each only a shallow local registration search—the analogous level-5 adjustment is four moves.
- The merge prefix is stable and yields seven distinct selectable states in the merged image. The current figure has no finish within the shallow neighborhood; the search is moving through the remaining deduplicated selections now.
- The successful level-5 policy has a stronger reusable structure: for each figure, choose a turn variant, then push it up and right until the existing assembly blocks it. I’m evaluating the five turn choices per level-6 figure with dense ranking (fewer components, then smaller occupied area), which is only 25 bounded rollouts and mirrors the proven earlier-level leg pattern.
- The push-policy pass exposed two false negatives: its preserved coordinates for pieces 1 and 4 landed in holes, so those selections never changed. I’ve replaced them with coordinates experimentally verified on occupied cells; the same dense evaluation can now actually place all five figures.
- The corrected push policy over-registered the pieces: it reached one component with only 26 occupied logical cells, whereas level 5 stays nearly non-overlapping until its final four-cell registration. I’m changing the dense score to minimize overlap loss while still reducing component count, rather than minimizing occupied area.
- The greedy turn choice still collapses too much information. I’m widening it to a small beam over turn combinations, deduplicated by frame and preserving different occupied-area outcomes. This covers the interaction between early turns and the final registration without paying the full 3,125-combination cost.
- The 40-state beam found no reward, so I’m removing the final pruning. After frame deduplication the true search is only about 1,600 turn combinations, and every rollout still uses the proven push policy. This will conclusively test that reusable policy family.

## Preserved local probes

- `layout_milp.py` (3475 bytes)
- `layout_search_clean.py` (4022 bytes)
- `layout_search_cover.py` (3873 bytes)
- `probe_branch.py` (2446 bytes)
- `probe_candidate.py` (1991 bytes)
- `probe_clean_connect.py` (2832 bytes)
- `probe_connect.py` (2560 bytes)
- `probe_final_bfs.py` (978 bytes)
- `probe_finish.py` (1946 bytes)
- `probe_goal.py` (981 bytes)
- `probe_known_shapes.py` (1625 bytes)
- `probe_l6.py` (2302 bytes)
- `probe_layout_search.py` (3780 bytes)
- `probe_piece_colors.py` (2401 bytes)
- `probe_prior.py` (1539 bytes)
- `probe_search.py` (2359 bytes)
- `probe_selections.py` (1953 bytes)
- `variant_layout_clean.py` (6168 bytes)
- `variant_layout_search.py` (5007 bytes)
- `wip_glue_notes.md` (1172 bytes)

Run or inspect the smallest relevant probe before writing another one.
