# Unverified frontier brief: ka59 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll work strictly inside the workspace and begin with `solver_index.md`, then use `frontier_brief.md` if present. I’ll keep observations compact, reproduce any inherited claims on clones, and make the level player compose only reusable legs.
- Baseline is clean and stops at level 7 (`levels=6`, 210 moves). I’m treating the frontier notes only as hypotheses; I’ll now replay the smallest preserved level-7 probes and inspect just the indexed player/leg and clone helper ranges needed to reproduce the mechanics.
- The clone evidence now confirms the essentials: only the two color-14 rings are selectable; arrows move the selected ring in three-pixel steps; both color-13 corridor objects advance one phase on every arrow; and the reward remains sparse after seating both small rings. The remaining verified subgoal is therefore moving the isolated color-11 ring toward its lower-left target. I’m tracing the already-solved level-6 transfer at object level, then I’ll test the corresponding level-7 wall contacts.
- A useful new mechanic is now reproduced: the lower cycling object can independently lift an unselected vertical ring across the sealed lower barrier, so both selectable rings can be staged in the central region. The apparent narrow-wall route is a dead end even with interlocking and selection swaps. I’m using a bounded symbolic search from the verified two-ring staging state, with the goal defined specifically as the first color-11 displacement rather than any frame change.
- The bounded search is still working through the two-ring state graph and has not produced a displacement yet. I’m keeping the fixed 30,000-state cap; if it exhausts, I’ll use its reachable-state evidence to split the search at the cycling-agent handoff instead of increasing the bound.
- The first broad graph exhausted after only 5,044 distinct observational states with no large-ring contact, which rules out the “lift both rings centrally first” branch. The alternate relay graph is much larger and is exceeding the useful probe budget, so I’m stopping it and reducing it to verified waypoint states around the upper and left barriers.

## Preserved local probes

- `probe_l4.py` (3341 bytes)
- `probe_l5.py` (3037 bytes)
- `probe_l6.py` (5807 bytes)
- `probe_l6_final.py` (956 bytes)
- `probe_l7.py` (2431 bytes)
- `probe_l7_agent_handoff.py` (1465 bytes)
- `probe_l7_build.py` (1301 bytes)
- `probe_l7_contact_bfs.py` (1275 bytes)
- `probe_l7_focus.py` (1670 bytes)
- `probe_l7_gap_bfs.py` (1947 bytes)
- `probe_l7_geometry.py` (1107 bytes)
- `probe_l7_interlock_trace.py` (791 bytes)
- `probe_l7_large_bfs2.py` (2766 bytes)
- `probe_l7_large_transfer.py` (986 bytes)
- `probe_l7_left_barrier.py` (1425 bytes)
- `probe_l7_linked_agents.py` (1214 bytes)
- `probe_l7_push.py` (988 bytes)
- `probe_l7_relay.py` (1953 bytes)
- `probe_l7_search.py` (4469 bytes)
- `probe_l7_select.py` (1549 bytes)
- `probe_l7_state.py` (2858 bytes)
- `probe_l7_top_contacts.py` (1410 bytes)
- `probe_l7_trigger.py` (1329 bytes)
- `probe_l7_upper_push_offsets.py` (1458 bytes)
- `probe_remote_push.py` (811 bytes)
- `probe_transfer_trace.py` (1310 bytes)

Run or inspect the smallest relevant probe before writing another one.
