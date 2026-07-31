# Unverified frontier brief: g50t level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- From the reproduced synchronization prefix, the exact-state search is much smaller—under 900 unique states by depth 15—and no longer needs duplicate visible histories once helper heading is explicit. It has not yet crossed the row-26 gate; I’m keeping the breadth-first run bounded and watching for either that crossing or reward.
- A single history per visible state is insufficient: it exhausts the apparent state graph without reward, matching the reproduced hidden-phase warning. I’ve verified the movement prefilter has no false negatives on the known cooperative handoff, and I’m rerunning the same bounded graph while retaining four concrete histories per avatar/helper/gate/heading state.
- Four-history search remains stable at depth 26 and has covered about 1,700 distinct visible/heading states; helper row 26 is still the best dense-progress point. The queue is now shrinking, so this run should soon tell us whether four histories capture the missing phase or whether the branch identity must be represented more directly.
- The four-history cap still closes the graph without reward, so “keep more arbitrary copies” is not a reliable state model. I’m replacing that cap with an explicit physical clock: successful movement count and valid USE count, alongside avatar/helper positions, gates, marker, and helper heading. This preserves completed patrol laps without multiplying equivalent path permutations.
- The physical-clock graph grows too broadly from the untouched start, so I’m seeding it at the independently replayed first handoff—the point where the helper has crossed its first barrier and the upper switch becomes reachable. This preserves every timing choice that matters for the second gate while removing already-settled opening permutations.
- The prioritized clock search has now tested synchronization histories out past 100 actions from the first handoff, including up to twelve valid commits, without ever moving the helper above row 26. That is strong evidence the missing transition is not “wait/use longer” on the two assumed switch rows; I’m letting the finite queue close, then I’ll inspect reachable special surfaces at the row-26 frontier for an affordance the row filter omitted.
- Allowing synchronized moves into currently closed-looking cells changed the reachable graph substantially, but the helper still never passes row 26 across thousands of long histories. The remaining likely omission is a valid interaction away from the two obvious surfaces, so after this bounded queue settles I’ll probe USE by actual marker/world change at every reachable state instead of by row.
- That probe found the omitted affordance immediately: after the first handoff, USE is valid at the reset/start positions `(32,26)` and `(26,26)`, even though it was inert there at the untouched level start. This is the hidden contextual switch the row-based planners missed. The search is now including these confirmed interactions and retaining their physical clocks.
- The contextual USE result suggests a better abstraction than action-by-action depth: enumerate bounded movement histories between valid commits, test USE observationally at every reached state, and keep several post-commit histories. I’ve updated the macro beam to do exactly that, including patrol-lap depth in its movement key and per-branch helper progress.
- The contextual macro beam is now producing many more real commit groups (29 after two commits). Its best-ranked branch repeatedly uses the reset tile, which preserves the helper’s row-26 progress but may overvalue cycling; I’m allowing a few stages to see whether it opens a new barrier, then I’ll adjust ranking toward newly exposed movement if it stalls.
- Penalizing repeated reset-tile commits changed the beam materially: by stage four its best branch ends with the helper at row 26 while a barrier segment is visibly open (`15` area 192 versus 203). That is the narrow crossing state we wanted; I’m continuing from these six branches to see whether the next movement crosses rather than bounces.
- The diversity-aware beam now reaches three distinct interaction rows and again arrives at the row-26/open-barrier state, but through a different sequence that includes the upper surface and one contextual reset use. This is a much more plausible gate-chain branch than repeated bottom commits.

## Preserved local probes

- `probe_altpath.py` (2729 bytes)
- `probe_chain.py` (1835 bytes)
- `probe_cycle2.py` (2406 bytes)
- `probe_gate2.py` (2441 bytes)
- `probe_gate3.py` (2218 bytes)
- `probe_groups.py` (4104 bytes)
- `probe_hidden.py` (2839 bytes)
- `probe_l7.py` (7147 bytes)
- `probe_l7_baseline.py` (1443 bytes)
- `probe_l7_macro_state.py` (8257 bytes)
- `probe_l7_state_bfs.py` (5334 bytes)
- `probe_l7_symbolic.py` (2837 bytes)
- `probe_latch.py` (1867 bytes)
- `probe_long.py` (2085 bytes)
- `probe_macro.py` (1875 bytes)
- `probe_macro_beam.py` (4774 bytes)
- `probe_next.py` (2204 bytes)
- `probe_repeat.py` (2511 bytes)
- `probe_retime.py` (1638 bytes)
- `probe_search.py` (817 bytes)
- `probe_stage2.py` (1744 bytes)
- `probe_stage3.py` (1901 bytes)
- `probe_stage_search.py` (3850 bytes)
- `probe_template_beam.py` (4661 bytes)
- `probe_timing_trace.py` (1425 bytes)
- `probe_top_phase.py` (2256 bytes)
- `probe_trace3.py` (1346 bytes)

Run or inspect the smallest relevant probe before writing another one.
