# Unverified frontier brief: g50t level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The brief’s claims are only hypotheses. I’m reproducing the baseline now, then I’ll inspect just the documented perception functions and the smallest preserved probes needed to validate avatar movement, autonomous timing, and the staged switch cycle.
- The first preserved cycle is reproducible but not a complete strategy: it grows avatar reachability from 22 to 25, exposes a second special surface, then loses that progress after the second commit. I’m now measuring the short post-commit timing window rather than treating visually similar states as identical.
- The naive retiming probe found no growth because it replayed the wrong post-commit phase. The preserved adaptive probe instead searches a bounded “press reachable switch, then patrol briefly” cycle while keeping hidden histories distinct; it is running now, and I’ll use its concrete path only if replay independently earns the level reward.
- The reproduced timing window is concrete: after committing the remote surface, one bottom-surface press arms the mover; a second bottom press with a short reversible delay, followed by ten patrol moves, re-exposes the remote surface and raises reachability to 24. I’m testing whether that verified cycle composes repeatedly and eventually exposes the goal before generalizing it into a leg.
- The cycle is not stationary: after the second remote commit the autonomous mover is parked at the lower barrier, so replaying the previous timing verbatim does nothing. I’m enumerating only ten delay phases and a 30-move patrol from that exact state, scoring true reward first, then reachable-area growth and exposed surfaces.
- The apparent dead end is a synchronization issue at the remote surface, not proof the cycle fails: the shortest remote walk advances the autonomous mover into a closed lower barrier. I’m varying only the reversible delay before that second remote commit and recording mover/barrier transitions, which isolates the phase choice without broad search.
- The second correctly phased remote commit creates genuine dense progress: after one further switch use, the helper reaches row 26 (up from row 44) but then oscillates below the next barrier. I’m probing from the exact row-26 moment; the next useful commit must be made there, so end-of-patrol snapshots would miss it.
- The row-26 probe shows that simple waiting is insufficient. The switch must be approached along a different movement history: shortest-path reachability collapses paths that leave the avatar at the same coordinate but put the autonomous helper in different phases. I’m now enumerating bounded movement-only histories to the two reachable surfaces, keyed by the visible mover/barrier state, then testing USE on those distinct arrivals.
- The bounded arrival search confirms 20 visibly distinct ways to reach the exposed surfaces, but one USE still cannot cross the row-26 barrier. I’ve lifted the search one level: it now preserves up to four hidden histories per visible post-USE state and explores repeated surface commits, ranked by the helper’s upward progress.
- This first beam is too expensive because it recomputes full avatar reachability for every retained history. I’m stopping it and tightening the abstraction around the verified gate moments, keeping clone exploration within the requested budget.
- The replay-based version avoids deep-clone growth and reduces the first post-USE layer to seven observable groups. It is still slower than ideal at later layers, so I’m watching the bounded stage count closely and will stop if it fails to produce new upward/helper states.
- The first post-row-44 commit layer has nine visible variants, all with the helper at row 32 and the same avatar reachability; their difference is timing/history. I’m testing one representative of each through exactly one further switch commit, preserving the variant identity rather than expanding a broad beam.

## Preserved local probes

- `probe_altpath.py` (2729 bytes)
- `probe_chain.py` (1835 bytes)
- `probe_cycle2.py` (2406 bytes)
- `probe_gate2.py` (2441 bytes)
- `probe_gate3.py` (2218 bytes)
- `probe_groups.py` (4104 bytes)
- `probe_hidden.py` (2839 bytes)
- `probe_l7.py` (7147 bytes)
- `probe_latch.py` (1867 bytes)
- `probe_long.py` (2085 bytes)
- `probe_macro.py` (1875 bytes)
- `probe_next.py` (2204 bytes)
- `probe_repeat.py` (2511 bytes)
- `probe_retime.py` (1638 bytes)
- `probe_search.py` (817 bytes)
- `probe_stage2.py` (1744 bytes)
- `probe_stage3.py` (1901 bytes)
- `probe_stage_search.py` (3850 bytes)
- `probe_top_phase.py` (2256 bytes)
- `probe_trace3.py` (1346 bytes)

Run or inspect the smallest relevant probe before writing another one.
