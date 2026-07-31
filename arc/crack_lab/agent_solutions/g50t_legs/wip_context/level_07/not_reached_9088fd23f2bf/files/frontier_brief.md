# Unverified frontier brief: g50t level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace and begin with the indexed API surface, then reproduce any useful prior observations from `frontier_brief.md` before probing level 7. I’ll keep probes compact and clone-bounded, and put all reusable logic in `legs.py` with `play_level_7` only composing legs.
- The baseline is clean and reaches exactly level 6 (`moves=243`, replay valid), so level 7 is the only missing dispatch. I’m now inspecting only the harness entry points and the narrow existing-leg bodies needed to construct a level-7 clone and summarize its blobs/action deltas.
- The level-7 frame confirms the controlled avatar experimentally: the compact color-9 block at `(26,26)` moves by six pixels on successful actions (DOWN and LEFT from start), while the color-14 block moves left on every successful avatar move. The two occupy separated regions; initial avatar reachability is only 22 positions, with one reachable special surface at `(38,20)`. This strongly matches the existing cooperative gate-chain mechanic, so I’m testing that reusable leg at bounded larger depth before adding anything new.
- The first verified commit is `[DOWN, DOWN, LEFT, USE]`. It does not enlarge avatar reachability, but it removes exactly one six-pixel segment from the color-11 special strip and resets both movers. That explains why the existing cooperative search fails: it forbids reusing a special position, while this configuration appears to require repeated work on the same surface. I’m reproducing the repeated transition now and measuring strip area, reachability, and reward per cycle.
- Repeated identical commits merely alternate the strip, so that visual change is a gate phase, not progress. The likely missing behavior is timing: after opening that phase, the controlled ring must make additional legal track moves so the autonomous ring can exploit it before the next reset. I’m switching from macro-only search to a bounded action-level observational BFS, keyed on the world below the UI, to discover a winning timing sequence without assuming the interaction.
- The compact visible-state search exhausts without a win, which is evidence that level 7 depends on hidden state: visually identical USE outcomes cannot be merged. I’m now testing that directly by performing USE at each of the 22 reachable track positions, then replaying the verified special-surface commit and grouping the resulting worlds. This preserves concrete histories and should reveal which apparently inert first commit arms a distinct transition.
- The timing probe exposes the dense progress signal: with the gate phase opened, three successful “pump” moves reduce the special strip again (area 42 → 38) as the autonomous ring reaches the gate. The ring otherwise bounces back. I’m testing whether a USE immediately after that contact preserves the smaller strip across reset; if so, the general leg is a cycle of “open/advance autonomous/reset” driven by monotonically shrinking barrier area.
- The bottom color-9 line is only the turn budget, so I’m discarding it as a progress measure. The autonomous ring follows a 10-step patrol and reverses at row 44; the promising timing point is when it is adjacent to the color-11 barrier. The current macro always reaches the switch in the shortest three moves. I’m testing loop-padded routes to the same switch, so USE occurs at each patrol phase—an affordance the existing reach-by-position planners intentionally collapse.
- Loop-padding confirms USE resets to the same gate state at every patrol phase, so timing the press itself is not the mechanic. A previously excluded top-row color-9 marker changes among three positions and is therefore state, not scenery. I’m rerunning the bounded action search with that marker included while still excluding only the bottom turn-budget row; this should prevent the incorrect state merges seen in the compact search.
- Including the top marker still yields no Markovian action-state solution, confirming that visually identical states can have different futures here. I’m moving to a concrete-history macro search: every reachable-position→USE history remains eligible, several histories are retained even when their frames match, and plain movement reachability is rechecked after each stage. This directly addresses the hidden-state warning while keeping the search bounded.
- The hidden-history search is still running within its cap; output is buffered, so there is no intermediate stage count yet. I’ll let this bounded pass finish, then either reuse its discovered path or tighten the retained-history schedule based on the stage frontier it reports.

## Preserved local probes

- `probe_l7.py` (7147 bytes)
- `probe_macro.py` (1875 bytes)
- `probe_search.py` (817 bytes)

Run or inspect the smallest relevant probe before writing another one.
