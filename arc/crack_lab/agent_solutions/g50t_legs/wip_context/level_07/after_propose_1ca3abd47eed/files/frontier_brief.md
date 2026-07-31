# Unverified frontier brief: g50t level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside this workspace and use only the documented harness/perception surface. I’ll first inspect the indexed API map, then the prior frontier brief if present, and reproduce any observation I rely on with compact clone probes.
- The frontier brief indicates level 7 is a cooperative, timing-sensitive gate chain where avatar-position-only reachability collapses important helper phases. I’m treating that only as a hypothesis; next I’m running the baseline and will validate the visible state transitions from preserved small probes before changing code.
- The preserved prefix reproduces a narrow gate moment: helper row 32 → 26 → 32, then the run terminates without reward on the next patrol input. That confirms timing matters and also reveals the old probe was stepping past a failure terminal. I’ll now rebuild the level-7 start symbolically, verify all five actions on clones, and search from the last safe gate moment with terminal-aware replay.
- The baseline confirms the controllable avatar is the color-9 ring: actions 1/2/3/4 are directional movement, action 5 is contextual USE, and every successful avatar move advances the color-14 helper one track cell. The dense progress signal is therefore helper height plus newly exposed special surfaces—not raw frame delta. A bounded observational BFS is now testing whether visible state alone suffices before I add a history-aware leg.
- The action-level BFS confirmed why a generic frame-deduplicated search is too costly and unsafe here. I’m switching to the level’s real affordance: enumerate only movement histories that arrive on a visible switch, apply one USE commit, retain several hidden-history variants per identical visible result, and rank branches by helper height and barrier reduction. This keeps the search bounded while preserving synchronization phase.
- The switch-macro search is producing genuine new synchronization states: after five commits it reaches post-USE states with the helper still at row 26 instead of resetting to row 32. That is denser progress than the preserved attempt achieved. I’m continuing this bounded beam and will independently replay any winning path before turning the strategy into a reusable leg.
- The broader macro search now preserves seven distinct post-commit visual groups (plus hidden variants), matching the prior observation much more closely. It is intentionally slower because deep clones carry history; I’m watching stage growth and will tighten only dimensions that have shown themselves redundant, without collapsing the hidden phases the level requires.
- The template beam confirms the important distinction between phases that look similar: three commits can return to the same visible baseline via much shorter or longer histories, while the fourth changes the exposed switch surface. I’m retaining those histories but pruning the beam by dense progress so the remaining search stays practical.
- The wide template search retained over a hundred distinct phase groups through four commits, but some losing branches leave the helper in a malformed/dead runtime state that raises on later actions. I’ve isolated those to throwaway clones and now discard them explicitly; they are failure states, not progress.

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
