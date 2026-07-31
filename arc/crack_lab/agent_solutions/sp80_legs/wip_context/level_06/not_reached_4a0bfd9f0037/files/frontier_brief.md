# Unverified frontier brief: sp80 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace and begin with the indexed source map, then reproduce any frontier observations before probing level 6. I’ll keep all experiments bounded and make `play_level_6` a composition of reusable legs only.
- Baseline is confirmed at level 5 (`moves=111`, replay valid). I’m now narrowing the level-6 geometry from compact object/delta summaries and checking only the existing player/leg interfaces needed for a clean implementation.
- The mechanics are now independently confirmed: action 6 selects one of four movable pieces, directions translate it by 3 pixels, and action 5 tests the arrangement. The board has three side-facing sockets plus one bottom-facing socket; I’m using lower-level win perturbations to infer each socket’s exact projection constraint before expanding the search.
- The central-stack family is exhausted without a reward hit, so simple gap-free connectivity is not the full rule. I’m isolating the hidden acceptance invariant on level 2, where all positions can be enumerated cheaply, then I’ll transfer that verified invariant to the four level-6 pieces.
- The lower-level enumeration confirms the useful invariant: wins occur when the movable projections bridge all port intervals as a continuous cover, not at one arbitrary absolute placement. Level 6 still has an additional directional/ordering constraint, so I’m sweeping the only unresolved coordinate—the vertical bar’s horizontal ray position—while exhaustively covering all socket-aligned rows.
- The sweep is progressing normally at the requested action cap. I also corrected an important probe artifact: some earlier branches selected an overlapping piece at its old click point, so I reordered selection to preserve object identity; the current search is testing genuine four-piece states.
- The corrected sweep has passed the previously problematic overlap regions without losing object identity. It has not found a win yet; I’m letting the bounded socket-aligned family finish before changing any solver code.
- The run remains healthy and capped; its silence means no reward hit among the tested states so far. The remaining branches are the lower-row combinations where the vertical bar changes which side-facing socket sees it first.
- The vertical-bar sweep exhausted 25,872 genuine states with no reward, which rules out its ray position as the missing variable. I’m now varying the two turn pieces and the marked horizontal bar together around exact physical junctions; this is the smallest remaining family consistent with both the projection invariant and directional line-of-sight.

## Preserved local probes

- `probe_axis_ranges.py` (1883 bytes)
- `probe_central_stack.py` (3763 bytes)
- `probe_chain_candidates.py` (1824 bytes)
- `probe_constraints.py` (4143 bytes)
- `probe_l1_map.py` (979 bytes)
- `probe_l2_acceptance.py` (2569 bytes)
- `probe_l6.py` (9720 bytes)
- `probe_search_l6.py` (4397 bytes)
- `probe_snapshots.py` (1466 bytes)

Run or inspect the smallest relevant probe before writing another one.
