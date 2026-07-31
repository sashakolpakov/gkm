# Unverified frontier brief: lf52 level 9

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Verified parent budget

- Exact parent boundary: level 8 at 544 actions.
- Remaining real-action budget under the harness cap: 56 of 600.
- This budget is verifier evidence. If it is insufficient, optimize earlier composed legs rather than searching an uncommittable suffix.

## Prior proposer progress

- The integrated leg hits the reproduced cap exactly at 600, still on level 9, confirming the 56-action checkpoint remainder is the sole blocker. I’m now replaying the composed leg on a clone at pristine level entry to verify the transcription still wins in exactly 102 actions.
- The level-9 leg and its 102-action JSON candidate are now clone-verified. I’m making one more focused prefix pass: decomposing level 5 into key runs and bridge/capture macros from parsed states, so I can see which of its 16 lattice moves and 57 carrier steps are structural versus avoidable.
- Level 5’s trace shows its long key runs move multiple loaded carriers around barriers; the first two are exact Manhattan minima, while the 15-step wrap is structural. I’m checking levels 2–4 with the exact key-run optimizer now—their smaller nonwrapped carrier graphs can yield quick, provable savings if the macro-BFS chose nonminimal action-cost routes.
- The earlier prefix appears locally tight, so I’m returning to level 9’s remaining untested context: directional keys while a peg or movable bridge is selected. If selection changes key semantics—especially for the loaded carrier peg—it could bypass the 74-action remote leapfrog and fit the 56-action budget.
- A more promising untested mechanic remains: I only exhaustively tested peg landings, not color-9 bridge landings. With multiple bridge agents, they may support long or chain jumps in one source/destination pair. I’m now trying every same-row/column empty landing for each bridge at entry and relay alignments, accepting only actual piece displacement.
- Bridge jumps are strictly one occupied midpoint; no long landings were valid. I’m extending that exhaustive check to diagonal destinations as the last geometric affordance not covered by the orthogonal model.
- Orthogonal-only behavior is confirmed for color-9 bridges. The color-15 arrow bridges still need a two-click movement test: at the corridor endpoint one sits at `(24,22)`, directly below a color-9 bridge with an empty landing at `(12,22)`. I’m testing whether that apparently legal leap moves the arrow bridge; a positive result would invalidate the “persistent bridge” assumption and open a much shorter plan.
- I’ve also updated the source index. I’m recording the reproduced level-9 mechanics, verified winning leg, and exact accounting blocker in `frontier_brief.md` so the next clean attempt can start from compact facts rather than the exploratory transcript.
- The implementation and candidate are preserved, so I’m removing only the temporary exploratory/optimization scripts I created. They contain no unique solution data; the compact verified frontier and JSON suffix retain the reproducible result.
- One final context could still beat the bound: selecting a remote bridge, scrolling the wrapped world, then clicking its landing. I only tested selected pieces before the remote scrolling phase. I’m checking whether selection persists across remote carrier shifts; if it does, one click pair could replace many leapfrog moves.
- Remote scrolling cancels selection; subsequent clicks simply select whatever is now under the coordinate. That shortcut is falsified. I’m removing this last temporary probe; the preserved implementation/frontier remains unchanged.
- Implemented level 9, but the supervisor-owned checkpoint budget blocks campaign completion. - Added the verified 102-action leg in [legs.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_lf52_l9_canonical_fresh_max180_after_fe713_20260728/legs.py:593). - Added level-9 dispatch in [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_lf52_l9_canonical_fresh_max180_after_fe713_20260728/players.py:45). - Clone verification: `levels_completed=9`, exactly 102 actions. - Preserved the suffix in [level9_candidate_102.json](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_lf52_l9_canonical_fresh_max180_after_fe713_20260728/level9_candidate_102.json). - Updated [frontier_brief.md](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_lf52_l9_canonical_fresh_max180_after_fe713_20260728/frontier_brief.md). Final harness result: ```text RESULT levels=8 moves=600 replay_ok=True err=RuntimeError: real-move/time cap ``` The validated prefix costs 544 actions, leaving 56. Under all verified mechanics, level 9 requires 102, so at least 46 prefix actions must be removed before the candidate can be adopted. `checkpoint.json` was not modified.

## Preserved local probes

- `level9_candidate_102.json` (1355 bytes)
- `level9_full_candidate.json` (5473 bytes)
- `wip_glue_notes.md` (444 bytes)

Run or inspect the smallest relevant probe before writing another one.

## Verified clean continuation

- Replayed `checkpoint.json` observationally without modifying it. Exact
  transitions are `(1,8), (2,42), (3,87), (4,149), (5,238), (6,331),
  (7,476), (8,544)`.
- Reproduced the pristine level-9 action surface:
  `env.actions == (1, 2, 3, 4, 6, 7)`. Coordinate action 6 selects and moves
  visible pieces; key 4 advances the entry carrier; action 7 undoes/deselects
  and is recorded as a real replay action.
- The entry board was extracted and solved as an abstract leapfrog graph.
  Its carrier-loading solution is 14 piece moves (28 clicks), with 127
  reachable abstract states; the preserved candidate uses this minimum.
- Exhaustive visible landing probes at entry and remote alignments found only
  orthogonal one-midpoint jumps. Long and diagonal shortcuts were not valid.
- `solve_multi_bridge_wrapped_carrier_peg_solitaire` and
  `level9_candidate_102.json` independently reach level 9 in 102 actions and
  produce the same final frame.
- Reward-verified deletion minimization reduced level 4 from 62 to 50 actions
  (`level4_ddmin_50.json`) and level 6 from 93 to 92 actions
  (`level6_ddmin_92.json`). Levels 5, 7, 8, and 9 admitted no verified
  contiguous macro/action deletion in the bounded probes.
- All eleven level-7 key runs were checked for shorter exact carrier-state
  paths: the first four by forward BFS and the remaining seven by
  bidirectional BFS. No replacement preserved the full reward suffix.
- A carrier-closure macro search tested the only alternate level-7 entry
  order. Loading the bridge first is initially symmetric (both payloads are
  staged by action 29), but its unique non-reversing transfers enter a carrier
  cycle and end at action 75 with no legal move in the full 64-state closure.
  No validated macro suffix spliced into the alternate states. On the
  validated open board, the only branch from the forced peg advance moves the
  bridge backward and likewise enters carrier cycles.
- `campaign_candidate_633.json` composes the two optimized prefix segments
  with the validated remaining checkpoint legs and level 9. A pristine
  level-1 clone reaches level 9 at action 633, with transitions
  `(1,8), (2,42), (3,87), (4,137), (5,226), (6,318), (7,463), (8,531),
  (9,633)`.
- `players.py` now dispatches level 9 by composing the existing multi-bridge
  leg. The real resumed harness result is:
  `RESULT levels=8 moves=600 replay_ok=True err=RuntimeError: real-move/time cap`.

The supervisor checkpoint still costs 544, leaving 56 actions for a
clone-verified 102-action level-9 leg (46-action gap). Even if the independently
verified 531-action optimized prefix is adopted, the full campaign costs 633,
so a further 33-action replacement—not a deletion or shorter exact key
alignment—is required.
