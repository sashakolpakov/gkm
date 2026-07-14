# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## tr87 level 1 (tile-cycle matching)
- Frame: a static reference row (color 10) plus an editable row of N glyph
  "tiles" (color 7) with a bracket cursor (color 0). Actions 1/2 = cycle the
  current tile forward/back through a small fixed glyph set (period 7 here);
  actions 3/4 = move cursor left/right between tiles. Only the current tile
  changes on an edit; only the cursor moves on a move.
- The visible reference row (10-strip) is NOT the match target (its glyphs are
  not even reachable). A row-63 bar just counts total edits (cosmetic, not
  correctness). => reward is OPAQUE: no per-tile feedback.
- Win = every tile set to a hidden target glyph. Solved by searching a clone's
  small reachable config space (cycle^n_tiles) for a config that raises
  levels_completed, then replaying the path.
- New legs: discover_tile_cycle_puzzle (auto-finds edit/move actions, n_tiles,
  cycle from action deltas), search_tile_cycle_config (nested-clone DFS w/
  early exit; last tile reuses one clone), solve_tile_cycle_puzzle (compose).
- Solution config for L1: per-tile edit-counts [5,5,3,6,5] (tile4 also 6 wins).

## Debrief refactor (post-L1): extract shared skills, players stay thin
- `players.play_level_1` is already thin (one call to `solve_tile_cycle_puzzle`).
  The duplication lived INSIDE legs.py, so that is where the refactor landed;
  "write each skill ONCE" now holds:
  - `action_bbox(env, a)` + `bbox_size(bbox)`: the clone->step->frame_delta->bbox
    probe was written twice (the discovery scan and the `edit_left_col` cursor
    read). Now one primitive; `edit_left_col` is just `action_bbox(...)[1]`.
  - `action_period(env, a, key_fn)`: "repeat action A until observation key_fn
    returns to its start" was written twice (n_tiles via MOVE on cursor-column,
    cycle via EDIT on tile-glyph). Now one period-finder, two `key_fn`s.
  - `replay_for_reward(env, path)`: "commit the clone-found plan on the real env
    and check levels rose" was inlined in the solver; now a standalone leg.

## Candidate HIGHER-ORDER leg: discover -> search-on-clone -> replay-on-real
- The recurring composition across opaque-reward puzzles is:
      spec = discover(env)            # perceive structure from action deltas
      path = search(env, spec)        # nested-clone DFS, early-exit on reward
      replay_for_reward(env, path)    # execute the winning path on the real env
- Captured as `solve_by_clone_search(env, discover, search)`. `solve_tile_cycle_puzzle`
  is now expressed as this HOF with `discover=discover_tile_cycle_puzzle` and a
  `search` closure over `search_tile_cycle_config`. Any future puzzle that is
  (a) perceivable from action deltas and (b) has a small clone-reachable config
  space with opaque reward can reuse the same HOF by supplying its own
  discover/search pair — the players remain one-line compositions.
