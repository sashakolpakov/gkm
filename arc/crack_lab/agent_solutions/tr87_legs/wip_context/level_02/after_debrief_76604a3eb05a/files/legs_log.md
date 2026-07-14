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

## tr87 level 2 (legend-decode glyph cipher)
- Same tile-cycle *mechanic* as L1 (discover_tile_cycle_puzzle finds edit=1,
  move=4, n_tiles=7, cycle=7), but a NEW win condition: 7 editable glyph tiles
  (5x5 fg=5 on VAL=11 border) must be set to a hidden target sequence. Reward
  is still opaque (EDIT changes only the tile; no feedback), and 7^7 is far too
  big to brute force -- so the target MUST be read from the frame.
- The frame is a Rosetta legend: 6 small 'key' boxes (border=7, one glyph each)
  each paired with a 'value' region (border=11, 1..k glyphs) on the same rows to
  its right (a `333` arrow sits between). A wide 'target' box (border=7, 4
  glyphs) is the coded word. Decoding: match each target glyph to a key, then
  concatenate that key's value glyphs -> exactly 7 glyphs = the tile targets.
- KEY GOTCHA: the same symbol is drawn in DIFFERENT poses across the legend,
  target box, and per-tile cycle sets (each tile cycles its own 7-glyph set,
  17 distinct total). Exact pixel match fails; matching by a D4-canonical form
  (min over 8 rotations/reflections) makes every lookup unique and correct.
  Decoded plan for L2: per-tile edit-counts [4,5,3,2,3,3,4].
- New legs (general for this cipher class): `_glyph`, `_glyph_canon` (D4 canon),
  `_bordered_boxes`/`_box_glyphs` (split a framed glyph-row on a fixed pitch),
  `discover_glyph_cipher_puzzle` (auto-detects VAL/BOX/FG colors, editable
  region, key/value legend pairs, target box; decodes to a canonical target
  sequence), `plan_glyph_cipher` (per-tile edit-count via clone), and
  `solve_glyph_cipher_puzzle` = the same discover->plan->replay HOF
  (solve_by_clone_search) used for L1. Player stays a one-line composition.

## Debrief refactor (post-L2): the L1 and L2 solvers shared a hidden skill
- Comparing `play_level_2` (glyph cipher) with `play_level_1` (clone search): both
  players are already one-line compositions, so the duplication again lived INSIDE
  legs.py. The repeated skill was PATH SERIALIZATION: turning a per-tile plan
  (a list of EDIT-press counts) into the concrete `edit`/`move` action stream by
  interleaving each tile's edit-run with a single MOVE (no trailing move).
    - L1's `search_tile_cycle_config` built this inline during its DFS
      (`path + [edit]*s + [move]`).
    - L2's `plan_glyph_cipher` built the same thing in a tail loop
      (`path += [edit]*s; if ti < n-1: path += [move]`).
- Extracted ONCE as `tile_cycle_path(edit, move, counts)`. `plan_glyph_cipher`
  now returns `tile_cycle_path(...)` directly; `search_tile_cycle_config` now
  carries the winning *per-tile edit-count list* through its recursion and
  serializes it once at the end via the same leg. Behaviour is unchanged
  (gkm_try still RESULT levels=2 moves=58 replay_ok=True; L1-from-scratch still
  28 moves replay_ok=True).

## Candidate HIGHER-ORDER leg: solve_tile_row(env, plan_fn)
- With `tile_cycle_path` factored out, L1 and L2 collapse to the SAME shape and
  differ in only one place -- how the per-tile edit-count plan is derived:
      spec   = discover_tile_cycle_puzzle(env)   # find edit/move, n_tiles, cycle
      counts = plan_fn(env, spec)                # <-- the ONLY level-specific step
      path   = tile_cycle_path(edit, move, counts)
      replay_for_reward(env, path)
  where `plan_fn` is:
    * L1: a global clone DFS over configs (opaque reward, counts found by search);
    * L2: a per-tile frame decode (legend cipher gives each tile's target glyph,
      counts found by cycling a clone until the glyph matches).
  This suggests a higher-order leg `solve_tile_row(env, plan_fn)` = discover the
  tile-row structure -> `plan_fn` -> `tile_cycle_path` -> `replay_for_reward`.
  It is a SPECIALISATION of `solve_by_clone_search` for the tile-row family: the
  "search" half is fixed to "serialize a per-tile count plan", so any new
  tile-row level would only supply a `plan_fn`. Not yet extracted (two data
  points), but this is the pattern to watch for the next tile-row variant.
