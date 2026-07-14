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

## tr87 level 3 (legend cipher with multi-glyph keys -> word segmentation)
- Same tile-cycle MECHANIC as L1/L2 (discover_tile_cycle_puzzle finds edit=1,
  move=4, n_tiles=7, cycle=7; editable A-box at (51,7,57,55), 5x5 fg=5 glyphs).
  Same "decode a legend to get the hidden tile targets" GOAL as L2. Reward is
  opaque, 7^7 too big -> target MUST be read from the frame.
- NEW structure vs L2: keys AND values are BOTH bordered boxes (two distinct
  border colors: KEY=11, VAL=10), laid out as key->value PAIRS (a `333` arrow
  between), several pairs per row. Crucially a single KEY box may span SEVERAL
  glyphs (key glyph-lengths here were 1,2,1,2,3,1). So the coded 'word' (the one
  unpaired KEY-colored box, 8 glyphs) cannot be read glyph-by-glyph like L2 --
  it must be SEGMENTED (tokenised) into keys first, then each key replaced by
  its value glyph-sequence -> exactly n_tiles(=7) target glyphs.
- Discovery is fully automatic: VAL/FG from the ring around the EDIT region;
  the OTHER glyph-tall bordered-box color = KEY; the VAL box containing the edit
  region = the editable tiles; each KEY box paired with the nearest VAL box on
  its row to the right; the ONE unpaired KEY box = the coded word. D4-canonical
  glyph forms make legend/word/tile lookups pose-invariant (reused _glyph_canon).
  Decoded plan: word tokenises uniquely as keys [K0,K4,K2,K5,K1] -> 7 tile
  targets; per-tile edit-counts [6,2,4,3,1,4,5] (via plan_glyph_cipher on clone).
- New legs (general for this cipher class): `_box_glyphs_canon` (split a box
  into D4-canonical glyphs), `_segment_word` (unique DFS tokenisation of a glyph
  word into keys), `discover_glyph_legend_puzzle` (auto-detects KEY/VAL/FG,
  editable box, key->value legend pairs, the unpaired coded word; segments and
  decodes to a canonical target sequence -> spec for plan_glyph_cipher),
  `solve_glyph_legend_puzzle` = the SAME discover->plan->replay HOF
  (solve_by_clone_search) used for L1/L2. Player stays a one-line composition.
- REUSE: the entire back half (plan_glyph_cipher -> tile_cycle_path ->
  replay_for_reward) and the front half (discover_tile_cycle_puzzle, _glyph,
  _glyph_canon, _box_glyphs) were reused unchanged; only the legend-parsing +
  segmentation front-end was new. gkm_try: RESULT levels=3 moves=97 replay_ok=True.

## Debrief refactor (post-L3): L2 and L3 discovery shared a whole front-end
- Comparing `play_level_3` (legend cipher with segmentation) with the earlier
  players: all three players are already thin one-line compositions, so — as
  after L1 and L2 — the duplication lived INSIDE legs.py. This time the repeat
  was not a tiny primitive but an entire PERCEPTION FRONT-END shared verbatim by
  `discover_glyph_cipher_puzzle` (L2) and `discover_glyph_legend_puzzle` (L3):
    * run `discover_tile_cycle_puzzle` and unpack edit/move/n_tiles/cycle;
    * read the edit-region bbox and derive gh, gw, pitch;
    * infer VAL (border color = the ring around the edit region) and FG (the
      non-VAL color inside it) — an identical ~10-line ring scan in both;
    * collect all glyph-tall bordered boxes (`tall`) and find the VAL box that
      contains the edit region (the editable tiles).
  Each copy also carried its own local `contains(b, bb)` closure.
- Extracted ONCE as `discover_glyph_tiles(env)` (returns a base dict:
  edit/move/n_tiles/cycle, f, ebb, gh/gw/pitch, VAL, FG, tall, editbox) plus a
  generic `_box_contains(outer, inner)` bbox primitive. Both level-specific
  discoverers now start with `base = discover_glyph_tiles(env)` and only add
  their own legend-parsing tail (L2: single-glyph keys + a wide target word,
  matched glyph-by-glyph; L3: multi-glyph key boxes paired with value boxes +
  one unpaired word, matched by `_segment_word` tokenisation). Behaviour is
  unchanged: gkm_try RESULT levels=3 moves=97 replay_ok=True, and a full
  from-scratch run of all three players is likewise levels=3 moves=97.

## Candidate HIGHER-ORDER leg: solve_glyph_cipher(env, decode_fn)
- With `discover_glyph_tiles` factored out, L2 and L3 now collapse to the SAME
  shape and differ in only ONE place — how the legend is parsed and the coded
  word is turned into the canonical target glyph sequence:
      base = discover_glyph_tiles(env)          # shared perception front-end
      seq  = decode_fn(base)                     # <-- the ONLY level-specific step
      spec = {**mechanic, editreg, seq_canon=seq}
      solve_by_clone_search(discover=lambda: spec, search=plan_glyph_cipher)
  where `decode_fn` is:
    * L2: pair single-glyph key boxes with value regions, match each target-word
      glyph to a key (D4-canonical), concatenate the paired value glyphs;
    * L3: pair multi-glyph key boxes with value boxes, uniquely SEGMENT the
      coded word into keys (`_segment_word`), concatenate the paired values.
  This suggests a higher-order leg `solve_glyph_cipher(env, decode_fn)` =
  discover_glyph_tiles -> `decode_fn` -> plan_glyph_cipher -> tile_cycle_path ->
  replay_for_reward. It is a SPECIALISATION of `solve_by_clone_search` for the
  glyph-cipher family (the "search" half is fixed to plan_glyph_cipher over a
  decoded target sequence), just as `solve_tile_row` was proposed for the raw
  tile-row family. Not yet extracted (the two decoders still differ enough to
  earn their own named legs), but this is the pattern to collapse on the next
  glyph-cipher variant.
