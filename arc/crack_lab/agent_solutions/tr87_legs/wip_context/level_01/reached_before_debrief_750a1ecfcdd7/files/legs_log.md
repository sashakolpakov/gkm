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
