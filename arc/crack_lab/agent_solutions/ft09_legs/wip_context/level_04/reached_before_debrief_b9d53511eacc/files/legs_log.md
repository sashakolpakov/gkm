# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## ft09 level 1 — toggle-tile board
- Only action is coordinate click env.step(6,x,y); x=col, y=row.
- Interactive region = a framed 3x3 board (right side). Each surrounding tile
  toggles color 9<->8 on click; center tile is a fixed icon (non-clickable).
- A hidden target subset of tiles must be set; completing it raises the level.
  Matching a visible reference grid was NOT the rule — found by subset search.
- Legs added: discover_toggle_tiles (cluster clicks by changed-region bbox,
  ignoring bottom status bar), search_toggle_solution (subset search fewest
  clicks first on clones), solve_toggle_board (commit found clicks). 4 moves.

## Candidate higher-order leg: plan_and_commit (plan on clones -> replay real)
- Recurring composition seen every level so far: a *planner* searches on
  env.clone()s for an action sequence that raises levels_completed (or hits some
  goal predicate) — e.g. search_toggle_solution, and perception.bounded_bfs —
  then the caller *commits* that sequence action-by-action on the real env.
- Extracted the commit loop ONCE as commit_plan(env, plan, apply_fn) and the
  full pattern as plan_and_commit(env, planner, apply_fn). The only per-game
  parts are the planner and apply_fn (how one planned action is executed:
  _click_xy for coordinate boards, env.step for directional actions).
- solve_toggle_board is now thin: plan_and_commit(env, search_toggle_solution,
  _click_xy). Future players/legs reuse plan_and_commit rather than re-writing
  the search->replay commit loop. Behaviour unchanged (still 4 moves).

## ft09 level 2 — another toggle-tile board (same skill, reused verbatim)
- Discovered by running the arena: after replaying the level-1 clicks, level 2's
  interactive region is a WIDER 3-row toggle grid (~13 independent tiles) rather
  than a framed 3x3. Mechanic is identical: each tile toggles on a coordinate
  click env.step(6,x,y); a hidden target subset completes the level.
- discover_toggle_tiles found 13 tiles; search_toggle_solution returned a 7-click
  subset that raises levels_completed. No new leg was written — play_level_2 is a
  thin one-liner: solve_toggle_board(env), identical to play_level_1.
- Confirms each skill is written ONCE: the discover -> subset-search -> commit
  chain lives only in legs.py; both players just compose it.

## Recurring composition pattern (candidate higher-order leg) — reconfirmed
- The pattern now recurs across TWO levels: a per-level player collapses to a
  single leg call because the underlying skill generalises. The reusable core is
  solve_toggle_board = plan_and_commit(env, search_toggle_solution, _click_xy):
    discover interactive tiles (probe clones) -> subset-search a goal-reaching
    click set (on clones) -> commit the clicks (on the real env).
- The generic higher-order leg is plan_and_commit(env, planner, apply_fn):
  "plan on clones -> replay on the real env". Only the planner and the one-action
  applier vary per game; the search->replay commit loop is written ONCE. Two
  toggle levels reused it unchanged, so it is promoted from "candidate" to a
  load-bearing shared leg for coordinate-click boards. Behaviour unchanged
  (levels=2, moves=11: 4 for level 1 + 7 for level 2).

## ft09 level 3 — pattern-key toggle board (new leg, decode not search)
- Same coordinate-click substrate, but 23 independent toggle blocks (8<->12) on a
  6x6-cell lattice => 2^23 subsets, far too many for search_toggle_solution
  (which caps at 16 tiles). No dense feedback exists: the bottom bar is only a
  move counter (fill = m - floor((m+1)/3), +0 every 3rd move, block-independent),
  and no hidden indicator responds to combinations. All symmetric-subset searches
  (D2 / top-bottom / left-right) fail => the target is asymmetric and must be
  DECODED, not searched.
- The board carries 4 non-clickable 'pattern' blocks, each a 3x3 mini-key: a solid
  centre colour (block-colour 8 or selected-colour 12) plus mark(2)/blank(0) cells.
  RULE (found by probing readings): a grid-neighbour block is toggled iff
  (centre == block_colour) == (cell == mark_colour). This decodes to a unique
  14-block set that raises the level; verified purely from pixels.
- New legs: _grid_of_blocks (perceive an equal-square-cell lattice: bg, block
  size via run-length mode, per-slot uniform vs pattern), discover_pattern_key_clicks
  (apply the decode rule, return block-centre clicks), solve_pattern_key_board =
  plan_and_commit(env, discover_pattern_key_clicks, _click_xy). play_level_3 is a
  one-liner composing that leg. Reuses plan_and_commit/_click_xy unchanged.

## Candidate higher-order leg: solve_click_board (planner-parameterised click board)
- DEBRIEF comparison after level 3: put the three players side by side and the
  duplication is not in players.py (all three are one-liners) but INSIDE the
  named solvers. solve_toggle_board and solve_pattern_key_board were BOTH literally
  `plan_and_commit(env, <planner>, _click_xy)` — the coordinate-click applier
  `_click_xy` was bound in TWO places, and every ft09 board reduces to the same
  shape: "produce a click list, then click it".
- Extracted that shape ONCE as solve_click_board(env, planner) = plan_and_commit(
  env, planner, _click_xy). The only per-board part is the planner: a subset
  SEARCH on clones (search_toggle_solution, levels 1-2) or a pixel DECODE with no
  search (discover_pattern_key_clicks, level 3). Two planner *flavours*, one
  commit substrate. solve_toggle_board / solve_pattern_key_board are now thin
  aliases that just pick the planner; _click_xy is written once.
- The recurring pattern promoted from candidate to load-bearing:
    coordinate-click board solver = solve_click_board(env, planner)
                                   = plan_and_commit(env, planner, _click_xy)
  i.e. plan_and_commit specialised to the coordinate-click applier. Future
  ft09-family levels should supply a planner (search or decode) and reuse
  solve_click_board rather than re-binding _click_xy. Behaviour unchanged
  (levels=3, moves=25, replay_ok=True): pure refactor, no new clicks.

## ft09 level 4 — MULTI-STATE pattern-key board (new leg, 3-state generalisation)
- Same coordinate-click substrate and lattice-of-blocks + non-clickable pattern
  keys as level 3, BUT each clickable block CYCLES THREE colours per click
  (discovered on a clone: 9 -> 8 -> 12 -> 9; the top-right 9/8/C legend depicts
  exactly these three states). So a block's target is a 3-valued state reached by
  0/1/2 clicks, not a binary toggle. All blocks start at 9.
- Grid is irregular (5x5 with the bottom two rows indented to the middle 3
  columns); _grid_of_blocks handled it verbatim: 18 clickable + 3 pattern blocks.
- Decode RULE (found by enumerate-and-verify, not hand-derived): each pattern key
  is a 3x3 mini-picture with centre colour C in {9,12} and surround cells in
  {mark=2, blank=0}; a grid-neighbour's TARGET state is  8 if its cell is the
  mark(2), else the pattern's centre colour C (9 or 12). Neither a pure
  cell->state map (center ignored) nor an additive clicks-mod-3 scheme completes;
  the winning consistent (cell,centre)->state table is
  {(0,9):9, (2,9):8, (0,12):12, (2,12):8}.
- Rather than hardcode that table, the new planner discover_pattern_state_clicks
  DISCOVERS the per-click colour cycle on a clone (index = clicks-to-reach), reads
  each neighbour's (cell,centre) votes, then ENUMERATES consistent
  (cell,centre)->state mappings and VERIFIES each on a clone, returning the first
  click sequence (block centres, repeated per cycle step) that raises the level.
  This subsumes the binary case as a 2-length cycle.
- New legs: _cell3 (read a block as a 3x3 of ninth-centre colours; factored out of
  discover_pattern_key_clicks' inner helper), _block_state_cycle (probe one block's
  per-click colour cycle on a clone), discover_pattern_state_clicks (the
  enumerate+verify multi-state decoder), solve_pattern_state_board =
  solve_click_board(env, discover_pattern_state_clicks). play_level_4 is a
  one-liner composing that leg. Reuses solve_click_board/_click_xy/_grid_of_blocks
  unchanged. Result: levels=4, moves=46 (25 + 21), replay_ok=True.
