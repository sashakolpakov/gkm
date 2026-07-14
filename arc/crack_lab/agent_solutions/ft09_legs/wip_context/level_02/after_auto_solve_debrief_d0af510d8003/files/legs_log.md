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
