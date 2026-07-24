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

## ft09 level 4 — multi-state pattern-key board (generalised level 3)
- Same coordinate-click lattice as level 3, but each block is a THREE-STATE cell
  cycling 9->8->12->9 on each click (verified by probing one block on a clone;
  top-right legend lists the three states in that order). 21 uniform blocks, 3
  fixed 'pattern' keys at (1,1),(1,3),(3,2); pattern blocks ignore clicks.
- Mini-key format matches level 3: a solid centre colour (9 or 12 here) plus
  mark(0)/blank(2) cells. DECODE (found by a 27-way global-target search on
  clones, then reduced to a rule): each marked neighbour block's TARGET colour ==
  its key's centre colour; every OTHER uniform block takes one shared DEFAULT
  colour. Here marks->12/9 per key, default->8. Click cost per block = forward
  distance in the click cycle from its current colour to its target.
- The only free unknown is the single DEFAULT colour, resolved by trying each of
  the 3 cycle colours on a clone and keeping the one that raises levels_completed
  (3 clones, cheap). 21 clicks total; RESULT levels=4 moves=46 replay_ok=True.
- New legs: discover_multistate_key_clicks (probe cycle -> decode marks/centres
  -> per-block click counts -> resolve default on clones), solve_multistate_key_board
  = solve_click_board(env, discover_multistate_key_clicks). Reuses _grid_of_blocks,
  solve_click_board/_click_xy unchanged. play_level_4 is a one-liner. This is the
  k-state generalisation of discover_pattern_key_clicks (the 2-state case).

## DEBRIEF after level 4 — pattern-key decoders share a decode skeleton
- Put the four players side by side: they are ALREADY thin one-liners (each just
  picks a solver), so the duplication was not in players.py but INSIDE the two
  pattern-key decoders discover_pattern_key_clicks (level 3, 2-state) and
  discover_multistate_key_clicks (level 4, k-state). Both re-implemented the same
  low-level skills verbatim:
    * the grid-perceive + "has patterns AND uniform blocks" guard preamble,
    * cell3(blk) — sampling a decorated block's 3x3 mini-key (defined TWICE),
    * the mark-colour = least-common non-centre cell computation (TWICE),
    * the block-centre -> (x, y) click conversion (c + s//2 - 1, r + s//2 - 1)
      (open-coded in three places), and
    * reading a block's current centre colour.
- Extracted each as a shared, named leg, WRITTEN ONCE: pattern_key_grid (perceive
  + guard), read_mini_key (was cell3), mark_color, block_click_xy,
  block_center_color, and key_marked_neighbours (yield each mini-key's marked
  neighbour + that key's centre colour — the "read the marks" step level 4 needs).
  Both decoders now compose these; only their genuinely different DECODE RULE
  remains inline (level 3: a 2-state toggle predicate over ALL neighbours keyed on
  block colour; level 4: marked-neighbour -> centre colour + one shared default,
  driven by click-cycle distance). Behaviour unchanged (levels=4, moves=46,
  replay_ok=True): pure refactor, no new clicks.

## Candidate higher-order leg: decode_pattern_key(env, rule) — parameterised decoder
- The recurring composition across levels 3 and 4 is a pattern-key DECODE pipeline:
    perceive lattice (pattern_key_grid) -> read mini-keys (read_mini_key) ->
    find the mark colour (mark_color) -> apply a per-neighbour RULE to derive a
    per-block goal -> turn goals into block-centre clicks (block_click_xy) ->
    hand the click list to solve_click_board(env, planner).
  The ONLY per-level variation is the RULE that maps (mini-keys, mark, current
  block colours) to a per-block target/selection, plus how a target becomes a
  click count (1 click for the 2-state toggle, cycle-distance clicks for k-state).
- Candidate leg shape: decode_pattern_key(env, rule) where `rule(grid, mark,
  frame) -> {block_idx: click_count}` returns how many clicks each block needs;
  the leg builds the flat click list via block_click_xy and returns it, so every
  pattern-key planner collapses to supplying `rule`. Not promoted yet: the two
  rules are still structurally different (level 3 keys on block colour over ALL
  neighbours and toggles; level 4 keys only on MARKED neighbours over a colour
  cycle with a searched default), so forcing one signature now risks over-fitting.
  If a THIRD pattern-key variant appears and reuses this skeleton, promote
  decode_pattern_key from candidate to a load-bearing shared leg (the pattern-key
  analogue of solve_click_board for the coordinate-click substrate).

## ft09 level 5 — coupled binary pattern-key board
- Level 5 keeps the decorated-key lattice and binary colours, but a click may
  toggle several cells. `discover_coupled_key_clicks` therefore probes each
  control on a clone to recover its effect vector, reads inert mini-keys as
  SAME/OPPOSITE target constraints, and solves the resulting parity equations
  over GF(2). It tries the two possible token meanings on clones and returns the
  first prefix that actually raises `levels_completed`.
- Compared with levels 3 and 4, the board perception and coordinate-click
  substrate are unchanged; only the decode rule is new. The existing shared
  legs `read_mini_key`, `block_click_xy`, `block_center_color`, and
  `solve_click_board` are reused. The repeated validated-grid + frame-snapshot
  preamble from all three planners is now written once as
  `pattern_key_context`. All five players remain thin one-call compositions.

## Recurring composition pattern (candidate higher-order leg) — key context -> rule -> verified clicks
- Across levels 3, 4, and 5 the stable pipeline is now:
  `pattern_key_context(env)` -> variant-specific constraint rule -> click plan ->
  clone validation -> `solve_click_board` commit. The variation is entirely in
  the middle rule: direct two-state selection, cyclic colour targets, or a
  coupled GF(2) system.
- Candidate higher-order shape: `decode_pattern_key(env, rule, validator)`, where
  the shared leg supplies the recognized context and flattens/validates clicks,
  while `rule(context)` supplies only constraints or click counts. This remains
  a candidate rather than a forced abstraction because level 5 produces coupled
  control variables rather than independent per-cell click counts. The stable
  shared portion has nevertheless been extracted as `pattern_key_context`.

## ft09 level 6 — coupled controls reused across a new layout
- Compared `play_level_6` with all earlier players. Like levels 1–5 it is already
  a thin one-call composition; more specifically it is identical in shape to
  level 5: both call `solve_coupled_key_board(env)`. Level 6 changes the layout
  and decorates controls with direction marks, but introduces no new skill. The
  existing coupled-board leg already discovers click effects from clones rather
  than assuming positions or appearances, reads inert adjacent clue symbols,
  solves the GF(2) constraints, and commits the verified prefix.
- The coupled-board skill therefore remains written ONCE in `legs.py` and is
  reused verbatim by both players. No level-specific mechanics were moved into
  `players.py`; each player only selects its shared solver leg.
- One literal repetition remained among earlier planners: clone an environment,
  replay a proposed action list, and test whether `levels_completed` advanced.
  Extracted that behavior ONCE as `plan_reaches_next_level(env, plan, apply_fn)`
  and reused it in the toggle-subset and multistate-default planners. The level-6
  coupled planner deliberately retains its distinct stepwise loop because it
  must return the first goal-reaching prefix, not merely validate a full plan.

## Candidate higher-order leg — propose -> clone-validate -> commit
- The recurring composition is now explicit at two nested levels: a planner
  proposes candidate action lists and validates them with
  `plan_reaches_next_level`; once it returns the winning list,
  `plan_and_commit` replays it on the real environment. In compact form:
  `candidate generator -> clone validation -> winning plan -> real commit`.
- A future higher-order leg could own the candidate iteration as well, taking a
  candidate generator plus `apply_fn`, returning the first level-reaching plan.
  It remains a candidate because coupled boards require prefix-aware validation,
  while toggle and multistate planners validate whole candidates. The shared,
  behavior-identical portion is now centralized without forcing those different
  stopping semantics into one abstraction.
