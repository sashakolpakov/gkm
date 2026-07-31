# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## bp35 L1 — first solve (17 moves, replay_ok)

Substrate cracked from raw frames. The earlier frontier's reading ("clear the
yellow components, search the phase layouts") mis-framed the game: it is not a
tile-clearing puzzle, it is a **vertical climb** and the "phase changes" were
the world scrolling.

Discovered mechanics:

* **Band lattice.** The 64x64 frame is a 6px lattice; sampling
  `(3 + 6i, 15 + 6j)` yields a 10 x 7 symbolic grid. Everything else is
  texture. Row 63 is a move counter (one pixel per action).
* **Fixed camera, scrolling world.** The avatar always renders on band row 6.
  What looked like a layout change is the world scrolling down by the number
  of bands the avatar just gained.
* **Anti-gravity.** After every action the avatar rises through open cells
  until blocked. Height is therefore bought by *opening the cell overhead*,
  and a single click can pay for several bands.
* **Click = coordinate action.** `env.step(6, x=col_px, y=row_px)` removes a
  colour-14 block anywhere on screen. Any other click is a no-op that still
  burns a move.
* **Rising floor.** A kill floor rises one band every two moves and starts
  ~8 bands below the avatar. Death on contact. This is the real budget:
  the climb must average better than one band per two moves.
* **Win.** A prize cell (a colour outside the terrain palette, 7 here) is
  revealed on the avatar's row at the top of the tower; stepping onto it
  completes the level.

New legs, all written against the lattice rather than against L1:

| leg | role |
| --- | --- |
| `band_grid` / `cell_symbol` | raw frame -> symbolic 10x7 terrain |
| `avatar_column`, `moves_used`, `find_symbol` | cheap observations |
| `click_cell`, `run_plan` | action primitives |
| `climb_plans`, `walk_plan`, `_column_plan` | plan enumeration |
| `climb_tower`, `reach_prize`, `climb_to_prize` | composites |

Dense progress measure that made this tractable: **bands gained per move**.
`climb_tower` greedily takes the best-ratio column plan each iteration, which
is what keeps the ascent ahead of the rising floor. No level-1 constants leak
into the legs; `play_level_1` is a single `climb_to_prize(env)` call.

Reuse note for later levels: expect the same lattice and the same
climb/click/prize verbs. If a later level differs, the likely deltas are the
terrain palette (`WALL_COLORS` / `BLOCK_COLORS`) and the plan enumerator
needing lookahead instead of greedy ratio — not the observation legs.

## bp35 L2 — second solve (60 moves cumulative, replay_ok)

The L1 reuse note called it: the observation legs survived untouched, and the
one thing that broke was exactly the predicted thing — **greedy ratio is not
enough once a column can be a trap**.

What L2 actually changed, all confirmed on clones:

* **Same lattice, same verbs.** `band_grid` / `avatar_column` / `moves_used` /
  `click_cell` transferred with zero edits. The move counter resets per level.
* **Off-palette colour is not a prize any more.** L1's win object was colour 7;
  L2's colour-15 objects are *hazards*. They are **solid sideways** (stepping
  into one is a silent no-op) and **fatal from below**: each renders 4px tall
  inside its band with colour-9/11 "legs" hanging into the band beneath, so
  coming to rest directly under one kills. `cell_symbol` still calls them
  `PRIZE`; nothing in the library now decides prize-vs-hazard, because a clone
  probe answers it for free. That is the real lesson — *don't extend the
  symbolic alphabet, let the successor generator drop dead children*.
* **Win is height, not contact.** No goal cell is ever touched. The level fires
  after ~27 bands of ascent.
* **The rising floor is the whole budget.** Measured directly by burning idle
  LEFT/RIGHT moves from the L2 start: the avatar is engulfed on move 14, i.e.
  the floor starts 7 bands below and rises one band per two moves. Being
  engulfed does **not** set `terminal()` — the avatar simply vanishes from the
  frame, so `avatar_column(...) is None` is the liveness test, not
  `env.terminal()`.
* **Plans go stale mid-plan.** A click can lift the avatar 7 bands at once, and
  the world scrolls by that much, so L1's precompute-then-run column plans
  addressed the wrong cells. Every new action leg re-reads the frame per step.

Where greedy died: at 21 bands the avatar sits in a one-wide shaft whose only
overhead block opens into a chamber capped by hazards — a guaranteed kill. The
cheap-but-fatal option (`cost 1`) beat the correct 11-move detour to the
neighbouring shaft on height-per-move, every time.

New legs:

| leg | role |
| --- | --- |
| `band_shift` | height gained between two frames, by raw-row offset match |
| `act`, `run_actions`, `click_action` | record/replay action tuples incl. `(6, x, y)` |
| `walk_to` | scroll-safe sideways walk that clears blocks as it goes |
| `climb_macros` | successors = walk-to-column (± clear overhead), dead ones dropped |
| `climb_search` | best-first ascent on accumulated height, move count as tie-break |
| `climb_by_search` | plan on clones, commit on the real env |

Dense progress measure, unchanged in spirit but now the *search key* rather
than a greedy score: **bands gained** (`band_shift`), with move count only
breaking ties, since the floor makes moves cheap-but-not-free. 800 expansions
finds L2 in ~13s. `play_level_2` is one `climb_by_search(env)` call.

Also fixed while here: `run_plan` / `climb_tower` / `reach_prize` tested
`env.levels_completed` for truthiness, so every one of them no-oped from L2
onward (`climb_to_prize` cheerfully returned True having taken zero actions).
They now compare against the level in progress on entry. Any future leg must
be level-relative, never truthy-on-levels_completed.

Reuse note for L3: try `climb_by_search` first — it subsumes `climb_tower` and
is hazard-agnostic. The likely deltas are a successor set that needs more than
"walk to a column, clear overhead" (e.g. a down/drop verb, or an object to
push), and a heuristic that needs more than height if the goal stops being the
ceiling.

## bp35 L4 — composition debrief (113 moves cumulative, replay_ok)

Compared with the earlier players, L4 introduces a new search policy but not a
new execution pattern. L2, L3, and L4 all had the same three-part composition:

1. search safely on clones for an action route;
2. replay the route on the real environment;
3. report success relative to the level active before replay.

That recurring pattern is the candidate higher-order leg **plan a route, then
commit it**. It now lives once as `plan_and_commit(env, search_leg, **options)`;
the players remain thin by supplying only their policy (`climb_search`,
`local_hazard_climb_search`, or `gravity_room_search`). The old named composite
legs delegate to the same helper for compatibility, so their behavior is
unchanged.

## bp35 L7 — composition debrief (256 moves cumulative, replay_ok)

The players remain thin compositions: L1 invokes a direct climb composite;
L2--L5 select search-and-commit policies; and L6--L7 invoke scripted
support-room composites. L7 differs from L6 only in one execution concern:
some route entries are symbolic gravity flips whose click coordinates must be
resolved from the current frame after scrolling.

The recurring candidate higher-order leg is **interpret and execute an action
route**: guard each step, optionally resolve a contextual token, apply the
resulting action, and optionally recognize success relative to the level
active on entry. This now lives once in `run_action_route`; ordinary
`run_actions` delegates to it, while `cross_staged_gravity_zigzag` supplies
only its gravity-token resolver and level-relative stop policy. The staged
supports and zigzag route remain level-specific data, so no route behavior was
generalized away.
