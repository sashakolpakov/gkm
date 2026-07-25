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
