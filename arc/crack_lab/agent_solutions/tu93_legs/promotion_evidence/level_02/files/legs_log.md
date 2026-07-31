# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## tu93 level 1 — fixed-block grid maze
- Frame: 64x64. bg=5. Avatar = a 3x3 block (colour 9) with a single-pixel
  marker (colour 4) inside it. Goal = a distinct 3x3 block (colour 14).
- Actions 1/2/3/4 = UP/DOWN/LEFT/RIGHT, moving the avatar one NODE (6 px = two
  3-px blocks) when the intervening edge is open. Blocked moves still burn a
  turn (a bottom-row step-counter bar of colour 6 decrements each move).
- Maze structure: 3x3 blocks; NODES at even block indices, EDGES at odd. An
  edge is passable iff its block is non-bg (colour 2 here); bg (5) = wall.
- New legs (general, source-free):
  - parse_block_maze: origin via non-bg bbox, ignoring HUD bars (a colour
    confined to a single row/col spanning >=50% of the frame — robust to the
    depleting counter). Builds node graph, finds start (block holding the
    rarest/marker colour) and goal (node colour != open-path and != start).
  - maze_path_actions: BFS in node space -> key-action list.
  - drive_block_maze: re-plans from the LIVE frame each step and commits one
    action; robust to avatar/marker dynamics. Stops on level gain/terminal.
- play_level_1 = drive_block_maze(env). Reuse for any later block-maze config.

## Debrief (post-clear) — refactor + candidate higher-order leg
- Compared play_level_1 against the players in players.py: it is the only
  player so far, and it is already thin (one leg call). No cross-player body
  duplication to fold. The real duplication risk was *inside* legs.py:
  drive_block_maze had welded two separable skills — the maze-specific PLANNER
  and a generic closed-loop DRIVER loop.
- Refactor: extracted the driver loop into its own named leg `drive_replan`
  (written ONCE). drive_block_maze is now thin composition:
  `drive_replan(env, lambda f: maze_path_actions(f, cell))`. Behaviour is
  identical (same one-action-per-step, same stop conditions).
- Candidate HIGHER-ORDER leg (recurring composition pattern):
  **closed-loop re-planning**  =  sense -> plan(frame) -> commit ONE action
  -> re-plan from fresh frame -> repeat, halting on level-gain/terminal/no-path.
  Signature: `drive_replan(env, plan_fn, max_steps)`. It is a general adapter
  turning ANY per-frame planner `plan_fn(frame) -> [actions]` into a robust
  env-executing player. Reuse for future levels: write only a level-specific
  `plan_fn`, then compose with `drive_replan`. This is the composition shape to
  expect for grid/pathing puzzles where the avatar may drift or planning must
  react to live state.
