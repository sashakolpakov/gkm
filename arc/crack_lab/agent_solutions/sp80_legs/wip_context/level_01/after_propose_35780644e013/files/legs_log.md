# Legs Log for sp80

## Level 1 discovery (solved)

**Frame layout:**
- Row 0: color-14 top border (timer depletes right-to-left)
- Rows 1-7, cols 36-39: color-4 (3 rows) + color-6 (4 rows) target/indicator object (static)
- Rows 16-19, cols 12-31: color-9 moveable block (4 rows × 20 cols)
- Rows 52-59: two color-11 U-shaped goal structures
  - Left: posts at cols 16-19 and 24-27, bottom bar rows 56-59
  - Right: posts at cols 40-43 and 48-51, bottom bar rows 56-59
- Rows 60-63: color-1 bottom border

**Actions discovered:**
- Action 1 = UP: moves 9-block up 4 pixels (stops at rows 12-15)
- Action 2 = DOWN: moves 9-block down 4 pixels (stops at rows 44-47)
- Action 3 = LEFT: moves 9-block left 4 pixels (stops at cols 0-19)
- Action 4 = RIGHT: moves 9-block right 4 pixels (stops at cols 44-63)
- Action 5 = USE: conditionally completes level (no visible change until win)
- Action 6 = no effect (timer only)

**Win condition:** After 3 RIGHT moves (block at rows 16-19, cols 24-43), USE (action 5) completes level 1.

**Winning path:** [4, 4, 4, 5]

## Legs added to legs.py

- `play_fixed_sequence(env, sequence)`: plays a fixed list of actions
- `bfs_win(env, ...)`: BFS over full-frame hash to find winning path
- `bfs_win_compact(env, key_fn, ...)`: BFS with custom key function
