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
- `make_bbox_key(color)`: factory that returns a compact BFS key function
  tracking the bounding box (row_min, row_max, col_min, col_max) of all
  pixels with the given color value; returns None when the color is absent.
- `bfs_or_fallback(env, key_fn, fallback, ...)`: tries `bfs_win_compact`;
  plays the discovered path, or *fallback* if BFS finds nothing.

## Level 2 discovery (solved)

**Frame layout (after level 1):**
- 3 color-11 sockets (∏, top-bar + hanging posts) at top, unreachable.
- 3 movable blocks: A (12 wide, cols8-19), B (12 wide, cols28-39),
  C (20 wide, cols20-39). One block is coloured 9 (selected), rest 8.
- color 4/6 bottom-right = static level decoration.

**Key mechanic — action 6 is a COORDINATE action:** `env.step(6, x, y)`
selects/grabs the block whose cell is at (x=col, y=row); it turns 9 and the
directional actions 1..4 then move THAT block. So any block can be driven.
- Blocks move 4px/step, pass THROUGH each other, bounded to rows 16..51.
- Sockets are decoration: a hard wall keeps blocks at row>=16 (never reach them).
- Action 5 (USE) is the commit and cycles selection; without a final USE no
  configuration wins (verified by exhaustive column sweep).

**Win condition (found by exhaustive clone sweep + USE):** put a 12-wide block
at cols 20-31 and the 20-wide block C at cols 28-47, then press USE. Row is
irrelevant (as in level 1, alignment is column-only). Winning drive used:
A->cols20-31 (right x3), B->cols36-47 (right x2), C->cols28-47 (right x2), USE.

## New leg added

- `click_select(env, x, y)`: coordinate interaction — `env.step(6, x, y)` to
  grab the object under a pixel so 1..4 move it. Recorded as [6, x, y].

`play_level_2` composes `click_select` + `play_fixed_sequence` (moves + USE).

## DEBRIEF after clearing level 2 — refactor

Comparing `play_level_2` to the earlier `play_level_1`:

- `play_level_1` is already thin: a single `bfs_or_fallback(...)` call.
- `play_level_2` inlined the same two-step idiom **three times**:

  ```
  click_select(env, x, y)          # grab a block
  play_fixed_sequence(env, moves)  # drive it with 1..4
  ```

That "select-then-drive" idiom is now written ONCE as the leg
`grab_and_move(env, x, y, moves)` (= `click_select` + `play_fixed_sequence`).
`play_level_2` is now thin composition: three `grab_and_move` lines plus a
final `play_fixed_sequence(env, [5])` (USE / commit). No behaviour change —
`grab_and_move` emits exactly the same action stream as the old inline code.

## New leg added (refactor)

- `grab_and_move(env, x, y, moves)`: select-then-drive. Grabs the object under
  pixel (x, y) via the coordinate action (6), then plays a fixed list of
  directional moves on it. Encapsulates the idiom repeated 3× in `play_level_2`.

## Recurring composition pattern (candidate higher-order leg)

### Multi-object drive (new observation, level 2)

For levels with several independently-movable objects, the player is a *list
of (grab-point, move-plan) pairs* followed by a commit:

```
plan = [((x1, y1), moves1),
        ((x2, y2), moves2),
        ...]
for (x, y), moves in plan:
    grab_and_move(env, x, y, moves)
play_fixed_sequence(env, [USE])
```

Candidate higher-order leg: `drive_objects(env, plan, commit=[5])` — take a
list of `((x, y), moves)` tuples, `grab_and_move` each in turn, then play the
commit sequence. A player collapses to a single declarative `plan` (WHERE to
grab + HOW to move each object) with all iteration/commit logic in the leg.

### Single-object search (level 1)

Every single-object level player follows the same three-step skeleton:

```
key_fn  = make_bbox_key(<moveable-object color>)
path    = bfs_win_compact(env, key_fn, ...)          # search
play    = path if path else <hardcoded fallback>
play_fixed_sequence(env, play)
```

`bfs_or_fallback(env, key_fn, fallback, …)` encapsulates this pattern.
A player becomes a single line: choose *what* to track (the color) and
*what* to do if search fails (the fallback); all search-and-play logic
lives once in the leg.  Future players should default to this leg and only
reach for lower-level primitives when the state key or win condition is
more complex than a single-color bounding box.
