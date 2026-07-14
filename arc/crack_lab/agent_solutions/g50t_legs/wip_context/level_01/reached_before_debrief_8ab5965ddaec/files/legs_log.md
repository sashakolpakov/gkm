# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## g50t level 1
- Avatar = the object that TRANSLATES under a move (detected via colour+shape
  match). Movement actions 1/2/3/4; action 5 = USE.
- USE on a "switch" tile reconfigures the maze (opens gates -> avatar reachable
  set grows); elsewhere USE is a reset. Goal = drive avatar into goal region;
  reward is sparse (levels_completed).
- HIDDEN STATE: byte-identical frames can transition differently, so frame-key
  BFS is invalid. Plan as concrete action sequences on clones, dedup only on
  avatar position within a fixed gate config.
- Legs added: avatar_tl, _move_explore, plan_unlock_reach / solve_unlock_reach
  (general "unlock-then-reach"). play_level_1 = solve_unlock_reach.
