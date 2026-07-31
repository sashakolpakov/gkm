# Verified frontier brief: lf52 level 9

All facts below were reproduced through `frame()`, `clone()`, and `step()`.

## Accounting

- Validated level-8 checkpoint boundary: 544 actions.
- Default harness cap: 600 actions, leaving 56 at level 9.
- Current composed level-9 leg wins on a pristine level-entry clone in 102
  actions, so default checkpoint replay stops at 600 with level 9 unfinished.

## Mechanics

- Action 4 moves the bordered carrier right at entry; actions 1, 2, and 3
  are initially blocked.
- Action 6 uses source/destination coordinate clicks.
- Action 7 undoes one low-level step.
- Color-14 pieces are pegs.
- Color-9 pieces are movable bridges. A peg jumps over a bridge without
  consuming it. A bridge can jump over either a peg or another bridge.
- Color-15 arrow bridges are persistent and not coordinate-movable.
- The real dense measure is peg count, but one peg on the entry board is not
  enough: loading it into the carrier reveals a wrapped remote relay and one
  additional peg. Reward fires when that combined relay reaches one peg.
- Exhaustive clone probes found no long, diagonal, or color-15 bridge moves.

## Preserved result

- `solve_multi_bridge_wrapped_carrier_peg_solitaire` in `legs.py` is the
  clone-verified 102-action solution.
- `players.play_level_9` only composes that leg.
- `level9_candidate_102.json` preserves the exact action suffix.
- A joint symbolic search covered all 58 reachable bridge arrangements at
  carrier entry. No solution exists within 56 actions under the verified
  move rules; the shortest found and replay-verified path is 102 actions.
- Reaching level 9 under the current 600-action campaign cap therefore needs
  at least 46 actions removed from levels 1--8, or a newly verified mechanic
  outside the move rules above.
