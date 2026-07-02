# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## wa30 level 1 (solved, 32 moves)
Warehouse/sokoban-carry level. Discovered mechanics:
- 4px logical grid; avatar = colour 14 + colour 0 nose, moves exactly 4px/step.
- Boxes (colour-4 border, colour-9 interior) are moved by GRAB (touch + action 5),
  then carried rigidly, then RELEASE (action 5); released box border -> colour 3.
- Target container: colour-9 border, colour-2 interior. Reward fires only when
  ALL target cells are filled (sparse; no partial credit for 1 or 2 boxes).
- Key gotcha: the grab leaves a 1px vertical carry offset that varies with the
  approach, so exact placement needs CALIBRATION (probe a release on a clone,
  measure box-vs-foot offset, then position precisely). -> leg `place_carried`.

New general legs added: foot, clusters, target_cells, move, goto (collision-aware),
grab_from_below, place_carried, carry_box_to_cell. Player just assigns boxes to
cells by column and composes carry_box_to_cell. These should transfer to later
levels (walls/helper) by reusing goto + carry with relay logic.

## wa30 level 2 (solved, ~69 L2 moves)
Same warehouse-carry world, now with (a) an autonomous HELPER agent (colour 12)
and (b) a HARD per-level step budget (70 moves; game `lose()`s when the counter
hits 0). Win condition (read from the engine): EVERY box's top-left rests on a
target cell AND none is currently carried. No walls; the container interior is
free space; there is NO gravity -- a released box stays exactly where dropped.

Decisive discoveries (all by experiment on clones):
- Grab makes a 2-cell-tall composite: the box rides ABOVE the avatar (border
  recolours 0, interior 9), so `foot()` reports the BOX's top-left while carried.
  Carrying therefore navigates fine with the normal `goto` (composite colours
  {0,9,14} aren't in the blocked set); the earlier "carry is slow" was a
  self-inflicted detour, not a mechanic.
- Stacking constraint: to drop a box into slot (row,col) the avatar must stand in
  the cell just below it (box rides above). Only the LOWEST free slot of a column
  is cleanly reachable (avatar parks in the free cell beneath / below the whole
  container). That's `deliver_to_bin` -> lowest_free_slot.
- The helper alone tops out at 4/5 boxes within 70 moves. It acts on EVERY avatar
  move, picking the nearest free box and pathing it to a target. Cooperation is a
  throughput sum: the avatar personally delivers 2 boxes to bottom slots, then
  PARKS and idles so the helper finishes the remaining 3 -> 5/5 -> reward. Grabbing
  more than needed just steals the helper's boxes and stalls at 4 (the helper
  always leaves its last box in transit at timeout).

New general legs: bin_rows, bin_columns, slot_occupied, lowest_free_slot,
deliver_to_bin, and the higher-order cooperative loop `fill_bin_with_helper`
(quota = boxes the avatar delivers before yielding to the helper). play_level_2 is
just `fill_bin_with_helper(env, box_color=4, quota=2)`.

Gotcha for future players: `levels_completed` is NON-zero on levels > 1, so loop
guards must compare against the level at entry (lvl0), not `not env.levels_completed`.

## Recurring composition pattern (candidate higher-order leg)
Note: players.py still has only ONE player, so there is no cross-player duplication
to dedupe yet. The pattern below is the composition *inside* play_level_1, hoisted
so it is written once and the player is a true one-liner.

Pattern observed: **match a set of movable sources to a set of fixed sinks under a
shared sort key, then transport each pair with a single transport skill.**
  cells  = sorted(target cells, key)
  boxes  = sorted(sources,      key)
  for src, sink in zip(boxes, cells): transport(src, sink)

Captured as higher-order leg `fill_targets(env, box_color, interior, key)`, which
closes over carry_box_to_cell as the transport step. play_level_1 is now just
`fill_targets(env, box_color=4)`.

Generalization to watch for on later levels: the transport step is a parameter.
When walls/helpers/relays appear, the same assign-then-transport skeleton should
take a different transport callable (e.g. a relay-carry or push), suggesting a
fully generic `assign_and_transport(env, sources, sinks, key, transport_fn)` once
a second concrete transport exists to justify writing it.

## wa30 level 3 (solved, 198 moves total; replay_ok)
Relay warehouse: a floor-to-ceiling wall (colour 2) splits the map. The avatar and
its boxes are on one side; an autonomous HELPER (colour 12) works the far side,
ferrying boxes from the wall into the container but unable to reach the avatar's
deeper boxes. Solution: the avatar grabs each box from its LEFT (new mirror-grab
`grab_from_left`, box then rides on the RIGHT at offset (0,+4)), drops it flush
against the wall column (`wall_col` picks the busiest 4-aligned column band so a
tall wall isn't confused with the container interior), waits for the helper to
collect it, then idles so the helper finishes. play_level_3 is just
`relay_to_helper(env, box_color=4, wall_color=2)`.

Debrief dedupe (this pass): now that TWO cooperative players exist
(`fill_bin_with_helper`, `relay_to_helper`) the concrete duplication was hoisted so
each skill is written ONCE:
- `in_progress(env, lvl0)` -- the `not terminal and levels_completed <= lvl0`
  loop-guard (was inlined ~5x across both legs; also the level-2 lvl0 gotcha).
- `nearest_to_foot(env, boxes)` -- the identical Manhattan `min(... foot ...)`
  next-box pick that both legs used.
- `yield_to_helper(env, lvl0, a=1)` -- the identical trailing idle loop that hands
  the floor to the helper until the level completes.
The players stay thin one-liners; the shared legs are now the single source of truth.

## Recurring composition pattern (candidate higher-order leg)
Second pattern, now visible across play_level_2 and play_level_3 (the first,
assign-then-transport, still stands for level 1). Both cooperative players share
one skeleton -- **do a bounded amount of your own work handing boxes to a
helper-reachable zone, then get out of the way and idle until the helper finishes:**

  lvl0 = env.levels_completed
  while own_work_remains() and in_progress(env, lvl0):
      handoff_one()          # deliver_to_bin (into a slot) | relay to the wall band
  step_aside()               # park / already-out-of-way
  yield_to_helper(env, lvl0) # idle; helper acts on every avatar step -> reward

The variable part is `handoff_one` (and its stop condition): level 2 hands boxes
into the container's lowest free slots up to a quota; level 3 relays boxes flush
against the wall, waiting per-box for the helper to pick them up. This is the
cooperative analogue of assign-and-transport: the transport target is the HELPER's
reach rather than the final sink, and the loop ends by yielding instead of
finishing personally.

Candidate leg to hoist once a THIRD cooperator appears:
`cooperate_then_yield(env, handoff_fn, done_fn=None, park=None)` closing over the
per-level handoff. Deferred for now (only two concrete cooperators, and their
handoff loops differ enough -- quota counter vs. per-box wait-for-pickup -- that a
premature merge would obscure more than it saves); the shared primitives above
(`in_progress` / `nearest_to_foot` / `yield_to_helper`) already remove the real
duplication.
