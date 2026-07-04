# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## bfs_to_level_up (legs.py)
Generic blind BFS over env clones, deduping states by frame bytes with an
auto-detected HUD/counter mask (probes the least-changing action, repeats it
to reveal cells that drift every step regardless of movement/action, e.g. a
"moves used" countdown bar). Commits the shortest winning path to the real
env. Used for level 1 (ls20): a 4-direction avatar sliding on a 5px lattice
through a maze that LOOKS solid (color 3) but is actually mostly open track;
a small stationary "switch" object toggles a HUD flag when stepped on, and
reaching a specific chamber after the switch completes the level. No
level-specific logic was hardcoded into the leg -- it should generalize to
other small-state-space levels in ls20 as long as HUD noise is the only
non-deterministic-looking part of the frame.

## Debrief: comparing play_level_1 to earlier players (cross-workspace)

`players.py` here has only one player so far (`play_level_1`), and it is
already a single call into `bfs_to_level_up` -- no repeated code within this
workspace to extract. Diffing against the sibling leg workspaces
(gkm_legs_ws_sp80, gkm_legs_ws_wa30) to look for a cross-game pattern instead:

- sp80's `shift_and_confirm(env, move_action, n, confirm_action)`: move n
  steps, then press one separate "lock it in" action.
- wa30's `move_interact(env, *steps)`: walk a sequence of (direction, times)
  steps, then press one separate "interact" action.
- ls20's `bfs_to_level_up(env)`: try candidate action sequences on *clones*,
  and only commit the one sequence that succeeds to the real env.

**Recurring shape:** every one of these legs is "propose/attempt a sequence
of provisional actions, then gate on a single decisive action or check
before it counts." sp80/wa30 hardcode the decisive action (confirm/interact)
as the last step of a known-good sequence; ls20 generalizes the same shape
by making the whole sequence provisional (on a clone) and the "decisive
step" the success check (`levels_completed` increased) rather than a fixed
action id.

## play_level_2 (players.py)
Level 2's frame is a much larger, more elaborate maze (multiple nested
chambers of color 3 walls, colored "rooms" in 5/9/11/8/12/0/1) but the same
`bfs_to_level_up` leg solved it with zero level-specific code -- confirms
the leg's HUD-mask + blind-BFS-with-frame-dedup approach generalizes across
very different level geometries in ls20, not just superficial variants of
level 1. No new leg was needed.

## play_level_3 (players.py)
Same `bfs_to_level_up` leg, zero new code, solved level 3 in one shot
(97 moves, replay_ok). Third level in a row where blind BFS + HUD-mask
dedup is sufficient -- ls20 levels so far are all small-state-space mazes
distinguished only by geometry/color layout, not by new mechanics, so no
new leg was warranted.

**Candidate higher-order leg:** `attempt_then_commit(env, propose, is_win)`
-- where `propose` yields one or more candidate action sequences (either a
fixed script like shift/move-interact, or a generator like BFS/DFS), and
`is_win(env_after)` decides whether that sequence's effect should be kept.
Fixed-script legs (`shift_and_confirm`, `move_interact`) would become
`attempt_then_commit` calls with `is_win` always true and `propose`
returning exactly one sequence; search legs (`bfs_to_level_up`) would supply
a `propose` that enumerates many candidates and an `is_win` checking level
progress. Not extracted yet since it would require touching two other
workspaces' legs.py for a pattern only 3 instances deep -- worth revisiting
if a 4th leg (in any game) repeats the shape.

## play_level_4 (players.py)
Same `bfs_to_level_up` leg again, zero new legs. The only change needed was
raising `max_nodes` from the default 6000 to 30000 in the `play_level_4`
call -- at the default budget BFS ran out of nodes before finding the
level-up transition (level 4's state space/solution path is larger, 140
total moves vs 97 for level 3). Confirms the leg's search itself
generalizes across all 4 levels seen so far; the only per-level knob ever
needed is the node budget, not new logic. Suggests a future refinement of
`bfs_to_level_up` could auto-escalate `max_nodes` (e.g. retry with 5x budget
on failure) instead of callers hardcoding a bigger number, but not worth
doing until a level actually needs >30000 or auto-escalation avoids
real repeated tuning.
