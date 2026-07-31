# Unverified frontier brief: tu93 level 9

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and begin with the indexed file ranges, then reproduce any inherited observations on bounded clones before changing the solver.
- Baseline is confirmed at seven completed levels. I’m now inspecting only the harness entry points and the existing dynamic-maze interfaces needed to enter level 8 and summarize its frame/actions compactly.
- The level-8 frame reproduces the familiar 3×3 block-maze family: color 9 is the only object moved by an available action at the start, color 14 is the goal, and the visible direction-marked waypoint uses colors 8/15. I’ll now test the existing reusable directional-waypoint leg entirely on a clone before deciding whether any new leg is necessary.
- The clone reached reward level 8 using the existing leg with a 300-state bound, so no new mechanic or leg was needed. I’ve added only the level-8 composition and am validating the full replay now.
- Implemented `play_level_8(env)` by reusing `drive_dynamic_directional_waypoints` in [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_tu93_budgeted_medium_screen/players.py:39). Validation passed: ```text RESULT levels=8 moves=166 replay_ok=True err=None ``` No new leg was required; syntax checks also pass.

## Preserved local probes

- `probe_l8.py` (1713 bytes)
- `wip_glue_notes.md` (292 bytes)

Run or inspect the smallest relevant probe before writing another one.
