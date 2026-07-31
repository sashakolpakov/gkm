# Unverified frontier brief: tu93 level 8

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the clean-room surface, begin with `solver_index.md`, then reproduce any frontier observations through bounded probes before editing the level-7 composition.
- Baseline is reproduced: the solver completes six levels and stops at level 7 (`levels=6`, replay valid). I’m now using the harness’s clone entry point and compact perception summaries to identify the level-7 avatar and action semantics.
- The reproduced level-7 frame uses the same block-maze mechanics: color 9 is the avatar; actions 2 and 4 move it down/right from the start; 8 and 13 are direction-marked nodes; 14 is the goal. A bounded clone trial confirms the existing dynamic directional-waypoint leg reaches the reward without modification, so level 7 only needs a composition entry.
- Implemented `play_level_7(env)` in [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_tu93_budgeted_medium_screen/players.py:34), reusing the existing dynamic directional-waypoint leg. Validation: ```text RESULT levels=7 moves=141 replay_ok=True err=None ``` Python syntax checks also pass.

## Preserved local probes

- `wip_glue_notes.md` (292 bytes)

Run or inspect the smallest relevant probe before writing another one.
