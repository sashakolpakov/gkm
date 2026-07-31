# Unverified frontier brief: tu93 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay within the workspace boundary, begin from `solver_index.md`, then reproduce only the observations needed to reach and probe level 5.
- Baseline is reproduced at level 5 entry (`levels_completed=4`). I’m now inspecting the preserved probe note and the narrow harness/API surfaces needed to create a fresh level-5 clone without relying on the previous attempt’s claims.
- The first controlled action test identifies color 9 at `(14,51)` as the avatar: action 3 moves it one node left, while blocked actions only consume the visible budget. Four color-12 objects move simultaneously on that successful turn, so this is a cooperative/dynamic maze configuration with a color-14 goal. I’m reducing the frame to parsed maze nodes and testing whether the existing dynamic-maze leg can solve this configuration unchanged.
- The existing general leg fits unchanged. On a bounded clone, using color-12 count as dense progress solved the level and raised `levels_completed` from 4 to 5; the resulting frame is level 6 (so its remaining color-12 pixels are unrelated). I’ll add the one-line level-5 composition, remove the temporary probe, and run end-to-end replay validation.
- Implemented `play_level_5(env)` in [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_tu93_budgeted_medium_screen/players.py:25), reusing the existing dynamic-maze leg with color 12 as the dense progress marker. Validation passed: ```text RESULT levels=5 moves=93 replay_ok=True err=None ``` Syntax compilation also passes.

## Preserved local probes

- `wip_glue_notes.md` (292 bytes)

Run or inspect the smallest relevant probe before writing another one.
