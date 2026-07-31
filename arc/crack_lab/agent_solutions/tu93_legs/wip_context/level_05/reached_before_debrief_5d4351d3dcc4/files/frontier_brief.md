# Unverified frontier brief: tu93 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay within the clean-room surface. I’ll begin with `solver_index.md`, then read `frontier_brief.md` if present, and use only targeted ranges plus compact clone probes.
- The frontier says level 3 was solved, but I’ll treat that only as a lead and reproduce the actual baseline now with the harness.
- Baseline is reproduced at level 4 (`levels_completed == 3`). I’ll now use the same documented callback surface as `gkm_try.py`: replay the validated path into a fresh environment, then report only level-4 color/components and one-step clone deltas for actions 1–4.
- The first experiment identifies the avatar as the color-9 3×3 object: only action 4 moves it (right) from the starting pocket. A color-12 3×3 object also moves downward on that same turn, so level 4 introduces a second agent that acts on successful player moves. I’ll map the maze and both agents’ turn-by-turn reachability next, using node coordinates rather than pixels.
- A bounded replay search found and reproduced a 17-move win. The dense progress signal is the color-8 gate count: it stays at one while the player times movement against the oscillating color-12 agent, then drops to zero on entry; only afterward does the goal become completable. I’ll package this as a general dynamic-maze leg that searches visible states while ignoring the depleting HUD row, and keep `play_level_4` as a one-line composition.
- The reusable leg is now in place: it searches to reduce the visible waypoint-color count first, commits that verified subgoal, then searches from the cleared-gate state to the level transition. I’m running the full solver and replay validator now.
- Full validation passes: `levels=4`, 64 total moves, replay valid. I’ll remove the temporary probe and do one final syntax and end-to-end check on the clean workspace.
- Implemented level 4 successfully. - Added a reusable dynamic-maze search leg with colored-gate progress tracking in [legs.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_tu93_budgeted_medium_screen/legs.py:314). - Added the composed level player in [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_tu93_budgeted_medium_screen/players.py:21). - Removed temporary probes. Final verification: ```text RESULT levels=4 moves=64 replay_ok=True err=None ``` Syntax compilation also passes.

## Preserved local probes

- `wip_glue_notes.md` (292 bytes)

Run or inspect the smallest relevant probe before writing another one.
