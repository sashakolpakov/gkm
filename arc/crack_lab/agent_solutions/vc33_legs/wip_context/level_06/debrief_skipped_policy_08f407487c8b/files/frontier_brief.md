# Unverified frontier brief: vc33 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay within the clean-room surface, begin with `solver_index.md`, then reproduce any prior observations before relying on them. I’ll inspect only targeted ranges and use compact clone probes.
- Baseline is reproduced at level 4 (`moves=54`, replay valid), so the current target is indeed level 5. The prior brief concerns the now-solved pressure-gate mechanic; I’ll treat its claims as untrusted and derive level 5 from fresh clones.
- I’m creating a small level-5-only probe that replays the validated checkpoint, reports compact components, and tests every visible component center once. This will identify the clickable affordances and moving object without assuming that level 4’s controls carry over.
- Fresh evidence shows two independently marked platforms: a color-14 platform above three horizontal gates and a color-11 platform below them, with their smaller matching targets on the opposite sides. The five effective buttons shift chamber pressure/platform columns by three cells; direct clicks elsewhere are initially inert. I’m now testing the dense subgoal “cross one intervening gate, then reduce final column mismatch” for each marker color, including only visible buttons and gates that visibly activate.
- The hypothesis is partially verified: visible pressure activation plus gate-clicking moves each marked platform across a barrier. The color-11 platform can cross both barriers and then each button press reduces its target-column gap by exactly three. The first attempt moved color 14 first and left its second crossing outside the 1,200-state bound, so I’m checking the opposite order with a slightly larger but still bounded search.
- The opposite-order search is still running within its finite state cap. It is slower because each node replays the validated 54-move prefix; I’m letting this bounded probe finish rather than expanding the search space further.
- The opposite order reaches level 5 on a clone. The verified sequence is cooperative: the lower color-11 platform crosses upward first and aligns its column; this pressure staging lets the upper color-14 platform cross downward, align, and trigger completion. I’m encoding that as one reusable “opposing platforms across horizontal pressure gates” leg, with the player only calling the leg.
- Implemented and verified level 5. - Added reusable horizontal pressure-gate leg in [legs.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_vc33_budgeted_medium_screen/legs.py) - Added level-5 composition in [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_vc33_budgeted_medium_screen/players.py) - Preserved the bounded discovery probe in [probe_level5.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_vc33_budgeted_medium_screen/probe_level5.py) Final verification: ```text RESULT levels=5 moves=118 replay_ok=True err=None ```

## Preserved local probes

- `probe_level4.py` (13644 bytes)
- `probe_level5.py` (4716 bytes)

Run or inspect the smallest relevant probe before writing another one.
