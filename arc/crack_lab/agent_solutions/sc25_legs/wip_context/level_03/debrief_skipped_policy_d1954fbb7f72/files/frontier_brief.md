# Unverified frontier brief: sc25 level 3

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside this workspace and use only the documented harness/perception surface. I’ll first read the index and prior frontier brief if present, then run the baseline and probe level 3 with compact symbolic summaries before editing the leg library and player.
- The baseline is stable: the current solver reaches level 3 (`levels=2`, 22 moves). I’m now reproducing the frontier’s level-3 observations from the preserved compact probe, especially avatar controls, the coordinate panel, barrier/exit geometry, and whether action 6 has a context-dependent effect.
- The compact frame resolves the geometry: the 9/10 avatar starts in the upper playfield; actions 1/2/3/4 are up/down/left/right. A color-13 barrier blocks the route to the matching 9/10 exit, while a color-4 obstacle blocks access to a color-6-ring/color-13 device on the right. The lower 3×3 panel is a local coordinate grid, and its separate clue shows three color-6 marks in the middle column. I’m probing panel state cycles and interactions at both obstacles now.
- The panel clue is now verified: selecting its three middle-column cells causes the marked panel to reset to a uniform state, which is a real accepted-pattern transition rather than three independent toggles. The right-side obstacle can be bypassed only by moving fully above or below it; my first route was one row too close. I’m correcting that route and testing whether contact with the enclosed color-13 device changes the barrier’s latent state, then returning to the exit in the same clone.
- A stronger interpretation emerged from the accepted-pattern reset: the panel behaves like a reusable command pad. Completing the clue once clears the marks, so a single submission may only advance a hidden mechanism by one unit. I’m testing repeated submissions of the verified middle-column pattern and measuring barrier/device positions after each round, using their displacement/removal as dense progress before checking the exit.

## Preserved local probes

- `probe_level2.py` (7657 bytes)
- `probe_level2_interactions.py` (4048 bytes)
- `probe_level3.py` (4545 bytes)
- `probe_level3_compact.py` (3022 bytes)
- `probe_level3_device.py` (2400 bytes)
- `probe_level3_push.py` (2331 bytes)
- `probe_level3_repeat.py` (1189 bytes)

Run or inspect the smallest relevant probe before writing another one.
