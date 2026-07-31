# Unverified frontier brief: ar25 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly within the workspace and begin from `solver_index.md`. I’ll use compact symbolic clone probes, infer level 5’s mechanics, then keep `players.py` as composition-only and place any genuinely new reusable behavior in `legs.py`.
- The existing solver reliably reaches the level-5 start. I’m now using that state only as a launch point for clone experiments: first compact components and one-step deltas for all seven actions, then context-dependent probes around whichever object actually responds.
- The context probes identify the control structure: `USE` cycles selection. On the second selection, left/right move the 15×15 colour-5 piece by one 3-pixel logical cell; before that, arrows move scanner bars instead. I’m reducing the frame to logical 3×3 cells now so I can score exact piece–silhouette overlap as dense progress rather than search on arbitrary pixel change.
- The exact silhouette fit is real but does not raise reward, even after both scan bars traverse it and after waiting through a full border cycle. I’m checking the other strong physical affordance now: the movable colour-5 object shares its colour with the bottom boundary, while colour 11 owns the right boundary. That suggests contact/connection may be the actual win condition rather than silhouette overlay.

## Preserved local probes

- `probe_current.py` (4401 bytes)
- `probe_l5.py` (799 bytes)
- `probe_level5.py` (6780 bytes)
- `probe_rewards.py` (1602 bytes)

Run or inspect the smallest relevant probe before writing another one.
