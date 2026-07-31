# Unverified frontier brief: re86 level 8

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside this workspace and begin with the indexed API surface, then reproduce any prior observations before relying on them. After establishing the level-8 mechanics on bounded clones, I’ll keep the player as leg composition only and verify the full run with `gkm_try.py`.
- The baseline is reproduced: `levels=7`, so level 8 is the live frontier. I’m now validating the preserved level-8 claims with the smallest existing probe, while inspecting only the documented cloning/action interfaces and the relevant leg helpers.
- The compact probe confirms the inherited structure: two selectable 13×13 outlines, 3-pixel movement, eight colored ring centers, and contextual repaint/deformation at bordered stations. The preserved geometry search is still bounded and running; its dense criterion is exact target-bounding geometry rather than incidental pixel change.
- The first inherited deformation is now reproduced exactly: it yields the required 19×7 outline and paints it 11. A straight descent is not valid—it crosses the lower swatch barrier and repaints the shape 14—so I’m treating “preserve color while crossing the barrier” as a separate verified subgoal before placement. I’m also checking the second outline’s observable state after isolation, because parking it through obstacles partially deforms it.
- The targeted search confirms many exact placements with the right 19×7 geometry but the wrong paint after crossing; it has not yet found a color-preserving route in the first 5,000 observable states. I’ve expanded only this bounded search (not the full game state space), allowing longer detours and using exact geometry/color at the target as the acceptance condition.
- The longer detour search is still running within the 30,000-state cap; clone reconstruction is the expensive part, not uncontrolled branching. While it runs, the observed target placements show the geometry itself survives the barrier—the unresolved state variable is repainting—so any accepted route must deliberately revisit color 11 after the last conflicting swatch.
- The dead-end result is useful: once the outline is already painted 11 above the barrier, no exact 11-colored placement appears in 30,000 reachable observable states. I’ve therefore restarted from the undeformed isolated square, allowing the solver to choose deformation and repaint order jointly; it is already reaching all three required rectangle aspect ratios at the target, which validates the rectangle-state model.
- A direct two-shape reward test is underway. The first route cleanly places a 19×7 outline at the four 11-markers without parking or damaging the second; the second remains a pristine 13×13 square. The bounded search is now moving only that square and checking the real level reward, with exact 10×16 target geometry logged separately if encountered.

## Preserved local probes

- `probe_l8_routes.py` (2741 bytes)
- `probe_level8.py` (3591 bytes)
- `search_l8_first.py` (2196 bytes)
- `search_l8_second.py` (2651 bytes)
- `search_level8.py` (4168 bytes)

Run or inspect the smallest relevant probe before writing another one.
