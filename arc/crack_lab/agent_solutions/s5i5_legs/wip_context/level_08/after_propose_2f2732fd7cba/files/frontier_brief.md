# Unverified frontier brief: s5i5 level 8

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace and begin with the indexed summaries, then reproduce any prior observations with bounded clone probes before editing the level-8 composition.
- The baseline reliably reaches level 8’s start (`levels=7`, 267 moves). I’m now inspecting only the harness entry points and current player/leg interfaces needed to clone that exact state and summarize its coordinate-action objects.
- The one-step probes identify nine effective controls: four paired panel controls at the top, four at the bottom, and one central reorientation control. Their deltas match a larger recombination of earlier slider/arm mechanics, but no existing single leg covers all nine, so I’m testing the finite coordinate-control state graph directly for a reward-reaching path.
- The bounded search is still running without a result yet; clone expansion is slower than the target rate because deep clone chains accumulate. I’m giving this first bounded pass a little longer, then I’ll switch to replay-from-root with a compact symbolic key if it remains unproductive.
- The symbolic map resolves the level as the same nine-control, four-link dual-opening chamber family used by level 7: five upper panels and four lower panels with the same control topology, rearranged obstacles. I’m first testing the existing general leg unchanged, as required by the library reuse rule.
- The unchanged four-link leg correctly declined the layout: level 8 has only two lower panels, not four. The action deltas instead reproduce level 3’s crossed-slider system exactly—two slider controls (colors 10 and 7) plus four coupled-barrier controls (8, 9, 12, 14)—with its buttons relocated to both top and bottom. I’m generalizing that existing leg’s button discovery from “bottom row” to the observed small button signature, then reusing its verified sequence.
- The reused level-3 sequence changes the intended objects but does not complete this rearrangement, confirming it needs a new reusable staging leg rather than an inline level script. The dense state now exposes two docking measures: the external marker/target pair and the internal arm tip/center target. I’m measuring how every left/right control changes those two distances, then I’ll search over symbolic arm geometry instead of full pixels.
- The missing ingredient is now clear: level 8 adds a central reorientation control that level 3 never used. I’m searching from the verified crossed-slider staging state, varying that reorientation and reversible panel moves while preserving the already-docked external marker whenever possible. This is much smaller than searching all 64×64 frames from the start.
- The first randomized refinement pass has covered 600 bounded trajectories without a reward hit. That rules out a simple “finish with a few arbitrary moves” interpretation; I’m letting the fixed-budget pass complete, then I’ll use its reachable states to target the arm geometry rather than continue unguided sampling.
- The raw coordinates reveal the decisive dense objective: color 13 is a single marker at `(4,19)`, while its same-color hollow target is centered at `(43,19)`. The color-14 right control advances that marker downward in 3-pixel steps but is collision-blocked after the first step. I’ll now prioritize states by marker descent and retain alternate barrier configurations at each plateau.
- A single central quarter-turn changes the color-14 limb from west-facing to south-facing and allows the marker to descend from row 4 to row 13. It then meets the horizontal color-7 barrier exactly as the geometry predicts. I’m switching to a best-first clone search keyed by marker row, so it explores only barrier configurations that preserve the deepest verified descent.

## Preserved local probes

- `probe8.py` (1767 bytes)

Run or inspect the smallest relevant probe before writing another one.
