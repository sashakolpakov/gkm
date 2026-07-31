# Unverified frontier brief: vc33 level 7

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and use only the indexed harness/perception surface. I’ll first inspect `solver_index.md`, then any `frontier_brief.md`, and run the requested baseline before designing compact level-6 probes.
- Baseline is confirmed: level 5 completes in 118 moves with a valid replay, so the unsolved state is level 6. I’ll now inspect only the small public harness/API definitions needed to obtain a level-6 clone, then produce symbolic component and click-delta summaries rather than raw frame dumps.
- The fresh level-6 frame has one marked object at the upper left (color 4 with a color-11 tip), a matching color-11 target embedded in the central barrier, and four cyan control-like pads around two blocked horizontal passages. Component-center clicks are inert, which is useful negative evidence but not enough: several large connected regions hide narrow edge controls. I’ll scan a bounded 4-pixel coordinate lattice (256 clone steps) and group identical effects symbolically.
- The effect grouping reveals the same core mechanic as the existing pressure-gate leg: the larger color-11 marker is the moving platform, the smaller marker is fixed, and two closed gates lie strictly between their columns. Clicking cyan pads advances chamber boundaries/platform pressure in three-cell increments. This matches `cross_pressure_gates_then_align_height`’s verified observational contract despite the new geometry, so I’ll first compose that existing leg for level 6 and test it unchanged.
- The existing leg makes 20 meaningful level-6 moves but does not complete it, so the mechanic overlaps but its stopping metric is too narrow for this layout. I’ll inspect the post-leg symbolic state—marker positions, remaining gates, and active gate colors—to identify the minimal generalization instead of adding level-specific coordinates.
- The bounded continuation search solves the clone in 8 more actions and exposes the missing dense metric: after all gates are crossed, progress must reduce the marker pair’s Manhattan gap, not require horizontal position to remain frozen while reducing height. I’ll generalize the existing leg’s post-gate metric accordingly; this keeps level 6 as pure composition and should preserve earlier behavior because gate-crossing priority remains unchanged.
- The failed generalization exposed an orientation issue: this level’s separating gates are horizontal, whereas the existing leg’s gate-progress test is deliberately column-oriented. I’ll preserve that proven leg and add one small reusable finisher: bounded visible-control search that repeatedly commits only paths reducing the unequal marker pair’s Manhattan distance (or completing the level). Level 6 will compose the existing crossing leg followed by this dense-progress alignment leg.
- Level 6 now completes: `RESULT levels=6 moves=148 replay_ok=True err=None`. I’m doing one final targeted syntax check and reviewing only the two edited solver files to ensure the player contains composition only and the new leg has no level-specific coordinates.
- Implemented level 6 successfully. - Added reusable `align_marker_pair_with_pressure_controls` leg using bounded BFS and Manhattan-distance progress. - Added `play_level_6` as composition-only logic. - Preserved the compact discovery probe in [probe_level6.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_vc33_budgeted_medium_screen/probe_level6.py). Final verification: ```text RESULT levels=6 moves=148 replay_ok=True err=None ``` Syntax checks also pass for [legs.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_vc33_budgeted_medium_screen/legs.py) and [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_vc33_budgeted_medium_screen/players.py).

## Preserved local probes

- `probe_level4.py` (13644 bytes)
- `probe_level5.py` (4716 bytes)
- `probe_level6.py` (3765 bytes)

Run or inspect the smallest relevant probe before writing another one.
