# Unverified frontier brief: su15 level 8

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay strictly inside the workspace and begin with `solver_index.md`, then use only the documented harness/perception surfaces and bounded clone probes. I’ll first reproduce the current level state, then isolate level 6’s mechanics before changing the leg library.
- The baseline reproduces exactly: level 5 completes in 98 moves, so level 6 is the active clone frontier. I’ll now inspect only the harness entry points and compact probe patterns needed to advance a clone and summarize level 6; no prior observation will be treated as evidence until reproduced.
- Level 6 is coordinate-only. The initial symbolic frame shows two large ring targets in separate regions, two decorated 4-cell objects, and a six-color palette-like strip at the top. A bare `step(6)` is inert, confirming coordinates are essential. Next I’m testing clicks on each playfield object, each ring, empty space, and each palette swatch to identify the avatar/action semantics by consistent deltas.
- The repeated probes reproduce the central hazard: the small decorated color-7 body advances every click and destroys the 7×7 color-8 square on contact. Clicking the small body freezes it; clicks can also reposition the large square in some contexts. That makes “square survives” a necessary invariant, while distance from the square to a ring is the first dense-progress candidate. I’m mapping the large square’s valid one-click destinations next, before testing either ring as a goal.
- Placing the color-8 square into either ring is reproducible partial progress—the ring’s visible color-9 area drops by exactly the square’s 49 cells—but reward does not fire. This rules out “choose either ring.” The likely next mechanic is that the moving body must transform or split the square so both rings can be occupied; I’m now measuring the collision products by color and component, rather than treating the disappearance as simple failure.
- The transformation ladder is now clear and repeatable: collision cuts `8(7×7) → 12(5×5) → 11(4×4) → 15(3×3) → 6(2×2) → 10(1×1) → gone`, ending the level without reward. So cutting is a failure path, not the objective. A stronger model fits all action deltas: every click steers the small avatar toward that coordinate by about four cells, while the large square is repositioned only when the click lies within its local 8-cell reach. With two rings and two persistent bodies, I’m testing simultaneous occupancy—park the square in one ring, then steer the avatar into the other—while preserving the square.
- The cooperative hypothesis is confirmed by reward: staging the large square in the farther (upper-left) ring, then continuing the moving body’s established diagonal trajectory, advances level 6 on the seventh click. The mirrored lower-right attempt fails because the body repeatedly cuts the square against the boundary. No existing leg expresses “stage a locally draggable square in a ring, then advance an autonomous diagonal body,” so I’m adding one small geometry-driven leg: detect the largest solid square and both rings, choose the farther ring, move in bounded increments, then click the alternate ring only to let the moving body finish. `players.py` will contain only that leg call.
- The new leg succeeds end-to-end: `RESULT levels=6 moves=105 replay_ok=True err=None`. I’m doing two final checks now: deterministic replay/syntax, and trimming the exploratory level-6 probe down to a compact symbolic reproduction of the seven-click reward path so future runs don’t emit the large discovery trace.
- The production solver is deterministic across both final runs and syntax checks pass. The compact probe exposed one maintenance issue: now that level 6 is dispatched, calling the full solver inside the probe advances past the intended observation point. I’m correcting the probe to replay only levels 1–5 before applying its preserved level-6 path, then I’ll run that one last time.
- Implemented and verified level 6. - Added reusable cooperative staging leg in [legs.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_su15_budgeted_medium_screen/legs.py:454). - Added composition-only level dispatch in [players.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_su15_budgeted_medium_screen/players.py:28). - Preserved a compact symbolic reproduction in [probe_level6.py](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_su15_budgeted_medium_screen/probe_level6.py). Validation passed twice: ```text RESULT levels=6 moves=105 replay_ok=True err=None ``` Syntax checks and the independent seven-click level-6 probe also passed.

## Preserved local probes

- `probe_dots.py` (903 bytes)
- `probe_level4.py` (5557 bytes)
- `probe_level5.py` (1909 bytes)
- `probe_level5_sequences.py` (2002 bytes)
- `probe_level6.py` (1230 bytes)
- `probe_route.py` (881 bytes)
- `probe_search.py` (2205 bytes)

Run or inspect the smallest relevant probe before writing another one.
