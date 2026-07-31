# Unverified frontier brief: wa30 level 9

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Verified parent budget

- Exact parent boundary: level 8 at 587 actions.
- Remaining real-action budget under the harness cap: 13 of 600.
- This budget is verifier evidence. If it is insufficient, optimize earlier composed legs rather than searching an uncommittable suffix.

## Prior proposer progress

- I’ll stay strictly inside this workspace and the documented `gkm_try.py`/`perception.py` surface. I’ll begin from `solver_index.md`, reproduce any inherited observations on fresh clones, then encode only reusable behavior in `legs.py` and keep `players.py` as composition.
- There is no inherited frontier brief, so every mechanic will be established from this run. I’m taking the required pristine baseline now, then I’ll inspect only the indexed perception helpers needed for compact probes.
- The baseline confirms level 1 is untouched (`levels=0`, `moves=0`). I’ve also confirmed the only supplied player/leg files are empty scaffolds. Next I’m adding a small observational probe that uses `env.frame()`, `env.clone()`, `env.step()`, and the supplied perception summaries only; it will not commit real moves.
- The first clone experiment identifies the avatar cleanly: the black/orange 4×4 object at macro-cell `(12,8)` moves one cell under actions 1–4; action 5 is inert in open space. Three matching 4×4 objects and one three-cell horizontal strip remain fixed during ordinary moves. I’m now testing contact from multiple sides and `USE` while adjacent, including whether the strip is solid or traversable.
- Contact behavior is now verified from three directions: facing an adjacent square changes its rim from 4→3 (selection), and `USE` changes that rim to black (attachment). While attached, moving away carries the square one cell in lockstep; a second `USE` releases it. The three-cell strip is traversable, making “place three carried squares onto its three cells” the strongest goal hypothesis. I’m testing one placement now and measuring progress by occupied strip cells plus Manhattan distance of the remaining square centers.
- The full hypothesis now solves a pristine clone in 33 actions. Dense progress advances exactly `0→1→2→3` occupied strip cells and assignment distance `8→6→3→0`; reward remains 0 until the final carried square is released, when `levels_completed` becomes 1. I’m encoding this as one reusable observational leg: a small symbolic planner for the verified “face, attach, translate, release” mechanic, rather than baking the discovered coordinates into the level player.
- The first generic planner exhausted its 250k symbolic-state cap without finding the already verified 33-step solution; the mechanic is sound, but uninformed breadth-first search wastes effort on irrelevant open-field walking. I’m switching that same bounded planner to the dense measure already validated—target occupancy/assignment distance, attachment cost, and distance to the next block—then re-verifying its emitted path on a fresh clone.
- The dense-guided planner now produces a 32-action route—one move shorter than the manual witness—and that exact emitted route reaches level 1 on a fresh clone. I’m running the actual harness next; if replay validation passes, I’ll do a final source/behavior sanity check without repository-wide inspection.

## Preserved local probes

- `probe_interactions.py` (1663 bytes)
- `probe_level1.py` (2041 bytes)
- `probe_placement.py` (1117 bytes)
- `probe_solution.py` (2372 bytes)
- `probe_tiles.py` (1126 bytes)
- `verify_planner.py` (642 bytes)

Run or inspect the smallest relevant probe before writing another one.
