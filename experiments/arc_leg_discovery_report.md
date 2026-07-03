# Learning the legs from the game, then forming the cone over them

The cone program had learned the **cone** over a fixed leg library, the **goal**
from reward, and classifier **macros** over given atoms — but never the **leg
bodies** themselves. On ARC-AGI-3 you can neither hand-build them (the right
primitives are game-specific) nor evolve them against the extrinsic reward
(`levels_completed` is never reached by a random policy → no gradient). This
adds the missing piece: learn legs from interaction with an **intrinsic** signal,
then compose the **cone over the learned legs**.

```bash
python3 experiments/run_arc_leg_discovery.py --game ls20 --mode offline
python3 experiments/run_arc_leg_discovery.py --game wa30
python3 -m pytest tests/test_cone_leg_discovery.py -q
```

## Method

- **Intrinsic signal (`cone_leg_discovery.py`).** A fragment is good if, from a
  reproducible state, repeating an action drives the avatar (found by action
  response, not hardcoded) in a *consistent, replicable* direction. The learned
  leg names a **displacement, not a colour** — channel-blind, so naturality is
  preserved and the leg can later be `CALL`ed bound to any colour slot.
- **Two controls, in the project's style.** (1) a **random-action floor**: a
  learned leg must beat the best-direction consistency a random option reaches
  by chance; (2) **held-out replication**: consistency/efficacy are scored on
  states not used to propose the leg, and legs that fail are reported, not hidden.
- **Cone over the learned legs (`cone_leg_composition.py`).** A channel-blind
  seeker reads the azimuth to the bound goal colour and `CALL`s the learned leg
  whose direction best closes the gap. It can only use directions that were
  *discovered* to be controllable — robustness the hand-built witness lacks.

## Results

**Open navigation (hermetic stub) — the pipeline closes.** Discovery learns all
four cardinal legs (consistency 1.0 each; random floor ≈ 0.4), and the cone
composed *purely from the learned legs* reaches **WIN**, matching the hand-built
`witness_seek_leg`. This is the end-to-end proof: legs learned from interaction,
cone formed over them, goal solved. (`tests/test_cone_leg_discovery.py`.)

**ls20 (real local frames) — learns the one real primitive.** Avatar = colour 12.

```text
random-action control floor: 0.3
LEARNED  move_up_until_blocked   dir=(0,-1) consistency=1.0 efficacy=0.8 mean_step=21.5
rejected move_right_until_blocked consistency=0.6 efficacy=0.0   (blocked: never moves)
rejected move_down_until_blocked  consistency=0.4 efficacy=0.0
rejected move_left_until_blocked  consistency=0.0 efficacy=0.0
```

Exactly one direction is reliably controllable; the rest are correctly pruned.
The cone over that single leg steers the block but completes no level (random
also 0/7): ls20 is a pattern-matching slide puzzle, so a 1-leg directional
library is genuinely insufficient — the honest gap is the *next* fragment.

**wa30 (real local frames) — honest null.** Avatar = colour 14, but no
directional leg clears the bar (UP reaches 0.8, below 0.85; the others are
inconsistent — the avatar does not cleanly translate in 3 of 4 directions). The
verdict states plainly that a directional library is the wrong primitive set
here (wa30 is Sokoban-like; it needs push/slide), and reports the discovery +
controls as the result rather than forcing a leg.

## What this establishes, and the next fragment

- The program can now **learn channel-blind legs from a game** with controls, and
  **form the cone over the learned legs** — demonstrated to WIN on open
  navigation and to recover the genuinely-reliable primitives on real ARC frames.
- The remaining gap is **leg expressiveness**, made precise: directional
  "move-until-blocked" is the first effect family; the puzzles need **push**
  (the discovery already flags colours that rigidly co-move with the avatar as
  push candidates), **slide-to-target**, and **align**. Each is the same
  intrinsic-effect recipe with a richer option/precondition, and once in the
  library the existing `arc_goal_induction` + `evolve_bound_task` machinery
  glues them toward the induced level goal — priced by free energy, floored by
  the random control.
