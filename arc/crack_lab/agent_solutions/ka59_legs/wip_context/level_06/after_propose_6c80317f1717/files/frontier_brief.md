# Unverified frontier brief: ka59 level 6

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace boundary, begin with the indexed API surface, then reproduce any preserved observations before probing level 6. I’ll keep probes compact and only commit a leg-composed player after verifying the mechanic on clones.
- Baseline reproduces cleanly at level 5 (`159` moves, replay valid). Next I’m using the harness to advance a clone through the existing players and summarize level 6 by components plus one-step deltas; I won’t rely on the frontier’s level-5 claims beyond what this run confirmed.
- The level-6 clone shows the selected movable object experimentally: the hollow color-14 3×3 ring at center `(52,49)` shifts under arrows; coordinate action on every other candidate leaves the movement effect unchanged, so those objects are not selectable in this state. A horizontal barrier at rows `30–35` has only a three-cell central gap, while the plausible small-ring target is centered near `(49,22)`. I’m now testing the route through that gap and tracking ring-to-target distance as the dense measure, while accounting for three objects that animate on every command.
- The naïve gap route stalls exactly where the moving 6×6 corridor piece occupies the opening; the ring’s distance improves from `(52,49)` to `(34,37)` but cannot enter the barrier. This confirms the animated piece is mechanically relevant rather than cosmetic. I’ve tightened the search key to only the persistent piece geometries and am comparing wait/push timings around that blocked state.
- The barrier crop resolves the apparent contradiction: the three-cell “gap” is actually filled by a color-2 vertical wall, so the ring cannot cross by ordinary movement. I also found a flaw in my first selection probe—the coordinates were blob centers, which land in the holes of ring-shaped objects. I’m correcting it to click confirmed colored pixels; that will determine whether the large upper ring or multicolor blocks are controllable and which side of the sealed barrier each can reach.
- Repeated pressure confirms the lower corridor piece has three discrete staging positions and a hard left endpoint; no reward or cross-barrier transfer occurs merely by holding left/up. I’m switching from pixel-state BFS to a bounded macro search: three identical commands form one complete movement/animation cycle, so the key can ignore transient colors and retain only persistent ring/block geometry. This directly searches cooperative staging sequences without paying for thousands of deep runtime clones.
- The macro search explored 1,500 distinct persistent configurations without a win, but completed within the probe budget. I’m instrumenting it with a goal-shaped score—the sum of the small ring’s distance to the small color-4 outline and the large upper ring’s distance to the large outline—so the next bounded run yields a verified subgoal path instead of another sparse failure.

## Preserved local probes

- `probe_l4.py` (3341 bytes)
- `probe_l5.py` (3037 bytes)
- `probe_l6.py` (5807 bytes)

Run or inspect the smallest relevant probe before writing another one.
